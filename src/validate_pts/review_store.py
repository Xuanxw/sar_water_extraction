"""核验点读取和属性保存；只修改人工字段，保留几何、抽样权重和其他图层。"""
from datetime import datetime, timezone
from contextlib import closing
from pathlib import Path
import sqlite3

import fiona
from pyproj import Transformer
from shapely import wkb

FIELDS = ('ref_class', 'confidence', 'note', 'reviewer', 'reference', 'review_date')


def gpkg_geometry(blob):
    """供GeoPackage自身的RTree触发器调用；不删除触发器或改写几何。"""
    if blob is None:return None
    # GeoPackage头为8字节，随后是可选包络，再后面是标准WKB。
    envelope = (blob[3] >> 1) & 7
    offset = 8 + {0:0, 1:32, 2:48, 3:48, 4:64}[envelope]
    return wkb.loads(bytes(blob[offset:]))


class ReviewStore:
    """加载已有统一GPKG；每次保存一个点，SQLite事务保证中断不留下半次写入。"""
    def __init__(self, path):
        self.path = Path(path)
        self.layer = 'verification_points'
        self.points = []
        with fiona.open(self.path, layer=self.layer) as src:
            required = {'point_id', 'scene_id', 'pred', *FIELDS}
            if not required.issubset(src.schema['properties']):
                raise ValueError('请选择汇总后的manual_review.gpkg，缺少核验字段')
            to_map = Transformer.from_crs(src.crs, 'EPSG:3857', always_xy=True)
            to_wgs = Transformer.from_crs(src.crs, 'EPSG:4326', always_xy=True)
            for feature in src:
                if feature['geometry']['type'] != 'Point':raise ValueError('核验图层必须为点')
                p = dict(feature['properties'])
                x, y = feature['geometry']['coordinates'][:2]
                p['map_xy'] = to_map.transform(x, y)
                p['lonlat'] = to_wgs.transform(x, y)
                self.points.append(p)
        self.by_id = {p['point_id']:p for p in self.points}
        if len(self.by_id) != len(self.points):raise ValueError('point_id存在重复，不能安全保存')
        self.backup_path = self.path.with_name(self.path.stem + '.before_gui.gpkg')
        self.undo_item = None

    def connect(self):
        conn = sqlite3.connect(str(self.path), timeout=3)
        conn.create_function('ST_IsEmpty', 1, lambda b: int(b is None or gpkg_geometry(b).is_empty))
        for name, index in (('ST_MinX',0),('ST_MinY',1),('ST_MaxX',2),('ST_MaxY',3)):
            conn.create_function(name, 1, lambda b, i=index: gpkg_geometry(b).bounds[i] if b else None)
        return conn

    def save(self, point_id, values, remember=True):
        """保存前自动备份；只更新人工字段，不接收位置或预测值修改。"""
        if values['ref_class'] not in (None, -1, 0, 1):raise ValueError('参考类别只能是0、1、-1或空值')
        p = self.by_id[point_id]
        old = {key:p[key] for key in FIELDS}
        conn = self.connect()
        try:
            if not self.backup_path.exists():
                pending=self.backup_path.with_suffix('.partial.gpkg')
                with closing(sqlite3.connect(str(pending))) as backup:conn.backup(backup)
                pending.replace(self.backup_path)
            conn.execute('BEGIN IMMEDIATE')
            current = conn.execute('SELECT '+','.join(FIELDS)+' FROM verification_points WHERE point_id=?', (point_id,)).fetchone()
            if current != tuple(old[k] for k in FIELDS):
                raise ValueError('这个点已被其他程序修改，请重新加载核验文件后再编辑')
            conn.execute('UPDATE verification_points SET '+','.join(k+'=?' for k in FIELDS)+' WHERE point_id=?',
                         tuple(values[k] for k in FIELDS)+(point_id,))
            conn.execute("UPDATE gpkg_contents SET last_change=? WHERE table_name='verification_points'",
                         (datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ'),))
            conn.commit()
        except Exception:
            conn.rollback();raise
        finally:conn.close()
        p.update(values)
        if remember:self.undo_item = (point_id, old)

    def undo(self):
        if self.undo_item is None:return None
        point_id, old = self.undo_item
        self.save(point_id, old, remember=False)
        self.undo_item = None
        return point_id
