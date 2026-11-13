"""读取BC Stripmap注释；产品字段的存在与其物理约定确认分别记录。"""
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET
import numpy as np

LIGHT_SPEED = 299792458.0
# 本模块不执行UTC到TAI转换；跨这些闰秒边界的数据明确拒绝。
LEAP_BOUNDARIES = ('1972-07-01','1973-01-01','1974-01-01','1975-01-01','1976-01-01',
                  '1977-01-01','1978-01-01','1979-01-01','1980-01-01','1981-07-01',
                  '1982-07-01','1983-07-01','1985-07-01','1988-01-01','1990-01-01',
                  '1991-01-01','1992-07-01','1993-07-01','1994-07-01','1996-01-01',
                  '1997-07-01','1999-01-01','2006-01-01','2009-01-01','2012-07-01',
                  '2015-07-01','2017-01-01')


def utc(value):
    """BC时间没有后缀时按产品UTC字段解释；不接受未定义的本地时区。"""
    t = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


def read_bc_metadata(path, xml_member=None):
    """只读XML或ZIP中的注释，不展开SLC大影像、不读取像元。"""
    path = Path(path)
    if path.suffix.lower() == '.zip':
        with ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith('.xml')
                     and '/annotation/' in n.lower() and '/calibration/' not in n.lower()
                     and '/noise/' not in n.lower()]
            if xml_member is None:
                if len(names) != 1:
                    raise ValueError(f'需要指定唯一的XML_MEMBER；候选：{names}')
                xml_member = names[0]
            raw = z.read(xml_member)
    else:
        raw = path.read_bytes()
    root = ET.fromstring(raw)
    for element in root.iter():
        element.tag = element.tag.split('}')[-1]

    def field(base, name):
        text = root.findtext(base + '/' + name)
        if text is None:
            raise ValueError(f'XML缺少字段：{base}/{name}')
        return text.strip()

    info = 'imageAnnotation/imageInformation'
    general = 'generalAnnotation/productInformation'
    time0 = utc(field(info, 'productFirstLineUtcTime'))
    last = utc(field(info, 'productLastLineUtcTime'))
    orbit = []
    orbit_times = []
    for element in root.findall('generalAnnotation/orbitList/orbit'):
        if element.findtext('frame', '').strip().lower() not in ('earth fixed', 'ecef'):
            raise ValueError('当前只支持明确为Earth Fixed/ECEF的轨道')
        t = utc(element.findtext('time'))
        orbit_times.append(t)
        orbit.append([(t-time0).total_seconds()] + [float(element.findtext(f'{part}/{axis}'))
                     for part in ('position','velocity') for axis in ('x','y','z')])
    orbit = np.asarray(orbit, dtype='float64')
    if orbit.ndim != 2 or orbit.shape[1] != 7 or len(orbit) < 2:
        raise ValueError('需要至少两个完整轨道状态')
    if not np.isfinite(orbit).all() or np.any(np.diff(orbit[:,0]) <= 0):
        raise ValueError('轨道含非有限数值或时间顺序异常')
    start, end = min(orbit_times+[time0]), max(orbit_times+[last])
    if start.year < 1972 or end.year > 2026:
        raise ValueError('当前UTC适配支持1972—2026；其他年份需更新时标规则')
    if any(start < utc(date+'T00:00:00') <= end for date in LEAP_BOUNDARIES):
        raise ValueError('跨闰秒数据需连续时标转换，当前适配器尚不支持')

    records = []
    for e in root.findall('dopplerCentroid/dcEstimateList/dcEstimate'):
        text = e.findtext('dataDcPolynomial')
        if text is None:
            raise ValueError('缺少dataDcPolynomial，不能静默改用其他Doppler定义')
        records.append([(utc(e.findtext('azimuthTime'))-time0).total_seconds(),
                        float(e.findtext('t0')), *map(float, text.split())])
    if records and len({len(r) for r in records}) != 1:
        raise ValueError('Doppler系数长度不一致')
    doppler = np.asarray(records, dtype='float64') if records else np.empty((0,2))
    if len(doppler) and (not np.isfinite(doppler).all() or np.any(np.diff(doppler[:,0]) <= 0)):
        raise ValueError('Doppler记录含非有限值或时间顺序异常')
    look = field(info, 'look_side').lower()
    if look not in ('left','right'):
        raise ValueError(f'未知视向：{look}')
    product = field('adsHeader', 'productType')
    if '-SM-' not in (xml_member or path.name).upper():
        raise ValueError('首版只支持明确标记为SM的BC Stripmap产品')
    grid = dict(wavelength=LIGHT_SPEED/float(field(general,'radarFrequency')),
                range0=LIGHT_SPEED*float(field(info,'slantRangeTime'))/2,
                range_spacing=float(field(info,'rangePixelSpacing')),time0=0.0,
                line_time=float(field(info,'azimuthTimeInterval')),
                columns=int(field(info,'numberOfSamples')),rows=int(field(info,'numberOfLines')),
                look_side=1 if look=='right' else -1)
    if min(grid[k] for k in ('wavelength','range0','range_spacing','line_time','columns','rows')) <= 0:
        raise ValueError('雷达网格尺寸、采样和波长必须为正')
    description = dict(product_type=product,mission_id=field('adsHeader','missionId'),
                       xml_member=xml_member,epoch_utc=time0.isoformat(),look_side=look,
                       orbit_records=len(orbit),doppler_records=len(doppler),
                       end_time_discrepancy_seconds=(last-time0).total_seconds()-(grid['rows']-1)*grid['line_time'],
                       note='字段来自雷达注释；不代表ORG GeoTIFF网格或成像约定已确认。')
    return dict(grid=grid,orbit=orbit,doppler=doppler,description=description,xml_bytes=raw)


def make_model(metadata, settings):
    """确认后才构建求解模型；不根据XML中有系数就自动猜测deskew或Doppler符号。"""
    if settings['radar_grid_confirmed'] is not True:
        raise ValueError('先确认XML的实际雷达网格及时间原点，再设置radar_grid_confirmed=true')
    mode = settings['doppler_mode']
    if mode not in ('zero','metadata'):
        raise ValueError('必须明确doppler_mode=zero或metadata；不能根据产品名称猜测')
    sign = 1 if mode=='zero' else settings['doppler_sign']
    if sign not in (-1,1):
        raise ValueError('metadata模式必须明确Doppler符号为1或-1')
    if mode=='metadata' and len(metadata['doppler']) < 2:
        raise ValueError('metadata模式需要至少两个Doppler时间估计，不外推')
    return dict(**metadata['grid'],orbit=metadata['orbit'],doppler=metadata['doppler'],
                doppler_mode=0 if mode=='zero' else 1,doppler_sign=sign,
                max_iterations=settings['max_iterations'],tolerance_hz=settings['tolerance_hz'],
                tolerance_pixels=settings['tolerance_pixels'],max_step_seconds=settings['max_step_seconds'])
