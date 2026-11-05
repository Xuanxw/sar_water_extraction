"""沿准备清单读取原始GeoTIFF/ZIP及同名XML，不猜用目录中其他景的参数。"""
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET
import hashlib
import json
import rasterio


def read_source_metadata(source_image):
    """返回元数据、原始栅格profile和XML字节；缺少XML时明确记录，不伪造角度。"""
    path = Path(source_image)
    xml_bytes, xml_source = None, ""
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            members = [n for n in archive.namelist() if Path(n).suffix.lower() in (".tif", ".tiff")]
            if len(members) != 1:
                raise ValueError(f"原始ZIP应包含唯一GeoTIFF：{path}")
            member = members[0]
            matches = [n for n in archive.namelist()
                       if str(Path(n).with_suffix("")).casefold() == str(Path(member).with_suffix("")).casefold()
                       and Path(n).suffix.lower() == ".xml"]
            if len(matches) > 1:
                raise ValueError(f"同景XML不唯一：{path}")
            if matches:
                xml_bytes, xml_source = archive.read(matches[0]), str(path)+"!"+matches[0]
        raster_path = "/vsizip/" + path.as_posix() + "/" + member
    else:
        raster_path = str(path)
        matches = [p for p in path.parent.iterdir() if p.suffix.lower() == ".xml" and p.stem.casefold() == path.stem.casefold()]
        if len(matches) > 1:
            raise ValueError(f"同景XML不唯一：{path}")
        if matches:
            xml_bytes, xml_source = matches[0].read_bytes(), str(matches[0])
    with rasterio.open(raster_path) as source:
        profile = source.profile.copy()
        metadata = dict(source_image=str(path), source_raster=raster_path, source_xml=xml_source,
                        xml_status="found" if xml_bytes else "missing", xml_sha256=hashlib.sha256(xml_bytes).hexdigest() if xml_bytes else "",
                        width=source.width, height=source.height, count=source.count, dtype=str(source.dtypes),
                        crs=str(source.crs), transform=str(tuple(source.transform)), pixel_x=source.res[0], pixel_y=source.res[1],
                        nodata=str(source.nodata), scales=str(source.scales), offsets=str(source.offsets),
                        raster_tags=json.dumps(source.tags(), ensure_ascii=False))
    fields = ("productType", "polarisation", "startTime", "stopTime", "absoluteOrbitNumber",
              "platformHeading", "incidenceAngleMidSwath", "look_side", "radarFrequency",
              "pixelValue", "outputPixels", "rangePixelSpacing", "azimuthPixelSpacing")
    values = {}
    if xml_bytes:
        # 去掉XML命名空间后按已知字段读取；像素网格以GeoTIFF为准，不能用斜距间距覆盖它。
        xml_root = ET.fromstring(xml_bytes)
        for node in xml_root.iter():
            key = node.tag.split("}")[-1]
            if key in fields and key not in values:
                values[key] = (node.text or "").strip()
    metadata.update({key: values.get(key, "") for key in fields})
    return metadata, profile, xml_bytes
