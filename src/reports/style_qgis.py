"""将设施暴露 GeoPackage 导出为带符号的 QGIS 图层定义。"""

import argparse
import json
from pathlib import Path

from qgis.core import (Qgis, QgsApplication, QgsCategorizedSymbolRenderer, QgsFillSymbol,
                       QgsLayerDefinition, QgsLineSymbol, QgsProject, QgsRendererCategory,
                       QgsVectorLayer)


def export_definition(config, folder):
    project = QgsProject.instance()
    group = project.layerTreeRoot().addGroup('道路与建筑潜在暴露')
    source = folder / 'facility_exposure.gpkg'
    buildings = 0
    for spec in config['facilities']:
        road = spec['kind'] == 'road'
        layer = QgsVectorLayer(f'{source}|layername={spec["key"]}',
                               f'{spec["label"]}｜' + ('道路相交段' if road else '建筑候选轮廓'), 'ogr')
        if not layer.isValid():
            raise OSError(f'无法加载 {spec["key"]}')
        if road:
            categories = [QgsRendererCategory(value, QgsLineSymbol.createSimple({
                'line_color': color, 'line_width': '0.85', 'line_style': style}), label)
                for value, color, style, label in [(0, '#AA2835', 'solid', '与新增水面相交'),
                                                   (1, '#6B1823', 'dash', '桥隧标记路段')]]
            layer.setRenderer(QgsCategorizedSymbolRenderer('bridge_tunnel', categories))
        elif buildings == 0:
            categories = [QgsRendererCategory(value, QgsFillSymbol.createSimple({
                'color': color, 'outline_color': 'white', 'outline_width': '0.16'}), label)
                for value, color, label in [(0, '#D36883', '重叠 <50%'), (1, '#AE144F', '重叠 ≥50%')]]
            layer.setRenderer(QgsCategorizedSymbolRenderer('if("overlap_pct" >= 50, 1, 0)', categories))
            buildings += 1
        else:
            layer.renderer().setSymbol(QgsFillSymbol.createSimple({
                'color': '#116964', 'outline_color': 'white', 'outline_width': '0.16'}))
            buildings += 1
        project.addMapLayer(layer, False)
        group.insertLayer(0, layer)
    saved, message = QgsLayerDefinition.exportLayerDefinition(
        str(folder / 'facility_exposure.qlr'), [group], Qgis.FilePathType.Relative)
    if not saved:
        raise OSError(message)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('maps', type=Path)
    args = parser.parse_args()
    app = QgsApplication([], False)
    app.initQgis()
    export_definition(json.loads(args.case.read_text(encoding='utf-8')), args.maps.resolve())
