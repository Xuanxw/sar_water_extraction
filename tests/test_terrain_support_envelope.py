"""高程包络的科学边界验证：向外取整、覆盖不足及缺测传播。"""
import sys
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from sar_geometry import terrain_support as module


class RegionalEnvelopeTests(unittest.TestCase):
    def make_inputs(self, root, missing=False, geoid_crs=4326):
        dem, geoid = root/'dem.tif', root/'geoid.tif'
        values = np.array([[-40., 100.], [900., 2000.]], dtype='float32')
        if missing:
            values[0, 0] = -9999
        with rasterio.open(dem, 'w', driver='GTiff', count=1, width=2, height=2,
                dtype='float32', crs=4326, transform=from_origin(1, 2, .25, .25), nodata=-9999) as dst:
            dst.write(values, 1)
        with rasterio.open(geoid, 'w', driver='GTiff', count=1, width=40, height=40,
                dtype='float32', crs=geoid_crs, transform=from_origin(-1, 3, .1, .1), nodata=-9999) as dst:
            dst.write(np.full((40, 40), -10., 'float32'), 1)
        return dem, geoid

    def test_height_datum_margin_and_outward_rounding(self):
        for crs in (4326, 4979):
            with tempfile.TemporaryDirectory() as tmp:
                dem, geoid = self.make_inputs(Path(tmp), geoid_crs=crs)
                result = module.scan_regional_dem_envelope([dem], geoid,
                    {'geographic_bounds_with_guard': [1.1, 1.6, 1.4, 1.9]}, 250, 500)
                self.assertEqual(result['height_bounds_m'], [-500., 2500.])
                self.assertEqual(result['geoid_undulation_range_m'], [-10., -10.])
                self.assertLessEqual(result['height_bounds_m'][0], -40-10-250)
                self.assertGreaterEqual(result['height_bounds_m'][1], 2000-10+250)

    def test_unobserved_external_domain_cannot_be_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            dem, geoid = self.make_inputs(Path(tmp))
            with self.assertRaisesRegex(ValueError, '未覆盖'):
                module.scan_regional_dem_envelope([dem], geoid,
                    {'geographic_bounds_with_guard': [0., 1.6, 1.4, 1.9]}, 250, 500)

    def test_missing_dem_must_not_become_a_finite_envelope(self):
        with tempfile.TemporaryDirectory() as tmp:
            dem, geoid = self.make_inputs(Path(tmp), missing=True)
            with self.assertRaisesRegex(ValueError, '缺测'):
                module.scan_regional_dem_envelope([dem], geoid,
                    {'geographic_bounds_with_guard': [1.1, 1.6, 1.4, 1.9]}, 250, 500)


    def test_extrema_cache_invalidates_changed_dem(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dem, geoid = self.make_inputs(root)
            domain = {'geographic_bounds_with_guard': [1.1, 1.6, 1.4, 1.9]}
            first = module.scan_regional_dem_envelope([dem], geoid, domain, 250, 500, cache_dir=root/'cache')
            previous_mtime = dem.stat().st_mtime_ns
            with rasterio.open(dem, 'r+') as dst:
                data = dst.read(1)
                data[1, 1] = 5000
                dst.write(data, 1)
            os.utime(dem, ns=(previous_mtime+1000000000, previous_mtime+1000000000))
            second = module.scan_regional_dem_envelope([dem], geoid, domain, 250, 500, cache_dir=root/'cache')
            self.assertEqual(first['height_bounds_m'][1], 2500.)
            self.assertEqual(second['height_bounds_m'][1], 5500.)

if __name__ == '__main__':
    unittest.main()
