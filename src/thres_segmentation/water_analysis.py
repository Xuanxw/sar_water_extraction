"""Georeferenced water area, change and boundary-constrained depth products.

Depth is a geometric estimate inspired by FLEXTH (2026, section 3.3.1), not
a reproduction of its Method B or KDFIMv2's temporal/hydraulic reconstruction.
"""
from contextlib import ExitStack
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_geom
from rasterio.windows import Window
from scipy import ndimage
from scipy.spatial import cKDTree


CHANGE_CLASSES = {0: "stable_dry", 1: "persistent_water", 2: "new_water", 3: "lost_water", 255: "unknown"}
DEPTH_CLASSES = {0: "outside_target", 1: "estimated", 2: "missing_elevation",
                 3: "insufficient_boundary", 4: "beyond_distance_limit",
                 5: "negative_depth_conflict", 6: "historical_water_excluded",
                 7: "unknown_history", 255: "invalid_current"}


def windows(source, size):
    for row in range(0, source.height, size):
        for col in range(0, source.width, size):
            yield Window(col, row, min(size, source.width-col), min(size, source.height-row))


def pixel_areas(crs, transform, shape):
    """m²: projected grid area (unit-corrected), or ellipsoidal geographic cells.

    Projected areas retain projection distortion; use an appropriate equal-area
    or local CRS for quantitative work. Geographic north-up grids use one value
    per row. Rotated geographic grids evaluate individual cell polygons.
    """
    from pyproj import CRS
    crs = CRS.from_user_input(crs)
    if crs.is_projected:
        factors = [a.unit_conversion_factor for a in crs.axis_info[:2]]
        area = abs(transform.a*transform.e-transform.b*transform.d)*np.prod(factors)
        if not np.isfinite(area) or area <= 0:
            raise ValueError("Invalid pixel area / affine transform")
        return float(area)
    if not crs.is_geographic or any(a.unit_name != "degree" for a in crs.axis_info[:2]):
        raise ValueError("Area requires a projected CRS or geographic CRS in degrees")
    geod = crs.get_geod()
    height, width = shape
    result = np.empty((height, 1 if transform.b == transform.d == 0 else width))
    for row in range(height):
        for col in range(result.shape[1]):
            corners = [transform*(col+x, row+y) for x,y in ((0,0),(1,0),(1,1),(0,1))]
            x,y = zip(*corners)
            if min(y) < -90 or max(y) > 90:
                raise ValueError("Invalid latitude in raster grid")
            result[row,col] = abs(geod.polygon_area_perimeter(x,y)[0])
    if not np.all(np.isfinite(result) & (result > 0)):
        raise ValueError("Invalid geographic pixel area")
    return result


def area_sum(mask, areas):
    return float(np.sum(np.asarray(mask, dtype=np.float64)*areas))


def quantities(area):
    return dict(m2=area, ha=area/10000, km2=area/1e6)


def _profile(source, dtype="uint8", nodata=255):
    return dict(driver="GTiff", height=source.height, width=source.width,
                count=1, crs=source.crs, transform=source.transform,
                dtype=dtype, nodata=nodata, compress="deflate", tiled=True,
                blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")


def _aligned(stack, path, target, resampling):
    source = stack.enter_context(rasterio.open(path))
    if not source.crs:
        raise ValueError(f"Missing CRS: {path}")
    # Float NaN destination prevents uncovered cells becoming valid zero water/land.
    return stack.enter_context(WarpedVRT(source, crs=target.crs,
        transform=target.transform, width=target.width, height=target.height,
        dtype="float64", nodata=float("nan"), resampling=resampling))


def _polygons(path, crs):
    import fiona
    result = []
    with fiona.open(path) as source:
        if not source.crs:
            raise ValueError(f"Missing vector CRS: {path}")
        for feature in source:
            geom = feature["geometry"]
            if geom is None or geom["type"] not in ("Polygon", "MultiPolygon"):
                raise ValueError("Water/coverage vectors must contain only nonempty polygons")
            result.append(transform_geom(source.crs, crs, geom))
    return result


def _burn(geometries, shape, transform):
    if not geometries:
        return np.zeros(shape, dtype=bool)
    return rasterize([(g,1) for g in geometries], out_shape=shape, transform=transform,
                     fill=0, dtype="uint8", all_touched=False).astype(bool)


def _history_reader(stack, reference_path, target, water_values, dry_values,
                    coverage_path, vector_assume_complete):
    coverage = _polygons(coverage_path, target.crs) if coverage_path else None
    vector = Path(reference_path).suffix.lower() in (".shp", ".gpkg", ".geojson", ".json")
    if vector:
        if coverage is None and not vector_assume_complete:
            raise ValueError("Water vectors require reference_coverage_path or vector_assume_complete=True")
        polygons = _polygons(reference_path, target.crs)
    else:
        if not water_values or not dry_values or set(water_values) & set(dry_values):
            raise ValueError("Historical water/dry values must be nonempty and disjoint")
        source = _aligned(stack, reference_path, target, Resampling.nearest)

    def read(window):
        shape = (int(window.height), int(window.width))
        transform = target.window_transform(window)
        if vector:
            water = _burn(polygons, shape, transform)
            valid = np.ones(shape, bool)
        else:
            data = source.read(1, window=window, masked=True)
            valid = ~np.ma.getmaskarray(data) & np.isfinite(data.data)
            valid &= np.isin(data.data, (*water_values, *dry_values))
            water = np.isin(data.data, water_values)
        if coverage is not None:
            valid &= _burn(coverage, shape, transform)
        return water & valid, valid
    return read


def read_reference_water(target_path, reference_path, reference_water_values=(1,),
                         reference_dry_values=(0,), reference_coverage_path=None,
                         vector_assume_complete=False):
    """按目标网格读取历史水体及有效域；类别最近邻，未知区域不作为陆地。

    供整图后处理复用与变化分析相同的参考数据契约。整景分析仍使用分窗口 reader。
    """
    with ExitStack() as stack:
        target = stack.enter_context(rasterio.open(target_path))
        if not target.crs:
            raise ValueError("Georeferenced CRS required for historical water correction")
        read = _history_reader(stack, reference_path, target, reference_water_values,
                               reference_dry_values, reference_coverage_path, vector_assume_complete)
        return read(Window(0, 0, target.width, target.height))


def estimate_depth(water, valid, elevation, elevation_valid, transform, crs,
                   history=None, history_valid=None, scope="new_water",
                   neighbors=100, min_boundary=3, max_distance_m=5000.,
                   max_boundary_samples=200000, cancel=None):
    """Per 4-connected component IDW² of valid dry shoreline elevations.

    Never sample scene edges, NoData or historical water as a dry shoreline.
    Use a local azimuthal equidistant CRS for distance, independent of grid units.
    Invalid/unsupported depths are NaN; a true zero depth is valid.
    """
    from pyproj import CRS, Transformer
    if scope not in ("new_water", "all_water"):
        raise ValueError("depth_scope must be new_water or all_water")
    if scope == "new_water" and history is None:
        raise ValueError("new_water depth requires a historical reference; use all_water explicitly otherwise")
    if min_boundary < 1 or neighbors < min_boundary or max_boundary_samples < min_boundary or not np.isfinite(max_distance_m) or max_distance_m <= 0:
        raise ValueError("Invalid depth interpolation parameters")
    water = water & valid
    target = water.copy()
    quality = np.zeros(water.shape, dtype="uint8")
    quality[~valid] = 255
    if scope == "new_water":
        target &= history_valid & ~history
        quality[water & history_valid & history] = 6
        quality[water & ~history_valid] = 7
    quality[target] = 3
    quality[target & ~elevation_valid] = 2
    depth = np.full(water.shape, np.nan, dtype="float32")
    level = depth.copy()
    distance_map = depth.copy()
    components, _ = ndimage.label(water)
    objects = ndimage.find_objects(components)
    center = transform*(water.shape[1]/2, water.shape[0]/2)
    lon, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(*center)
    metric = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m")
    project = Transformer.from_crs(crs, metric, always_xy=True)

    def coordinates(rows, cols):
        x = transform.a*(cols+.5)+transform.b*(rows+.5)+transform.c
        y = transform.d*(cols+.5)+transform.e*(rows+.5)+transform.f
        x,y = project.transform(x,y)
        points = np.column_stack((x,y))
        if not np.isfinite(points).all():
            raise ValueError("Depth distance projection failed")
        return points

    records = []
    for identifier, box in enumerate(objects, 1):
        if cancel and cancel():
            raise InterruptedError("Water analysis cancelled")
        if box is None:
            continue
        rr = slice(max(0,box[0].start-1), min(water.shape[0],box[0].stop+1))
        cc = slice(max(0,box[1].start-1), min(water.shape[1],box[1].stop+1))
        local = components[rr,cc] == identifier
        query_mask = local & target[rr,cc] & elevation_valid[rr,cc]
        if not np.any(local & target[rr,cc]):
            continue
        shoreline = ndimage.binary_dilation(local) & ~water[rr,cc]
        shoreline &= valid[rr,cc] & elevation_valid[rr,cc]
        if history is not None:
            shoreline &= history_valid[rr,cc] & ~history[rr,cc]
        br,bc = np.nonzero(shoreline)
        samples = len(br)
        if samples > max_boundary_samples:
            keep = np.linspace(0, samples-1, max_boundary_samples, dtype=int)
            br,bc = br[keep],bc[keep]
        record = dict(component=identifier, water_pixels=int(local.sum()),
                      boundary_pixels=samples, used_boundary_pixels=len(br),
                      estimated_pixels=0, negative_pixels=0)
        records.append(record)
        if len(br) < min_boundary:
            continue
        heights = elevation[rr,cc][br,bc]
        tree = cKDTree(coordinates(br+rr.start,bc+cc.start))
        qr,qc = np.nonzero(query_mask)
        for start in range(0,len(qr),8192):
            if cancel and cancel():
                raise InterruptedError("Water analysis cancelled")
            rows,cols = qr[start:start+8192]+rr.start,qc[start:start+8192]+cc.start
            distances, indices = tree.query(coordinates(rows,cols), k=min(neighbors,len(br)),
                                           distance_upper_bound=max_distance_m)
            if distances.ndim == 1:
                distances,indices = distances[:,None],indices[:,None]
            supported = np.isfinite(distances)
            weights = np.zeros(distances.shape)
            weights[supported] = 1/np.maximum(distances[supported],1e-6)**2
            total = weights.sum(axis=1)
            wse = np.divide((weights*heights[np.minimum(indices,len(br)-1)]).sum(axis=1),
                            total, out=np.full(len(rows),np.nan), where=total>0)
            raw = wse-elevation[rows,cols]
            # Cancel floating-point roundoff at an exact zero (not terrain error).
            raw[np.abs(raw) <= 1e-9] = 0.
            quality[rows,cols] = 4
            enough = supported.sum(axis=1) >= min_boundary
            conflict = enough & (raw < 0)
            good = enough & (raw >= 0)
            quality[rows[conflict],cols[conflict]] = 5
            quality[rows[good],cols[good]] = 1
            depth[rows[good],cols[good]] = raw[good]
            level[rows[enough],cols[enough]] = wse[enough]
            near = distances[:,0]
            distance_map[rows[enough],cols[enough]] = near[enough]
            record["estimated_pixels"] += int(good.sum())
            record["negative_pixels"] += int(conflict.sum())
    return depth, level, quality, distance_map, records


def analyze_water(mask_path, output_dir, reference_path=None,
                  reference_water_values=(1,), reference_dry_values=(0,),
                  reference_coverage_path=None, vector_assume_complete=False,
                  elevation_path=None, elevation_kind="DEM", elevation_scale=1.,
                  depth_scope="new_water", depth_neighbors=100, depth_min_boundary=3,
                  depth_max_distance_m=5000., depth_max_pixels=25000000,
                  block_size=512, cancel=None):
    """Analyze 0=dry, 1=water, 255/NoData=unknown mask; fresh output only.

    Area/change are windowed. Optional depth uses whole-scene connectivity and
    is guarded by depth_max_pixels, never silently tiled or downsampled.
    """
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if elevation_kind not in ("DEM", "DSM") or not np.isfinite(elevation_scale) or elevation_scale <= 0:
        raise ValueError("Specify DEM/DSM and a positive elevation-to-metre scale")
    output_dir = Path(output_dir)
    summary = dict(mask=str(Path(mask_path).resolve()), reference=str(reference_path) if reference_path else None,
                   elevation=str(elevation_path) if elevation_path else None,
                   reference_water_values=list(reference_water_values), reference_dry_values=list(reference_dry_values),
                   reference_coverage=str(reference_coverage_path) if reference_coverage_path else None,
                   vector_assume_complete=vector_assume_complete, warnings=[])
    with ExitStack() as stack:
        source = stack.enter_context(rasterio.open(mask_path))
        if not source.crs:
            if reference_path or elevation_path:
                raise ValueError("Georeferenced CRS required for change/depth analysis")
            summary.update(status="unavailable_missing_crs", current_water=None)
            output_dir.mkdir(parents=True,exist_ok=False)
            (output_dir/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
            return summary
        if elevation_path and source.width*source.height > depth_max_pixels:
            raise ValueError("Depth exceeds depth_max_pixels; use a hydrologically complete ROI or explicitly raise the memory limit")
        if elevation_path and depth_scope == "new_water" and not reference_path:
            raise ValueError("new_water depth requires a historical reference")
        history = (_history_reader(stack, reference_path, source, reference_water_values,
                    reference_dry_values, reference_coverage_path, vector_assume_complete)
                   if reference_path else None)
        elevation = _aligned(stack, elevation_path, source, Resampling.bilinear) if elevation_path else None
        output_dir.mkdir(parents=True,exist_ok=False)
        change = stack.enter_context(rasterio.open(output_dir/"water_change.tif", "w", **_profile(source))) if history else None
        vector = None
        if history:
            import fiona
            change.update_tags(classes=json.dumps(CHANGE_CLASSES), comparison="common_valid_coverage")
            vector = stack.enter_context(fiona.open(output_dir/"water_changes.gpkg", "w", driver="GPKG",
                layer="changes", crs_wkt=source.crs.to_wkt(), schema={"geometry":"Polygon",
                "properties":{"class_id":"int", "change":"str", "area_m2":"float"}}))
        totals = dict(current_water=0., current_valid=0., comparison_valid=0.,
                      historical_water=0., current_water_common=0., persistent_water=0.,
                      new_water=0., lost_water=0., stable_dry=0., unknown_history=0.)
        counts = dict(water_pixels=0, valid_pixels=0)
        for window in windows(source,block_size):
            if cancel and cancel():
                raise InterruptedError("Water analysis cancelled")
            data = source.read(1, window=window, masked=True)
            valid = ~np.ma.getmaskarray(data) & np.isin(data.data,(0,1))
            water = (data.data == 1) & valid
            transform = source.window_transform(window)
            areas = pixel_areas(source.crs,transform,data.shape)
            totals["current_water"] += area_sum(water,areas)
            totals["current_valid"] += area_sum(valid,areas)
            counts["water_pixels"] += int(water.sum())
            counts["valid_pixels"] += int(valid.sum())
            if history:
                previous,known = history(window)
                common = valid & known
                codes = np.full(data.shape,255,dtype="uint8")
                codes[common & ~water & ~previous] = 0
                codes[common & water & previous] = 1
                codes[common & water & ~previous] = 2
                codes[common & ~water & previous] = 3
                change.write(codes,1,window=window)
                for key,mask in (("comparison_valid",common),("historical_water",common & previous),
                                 ("current_water_common",common & water),("unknown_history",valid & ~known)):
                    totals[key] += area_sum(mask,areas)
                for value,key in CHANGE_CLASSES.items():
                    if value != 255:
                        totals[key] += area_sum(codes == value,areas)
                # Window fragments preserve exact raster boundaries; no smoothing.
                for value in (2,3):
                    labels, _ = ndimage.label(codes == value)
                    component_areas = np.bincount(labels.ravel(),
                        weights=np.broadcast_to(areas,data.shape).ravel())
                    for geom,identifier in shapes(labels,mask=labels>0,transform=transform):
                        vector.write(dict(geometry=geom,properties=dict(class_id=value,
                            change=CHANGE_CLASSES[value],area_m2=float(component_areas[int(identifier)]))))
        summary.update(status="complete", crs=source.crs.to_string(), transform=list(source.transform)[:6],
                       area_method="projected_grid_unit_corrected" if source.crs.is_projected else "ellipsoidal_geographic_cells",
                       counts=counts, current_water=quantities(totals["current_water"]),
                       current_valid=quantities(totals["current_valid"]))
        if source.crs.is_projected:
            summary["warnings"].append("Areas retain map projection distortion; use a suitable local or equal-area CRS.")
        if history:
            summary["change"] = {k:quantities(v) for k,v in totals.items() if k not in ("current_water","current_valid")}
            summary["change"]["net_change"] = quantities(totals["new_water"]-totals["lost_water"])
            summary["warnings"].append("Change is within common valid coverage; historical product/season differences are not automatically flood events. Vector polygons may be split at processing windows.")
        if elevation:
            data = source.read(1,masked=True)
            valid = ~np.ma.getmaskarray(data) & np.isin(data.data,(0,1))
            water = (data.data == 1) & valid
            terrain = elevation.read(1,masked=True)
            zvalid = ~np.ma.getmaskarray(terrain) & np.isfinite(terrain.data)
            z = terrain.data*elevation_scale
            zvalid &= np.isfinite(z)
            previous,known = history(Window(0,0,source.width,source.height)) if history else (None,None)
            depth,level,quality,distance,records = estimate_depth(water,valid,z,zvalid,
                source.transform,source.crs,previous,known,scope=depth_scope,
                neighbors=depth_neighbors,min_boundary=depth_min_boundary,
                max_distance_m=depth_max_distance_m,cancel=cancel)
            for name,array in (("water_depth_m",depth),("water_level_m",level),("boundary_distance_m",distance)):
                with rasterio.open(output_dir/(name+".tif"),"w",**_profile(source,"float32",-9999.)) as target:
                    target.write(np.where(np.isfinite(array),array,-9999.).astype("float32"),1)
                    target.update_tags(units="m",method="component_shoreline_idw2", elevation_kind=elevation_kind)
            with rasterio.open(output_dir/"depth_quality.tif","w",**_profile(source)) as target:
                target.write(quality,1); target.update_tags(classes=json.dumps(DEPTH_CLASSES))
            with (output_dir/"depth_components.csv").open("w",newline="",encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream,fieldnames=["component","water_pixels","boundary_pixels","used_boundary_pixels","estimated_pixels","negative_pixels"])
                writer.writeheader();writer.writerows(records)
            depth_area = volume = 0.
            for window in windows(source,block_size):
                r,c = window.toslices()
                areas = pixel_areas(source.crs,source.window_transform(window),depth[r,c].shape)
                good = quality[r,c] == 1
                depth_area += area_sum(good,areas)
                volume += area_sum(np.where(good,depth[r,c],0),areas)
            good = quality == 1
            summary["depth"] = dict(method="component_shoreline_idw2",scope=depth_scope,
                elevation_kind=elevation_kind,elevation_scale=elevation_scale,
                neighbors=depth_neighbors,min_boundary=depth_min_boundary,max_distance_m=depth_max_distance_m,
                quality_classes=DEPTH_CLASSES,quality_counts={str(k):int((quality==k).sum()) for k in DEPTH_CLASSES},
                estimated_area=quantities(depth_area),estimated_volume_m3=volume,
                area_weighted_mean_m=volume/depth_area if depth_area else None,
                max_m=float(depth[good].max()) if good.any() else None)
            summary["warnings"].extend([
                "Geometric estimate only: shoreline/terrain errors propagate to depth; no hydraulic or field validation.",
                "Elevation units must be converted to metres; resampling does not transform vertical datums or improve terrain resolution.",
                "Permanent water bathymetry is generally absent from DEM/DSM; all_water requires suitable submerged terrain."])
            if elevation_kind == "DSM":
                summary["warnings"].append("DSM measures surface height including canopy/buildings; depth is relative to that surface, not bare ground.")
    # Written only after all raster/vector handles close successfully.
    with (output_dir/"areas.csv").open("w",newline="",encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream,fieldnames=["category","m2","ha","km2"])
        writer.writeheader()
        for key in ("current_valid","current_water"):
            writer.writerow(dict(category=key,**summary[key]))
        for key,value in summary.get("change",{}).items():
            writer.writerow(dict(category=key,**value))
    (output_dir/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    return summary
