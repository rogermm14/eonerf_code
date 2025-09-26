"""
This script contains functions that are useful to handle satellite images and georeferenced data
"""

import numpy as np
import rasterio
import datetime
import os
import shutil
import json
import glob
import rpcm
from PIL import Image
import torch

def get_file_id(filename):
    """
    return what is left after removing directory and extension from a path
    """
    return os.path.splitext(os.path.basename(filename))[0]

def read_dict_from_json(input_path):
    with open(input_path) as f:
        d = json.load(f)
    return d

def write_dict_to_json(d, output_path):
    with open(output_path, "w") as f:
        json.dump(d, f, indent=2)
    return d

def rpc_scaling_params(v):
    """
    find the scale and offset of a vector
    """
    vec = np.array(v).ravel()
    scale = (vec.max() - vec.min()) / 2
    offset = vec.min() + scale
    return scale, offset

def rescale_rpc(rpc, alpha):
    """
    Scale a rpc model following an image resize
    Args:
        rpc: rpc model to scale
        alpha: resize factor
               e.g. 2 if the image is upsampled by a factor of 2
                    1/2 if the image is downsampled by a factor of 2
    Returns:
        rpc_scaled: the scaled version of P by a factor alpha
    """
    import copy

    rpc_scaled = copy.copy(rpc)
    rpc_scaled.row_scale *= float(alpha)
    rpc_scaled.col_scale *= float(alpha)
    rpc_scaled.row_offset *= float(alpha)
    rpc_scaled.col_offset *= float(alpha)
    return rpc_scaled

def latlon_to_ecef_custom(lat, lon, alt):
    """
    convert from geodetic (lat, lon, alt) to geocentric coordinates (x, y, z)
    """
    rad_lat = lat * (np.pi / 180.0)
    rad_lon = lon * (np.pi / 180.0)
    a = 6378137.0
    finv = 298.257223563
    f = 1 / finv
    e2 = 1 - (1 - f) * (1 - f)
    v = a / np.sqrt(1 - e2 * np.sin(rad_lat) * np.sin(rad_lat))

    x = (v + alt) * np.cos(rad_lat) * np.cos(rad_lon)
    y = (v + alt) * np.cos(rad_lat) * np.sin(rad_lon)
    z = (v * (1 - e2) + alt) * np.sin(rad_lat)
    return x, y, z

def ecef_to_latlon_custom(x, y, z):
    """
    convert from geocentric coordinates (x, y, z) to geodetic (lat, lon, alt)
    """
    a = 6378137.0
    e = 8.1819190842622e-2
    asq = a ** 2
    esq = e ** 2
    b = np.sqrt(asq * (1 - esq))
    bsq = b ** 2
    ep = np.sqrt((asq - bsq) / bsq)
    p = np.sqrt((x ** 2) + (y ** 2))
    th = np.arctan2(a * z, b * p)
    lon = np.arctan2(y, x)
    lat = np.arctan2((z + (ep ** 2) * b * (np.sin(th) ** 3)), (p - esq * a * (np.cos(th) ** 3)))
    N = a / (np.sqrt(1 - esq * (np.sin(lat) ** 2)))
    alt = p / np.cos(lat) - N
    lon = lon * 180 / np.pi
    lat = lat * 180 / np.pi
    return lat, lon, alt

def utm_from_latlon(lats, lons):
    """
    convert lat-lon to utm
    """
    import pyproj
    import utm
    from pyproj import Transformer

    n = utm.latlon_to_zone_number(lats[0], lons[0])
    l = utm.latitude_to_zone_letter(lats[0])
    proj_src = pyproj.Proj("+proj=latlong")
    proj_dst = pyproj.Proj("+proj=utm +zone={}{}".format(n, l))
    transformer = Transformer.from_proj(proj_src, proj_dst)
    easts, norths = transformer.transform(lons, lats)
    #easts, norths = pyproj.transform(proj_src, proj_dst, lons, lats)
    return easts, norths

def lonlat_from_utm(easts, norths, zonestring):
    """
    convert utm to lon-lat
    """
    import pyproj
    proj_src = pyproj.Proj("+proj=utm +zone=%s" % zonestring)
    proj_dst = pyproj.Proj("+proj=latlong")
    return pyproj.transform(proj_src, proj_dst, easts, norths)

def utm_zonstring_from_lonlat(lon, lat):
    import utm
    n = utm.latlon_to_zone_number(lat, lon)
    l = utm.latitude_to_zone_letter(lat)
    return "{}{}".format(n, l)

def dsm_pointwise_diff(in_dsm_path, gt_dsm_path, dsm_metadata, gt_mask_path=None, out_rdsm_path=None, out_err_path=None):
    """
    in_dsm_path is a string with the path to the NeRF generated dsm
    gt_dsm_path is a string with the path to the reference lidar dsm
    bbx_metadata is a 4-valued array with format (x, y, s, r)
    where [x, y] = offset of the dsm bbx, s = width = height, r = resolution (m per pixel)
    """

    unique_identifier = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pred_dsm_path = "tmp_crop_dsm_to_delete_{}.tif".format(unique_identifier)
    pred_rdsm_path = "tmp_crop_rdsm_to_delete_{}.tif".format(unique_identifier)

    # read dsm metadata
    xoff, yoff = dsm_metadata[0], dsm_metadata[1]
    xsize, ysize = int(dsm_metadata[2]), int(dsm_metadata[2])
    resolution = dsm_metadata[3]

    # define projwin for gdal translate
    ulx, uly, lrx, lry = xoff, yoff + ysize * resolution, xoff + xsize * resolution, yoff

    # crop predicted dsm using gdal translate
    #from osgeo import gdal
    #ds = gdal.Translate(pred_dsm_path, in_dsm_path, options=f"-projwin {ulx} {uly} {lrx} {lry} -tr {resolution} {resolution}")
    #ds = None
    #assert(os.path.exists(pred_dsm_path))

    # FOR JEANZAY
    import time
    os.system(f'gdal_translate -of GTiff {in_dsm_path} {pred_dsm_path} -projwin {ulx} {uly} {lrx} {lry} -tr {resolution} {resolution}')
    time.sleep(10)
    assert(os.path.exists(pred_dsm_path))

    if gt_mask_path is not None:
        #print(f"found the gt mask {gt_mask_path}!")
        with rasterio.open(gt_mask_path, "r") as f:
            mask = f.read()[0, :, :]
            water_mask = mask.copy()
            water_mask[mask != 9] = 0
            water_mask[mask == 9] = 1
            water_mask = water_mask.astype(bool)
        if ("CLS.tif" in gt_mask_path) and (os.path.exists(gt_mask_path.replace("CLS.tif", "WATER.png"))):
            #print("found alternative water mask!")
            mask = np.array(Image.open(gt_mask_path.replace("CLS.tif", "WATER.png")))
            water_mask = mask == 0
        with rasterio.open(pred_dsm_path, "r") as f:
            profile = f.profile
            pred_dsm = f.read()[0, :, :]
        with rasterio.open(pred_dsm_path, 'w', **profile) as dst:
            water_mask_ = np.zeros_like(pred_dsm)
            h_ = min(water_mask.shape[0], pred_dsm.shape[0])
            w_ = min(water_mask.shape[1], pred_dsm.shape[1])
            water_mask_[:h_, :w_] = water_mask[:h_, :w_]
            pred_dsm[water_mask_.astype(bool)] = np.nan
            dst.write(pred_dsm, 1)

    # read predicted and gt dsms
    with rasterio.open(gt_dsm_path, "r") as f:
        gt_dsm = f.read()[0, :, :]
    with rasterio.open(pred_dsm_path, "r") as f:
        profile = f.profile
        pred_dsm = f.read()[0, :, :]

    # register and compute mae
    import dsmr
    transform = dsmr.compute_shift(gt_dsm_path, pred_dsm_path, scaling=False)
    dsmr.apply_shift(pred_dsm_path, pred_rdsm_path, *transform)
    with rasterio.open(pred_rdsm_path, "r") as f:
        pred_rdsm = f.read()[0, :, :]
    h = min(pred_rdsm.shape[0], gt_dsm.shape[0])
    w = min(pred_rdsm.shape[1], gt_dsm.shape[1])
    max_gt_alt = rasterio.open(gt_dsm_path).read(1).max()
    min_gt_alt = rasterio.open(gt_dsm_path).read(1).min()
    pred_rdsm = np.clip(pred_rdsm, min_gt_alt - 10, max_gt_alt + 10)
    #pred_rdsm -= np.nanpercentile(pred_rdsm[:h, :w] - gt_dsm[:h, :w], 25)
    err = pred_rdsm[:h, :w] - gt_dsm[:h, :w]

    # remove tmp files and write output tifs if desired
    os.remove(pred_dsm_path)
    if out_rdsm_path is not None:
        if os.path.exists(out_rdsm_path):
            os.remove(out_rdsm_path)
        os.makedirs(os.path.dirname(out_rdsm_path), exist_ok=True)
        shutil.copyfile(pred_rdsm_path, out_rdsm_path)
    os.remove(pred_rdsm_path)
    if out_err_path is not None:
        if os.path.exists(out_err_path):
            os.remove(out_err_path)
        os.makedirs(os.path.dirname(out_err_path), exist_ok=True)
        with rasterio.open(out_err_path, 'w', **profile) as dst:
            dst.write(err, 1)

    return err

def compute_mae_and_save_dsm_diff(pred_dsm_path, src_id, gt_dir, out_dir, epoch_number, aoi_id, save=True):
    # save dsm errs
    gt_dsm_path = os.path.join(gt_dir, "{}_DSM.tif".format(aoi_id))
    if aoi_id in ["JAX_004", "JAX_260"]:
        gt_seg_path = os.path.join(gt_dir, "{}_CLS_v2.tif".format(aoi_id))
    else:
        gt_seg_path = os.path.join(gt_dir, "{}_CLS.tif".format(aoi_id))
    assert os.path.exists(gt_dsm_path), f"{gt_dsm_path} not found"
    assert os.path.exists(gt_seg_path), f"{gt_seg_path} not found"

    if "JAX" in aoi_id:
        gt_roi_path = os.path.join(gt_dir, "{}_DSM.txt".format(aoi_id))
        assert os.path.exists(gt_roi_path), f"{gt_roi_path} not found"
        gt_roi_metadata = np.loadtxt(gt_roi_path)
    else:
        # IARPA
        src = rasterio.open(gt_dsm_path)
        gt_roi_metadata = np.array([src.bounds.left, src.bounds.bottom, min(src.height, src.width), src.res[0]])
        del src

    from sat_utils import dsm_pointwise_diff
    rdsm_diff_path = os.path.join(out_dir, "{}_rdsm_diff_epoch{}.tif".format(src_id, epoch_number))
    rdsm_path = os.path.join(out_dir, "{}_rdsm_epoch{}.tif".format(src_id, epoch_number))
    diff = dsm_pointwise_diff(pred_dsm_path, gt_dsm_path, gt_roi_metadata, gt_mask_path=gt_seg_path,
                                       out_rdsm_path=rdsm_path, out_err_path=rdsm_diff_path)
    #os.system(f"rm tmp*.tif.xml")
    if not save:
        os.remove(rdsm_diff_path)
        os.remove(rdsm_path)
    mae = np.nanmean(abs(diff.ravel()))
    return mae

def dsm_mae(in_dsm_path, gt_dsm_path, dsm_metadata, gt_mask_path=None):
    abs_err = dsm_pointwise_abs_errors(in_dsm_path, gt_dsm_path, dsm_metadata, gt_mask_path=gt_mask_path)
    return np.nanmean(abs_err.ravel())

def sort_by_increasing_view_incidence_angle(root_dir):
    incidence_angles = []
    json_paths = glob.glob(os.path.join(root_dir, "*.json"))
    for json_p in json_paths:
        with open(json_p) as f:
            d = json.load(f)
        rpc = rpcm.RPCModel(d["rpc"], dict_format="rpcm")
        c_lon, c_lat = d["geojson"]["center"][0], d["geojson"]["center"][1]
        alpha, _ = rpc.incidence_angles(c_lon, c_lat, z=0) # alpha = view incidence angle in degrees
        incidence_angles.append(alpha)
    return [x for _, x in sorted(zip(incidence_angles, json_paths))]

def sort_by_increasing_solar_incidence_angle(root_dir):
    solar_incidence_angles = []
    json_paths = glob.glob(os.path.join(root_dir, "*.json"))
    for json_p in json_paths:
        with open(json_p) as f:
            d = json.load(f)
        sun_el = np.radians(float(d["sun_elevation"]))
        sun_az = np.radians(float(d["sun_azimuth"]))
        sun_d = np.array([np.sin(sun_az) * np.cos(sun_el), np.cos(sun_az) * np.cos(sun_el), np.sin(sun_el)])
        surface_normal = np.array([0., 0., 1.0])
        u1 = sun_d / np.linalg.norm(sun_d)
        u2 = surface_normal / np.linalg.norm(surface_normal)
        alpha = np.degrees(np.arccos(np.dot(u1, u2))) # alpha = solar incidence angle in degrees
        solar_incidence_angles.append(alpha)
    return [x for _, x in sorted(zip(solar_incidence_angles, json_paths))]

def sort_by_acquisition_date(root_dir):
    acquisition_dates = []
    json_paths = glob.glob(os.path.join(root_dir, "*.json"))
    for json_p in json_paths:
        with open(json_p) as f:
            d = json.load(f)
        date_str = d["acquisition_date"]
        acquisition_dates.append(datetime.datetime.strptime(date_str, '%Y%m%d%H%M%S'))
    return [x for _, x in sorted(zip(acquisition_dates, json_paths))]

def sort_by_day_of_the_year(root_dir):
    acquisition_dates = []
    json_paths = glob.glob(os.path.join(root_dir, "*.json"))
    for json_p in json_paths:
        with open(json_p) as f:
            d = json.load(f)
        date_str = d["acquisition_date"]
        acquisition_dates.append(datetime.datetime.strptime(date_str, '%Y%m%d%H%M%S'))
    return [x for _, x in sorted(zip(acquisition_dates, json_paths), key=lambda x: x[0].timetuple().tm_yday)]

def reproject_dsm_alt_to_satellite_image(dsm_path, out_h, out_w, rpc, other_val_path=None):
    # note: other_val_path can be used to reproject some other magnitude instead of altitude (like confidence)
    # important: other_val_path and dsm_path must have the same size
    from pyproj import Transformer, CRS

    assert os.path.exists(dsm_path)
    with rasterio.open(dsm_path, "r") as src:
        dsm = src.read(1)
        x_min = np.min([src.bounds.left, src.bounds.right])
        x_max = np.max([src.bounds.left, src.bounds.right])
        y_min = np.min([src.bounds.bottom, src.bounds.top])
        y_max = np.max([src.bounds.bottom, src.bounds.top])
        h = src.height
        w = src.width
        crs_src = src.profile["crs"]
    dsm = dsm.ravel()

    # sample points in utm all over the area
    pt_density = 2
    X, Y = np.meshgrid(np.linspace(x_min, x_max, w * pt_density), np.linspace(y_max, y_min, h * pt_density))
    easts, norths = X.ravel(), Y.ravel()
    dsm_cols, dsm_rows = np.meshgrid(np.linspace(0, w - 1, w * pt_density), np.linspace(0, h - 1, h * pt_density))
    dsm_cols, dsm_rows = dsm_cols.astype(int).ravel(), dsm_rows.astype(int).ravel()
    index1d = (dsm_rows * w + dsm_cols).astype(int)
    alts = dsm[index1d] #dsm[dsm_rows, dsm_cols]

    # convert utm to lonlat and project using the rpc
    crs_dst = CRS.from_proj4("+proj=latlon")
    transformer = Transformer.from_crs(crs_src, crs_dst)
    lons, lats = transformer.transform(easts, norths)
    cols, rows = rpc.projection(lons, lats, alts)

    valid_cols = (cols >= 0) & (cols < out_w)
    valid_rows = (rows >= 0) & (rows < out_h)
    valid_pts = valid_cols & valid_rows
    cols = cols[valid_pts]
    rows = rows[valid_pts]

    if other_val_path is None:
        alts = alts[valid_pts]
    else:
        assert os.path.exists(other_val_path)
        with rasterio.open(other_val_path, "r") as src:
            assert (src.width == w) and (src.height == h)
            other_val = src.read(1)
        other_val = other_val.ravel()
        alts = other_val[index1d][valid_pts]

    dsm_alts = np.zeros((out_h, out_w), dtype=np.float32)
    dsm_alts[:] = np.nan
    dsm_alts[rows.astype(np.int16), cols.astype(np.int16)] = alts

    return dsm_alts


def utm_to_lonlat_differentiable(easts, norths, zone, northernHemisphere=True):
    # source:
    # https://stackoverflow.com/questions/343865/how-to-convert-from-utm-to-latlng-in-python-or-javascript
    # zone is the utm zone number
    if not northernHemisphere:
        norths = 10000000 - norths

    device = easts.device
    a = torch.Tensor([6378137]).to(device)
    e = torch.Tensor([0.081819191]).to(device)
    e1sq = torch.Tensor([0.006739497]).to(device)
    k0 = torch.Tensor([0.9996]).to(device)

    arc = norths / k0
    mu = arc / (a * (1 - torch.pow(e, 2) / 4.0 - 3 * torch.pow(e, 4) / 64.0 - 5 * torch.pow(e, 6) / 256.0))

    ei = (1 - torch.pow((1 - e * e), (1 / 2.0))) / (1 + torch.pow((1 - e * e), (1 / 2.0)))

    ca = 3 * ei / 2 - 27 * torch.pow(ei, 3) / 32.0

    cb = 21 * torch.pow(ei, 2) / 16 - 55 * torch.pow(ei, 4) / 32
    cc = 151 * torch.pow(ei, 3) / 96
    cd = 1097 * torch.pow(ei, 4) / 512
    phi1 = mu + ca * torch.sin(2 * mu) + cb * torch.sin(4 * mu) + cc * torch.sin(6 * mu) + cd * torch.sin(8 * mu)

    n0 = a / torch.pow((1 - torch.pow((e * torch.sin(phi1)), 2)), (1 / 2.0))

    r0 = a * (1 - e * e) / torch.pow((1 - torch.pow((e * torch.sin(phi1)), 2)), (3 / 2.0))
    fact1 = n0 * torch.tan(phi1) / r0

    _a1 = 500000 - easts
    dd0 = _a1 / (n0 * k0)
    fact2 = dd0 * dd0 / 2

    t0 = torch.pow(torch.tan(phi1), 2)
    Q0 = e1sq * torch.pow(torch.cos(phi1), 2)
    fact3 = (5 + 3 * t0 + 10 * Q0 - 4 * Q0 * Q0 - 9 * e1sq) * torch.pow(dd0, 4) / 24

    fact4 = (61 + 90 * t0 + 298 * Q0 + 45 * t0 * t0 - 252 * e1sq - 3 * Q0 * Q0) * torch.pow(dd0, 6) / 720

    lof1 = _a1 / (n0 * k0)
    lof2 = (1 + 2 * t0 + Q0) * torch.pow(dd0, 3) / 6.0
    lof3 = (5 - 2 * Q0 + 28 * t0 - 3 * torch.pow(Q0, 2) + 8 * e1sq + 24 * torch.pow(t0, 2)) * torch.pow(dd0, 5) / 120
    _a2 = (lof1 - lof2 + lof3) / torch.cos(phi1)
    _a3 = _a2 * 180 / torch.pi

    lats = 180 * (phi1 - fact1 * (fact2 + fact3 + fact4)) / torch.pi

    if not northernHemisphere:
        lats = -lats

    lons = ((zone > 0) and (6 * zone - 183.0) or 3.0) - _a3

    return lons, lats

def rpc_projection_differentiable(rpc, lon, lat, alt):

    nlon = (lon - rpc.lon_offset) / rpc.lon_scale
    nlat = (lat - rpc.lat_offset) / rpc.lat_scale
    nalt = (alt - rpc.alt_offset) / rpc.alt_scale

    col = apply_rfm(rpc.col_num, rpc.col_den, nlat, nlon, nalt)
    row = apply_rfm(rpc.row_num, rpc.row_den, nlat, nlon, nalt)

    col = col * rpc.col_scale + rpc.col_offset
    row = row * rpc.row_scale + rpc.row_offset

    return col, row

def apply_rfm(num, den, x, y, z):
    # copied from the rpcm package
    return apply_poly(num, x, y, z) / apply_poly(den, x, y, z)
def apply_poly(poly, x, y, z):
    # copied from the rpcm package
    out = 0
    out += poly[0]
    out += poly[1]*y + poly[2]*x + poly[3]*z
    out += poly[4]*y*x + poly[5]*y*z +poly[6]*x*z
    out += poly[7]*y*y + poly[8]*x*x + poly[9]*z*z
    out += poly[10]*x*y*z
    out += poly[11]*y*y*y
    out += poly[12]*y*x*x + poly[13]*y*z*z + poly[14]*y*y*x
    out += poly[15]*x*x*x
    out += poly[16]*x*z*z + poly[17]*y*y*z + poly[18]*x*x*z
    out += poly[19]*z*z*z
    return out
