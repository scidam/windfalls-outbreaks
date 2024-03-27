# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 05:59:15 2020

@author: dmitry
"""

import os  # noqa: E402
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np # noqa: E402
from processtiff import GetGeoInfo # noqa: E402
import glob # noqa: E402
from osgeo import gdal # noqa: E402
import sys
from joblib import dump, load, Parallel, delayed
from skimage.io import imsave
from sklearn.metrics import confusion_matrix
from collections import Counter
import json
from osgeo import ogr
import geopandas
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point

from tqdm import tqdm
from functools import lru_cache

from tqdm.auto import tqdm
from joblib import Parallel

class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def get_data_by_coordinate_np(lats, lons, array, xmin, xres, ymax, yres):
    lat_inds = ((lats - ymax) / yres).astype(np.int64)
    lon_inds = ((lons - xmin) / xres).astype(np.int64)
    lat_ind_max, lon_ind_max = array.shape
    lat_ind_max -= 1
    lon_ind_max -= 1
    mask_lat = (lat_inds >= 0) & (lat_inds <= lat_ind_max)
    mask_lon = (lon_inds >= 0) & (lon_inds <= lon_ind_max)
    full_mask = mask_lat & mask_lon
    _res = np.empty_like(lats)
    _res[:] = np.nan
    _res[full_mask] = array[lat_inds[full_mask], lon_inds[full_mask]]
    return (_res, lats, lons)


@lru_cache(maxsize=None)
def preload_data_from_file(fname):
    print("Reading file", fname)
    data = gdal.Open(fname)
    geoinfo = data.GetGeoTransform()
    xmin = geoinfo[0]
    xres = geoinfo[1]
    ymax = geoinfo[3]
    yrot = geoinfo[4]
    xrot = geoinfo[2]
    yres = geoinfo[-1]
    array = data.ReadAsArray()
    return xmin, xres, ymax, yrot, xrot, yres, array


def get_data_features(lats, lons, path='./data/features/*.tif'):
    result = []
    names = []
    for fname in glob.glob(path):
        xmin, xres, ymax, yrot, xrot, yres, array = preload_data_from_file(fname)
        data_, lats_, lons_ = get_data_by_coordinate_np(lats, lons, array, xmin, xres, ymax, yres)
        result.append(data_.astype(np.float64).tolist())
        names.append(fname.split('\\')[-1].split('.')[-2])
    return dict((k, np.array(v)) for k, v in zip(names, result)), lats_, lons_


def read_tiff(fname):
    data = gdal.Open(fname)
    geoinfo = data.GetGeoTransform()
    NDV, xsize, ysize, *args = GetGeoInfo(fname)
    xmin = geoinfo[0]
    xres = geoinfo[1]
    ymax = geoinfo[3]
    yrot = geoinfo[4]
    xrot = geoinfo[2]
    yres = geoinfo[-1]
    array = data.ReadAsArray()
    lons = np.linspace(xmin, xmin + xsize * xres, xsize)
    lats = np.linspace(ymax, ymax + ysize * yres, ysize)
    LA, LO = np.meshgrid(lats, lons)
    array[array == NDV] = np.nan
    return array, LA, LO


def set_memmap_data(data):
    folder = './joblib_memmap'
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass

    data_filename_memmap = os.path.join(folder, 'data_memmap')
    dump(data, data_filename_memmap)
    return data_filename_memmap


def get_memmap_data(data_filename_memmap):
    mapped_data = load(data_filename_memmap, mmap_mode='r')
    return mapped_data


def check_name_valid(fname, wrong):
    current_fname = fname.split('\\')[-1].split('.')[-2]
    print(f"Testing fname: {current_fname[:-5]}.")
    return not (current_fname[:-5] in wrong)


def get_simple_windfall_polygon(shapefile='./data/wfl/kunashir_polygons.shp'):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataSource = driver.Open(shapefile, 0)
    layer = dataSource.GetLayer()
    featureCount = layer.GetFeatureCount()
    print(f"Total number of windfall regions: {featureCount}")
    total = 0
    for feature in layer:
        geometry = feature.GetGeometryRef()
        data = json.loads(geometry.ExportToJson())
        wfall_area = geometry.GetArea()
        yield (map(np.array, data['coordinates']), wfall_area)
        total += 1
    print(f"Total features were read: {total}")


def get_masked_windfall(DS=1):
    for array_map, area in get_simple_windfall_polygon():
        arrays = list(array_map)
        bbox_data = np.array([a.min(axis=0).tolist() + a.max(axis=0).tolist() for a in arrays])
        lon_min, lat_min = bbox_data.min(axis=0)[:2]
        lon_max, lat_max = bbox_data.max(axis=0)[2:]
        glons, glats = np.meshgrid(np.arange(lon_min, lon_max, DS), np.arange(lat_min, lat_max, DS))
        points = geopandas.points_from_xy(x=glons.ravel(), y=glats.ravel())
        result = np.vstack([points.within(Polygon(a)) for a in arrays]).any(axis=0)
        shape = glats.shape
        yield (glons.reshape(shape), glats.reshape(shape), result.reshape(shape), area)


def compute_inclusion(points_batch, memap_data):
    areas = memap_data.get('areas')
    polygons = memap_data.get('polygons')
    print(f"Taking points batch: {len(points_batch)}...")
    wfall = []
    wf_area = []
    for point in points_batch:
        p = Point(point[0], point[1])
        for poly, area in zip(polygons, areas):
            if poly.contains(p):
                wf_area.append(area)
                wfall.append(True)
                break
        else:
            wf_area.append(0)
            wfall.append(False)

    return wfall, wf_area, points_batch


def get_windfall_data(points, batch_size=10000, DS=5, n_jobs=5, shapefile='./data/wfl/kunashir_polygons.shp'):
    # load all polygons
    polygons = []
    areas = []
    for arrays, area in get_simple_windfall_polygon(shapefile):
        pols = list(map(lambda x: Polygon(x), arrays))
        pols = list(filter(lambda x: x.area > DS * DS, pols))
        polygons += pols
        areas += [area] * len(pols)
    print(f"Polygon dataset prepared..., {len(polygons)} polygons found.")

    mapped_fname = set_memmap_data({'areas': areas, 'polygons': polygons})

    results = ProgressParallel(n_jobs=n_jobs)(delayed(compute_inclusion)(points[i:i + batch_size], get_memmap_data(mapped_fname)) for i in range(0, len(points), batch_size))
    # compute_inclusion(points, get_memmap_data(mapped_fname))
    return results


if __name__ == "__main__":
    import pandas as pd
    from skimage.measure import label
    from matplotlib import pyplot as plt
    # Testing image reading (shape file reading )
    # for data in get_masked_windfall():
    #     print(data[-1])




    # =============================================================================

    # ------------------- experiment parameters ----------------------------------
    feature_intervals = {
        'aspect': np.arange(0, 370, 10),
        'slope' :  np.arange(0, 70, 2),
        #'forest-height': np.arange(0, 60, 1),
        'elevation': np.arange(0, 650, 10),
        'tree_cover': np.arange(20, 105, 2),
        'curvature': np.arange(-20, 20, 0.5),
        # 'curvature-plan': np.arange(-13, 13, 0.5),
        # 'curvature-prof': np.arange(-13, 13, 0.5),
        'patches': np.array([0, 10, 100, 1000, 10000, 100000, 1000000, 1500000, 2000000]),
        #'patches': np.array([0] + [np.exp(j) for j in range(1, 16)]),
        'morphology': np.arange(0, 12, 1)
    }


    common_path = './data'

    wfall_filename = os.path.join(common_path, 'Kunashir_wfall.tif')

    #---
    LON_MIN = 388726
    LON_MAX = 397959
    LAT_MIN = 4863294
    LAT_MAX = 4877011
    
    

    NPOINTS = 200000

    DS = 2  # Lat or Lon size
    # ----------------------------------------------------------------------------
    ind = 0
    skipped = 0
    total = dict()
    patch_size_cumulative = np.zeros(feature_intervals['patches'].shape)
    dropped_points = 0
    # ----- Containers; random choice of 1000 points from each tile ----
    X = []
    y = []
    # ------------------------------------------------------------------


    # LONS = np.random.uniform(LON_MIN, LON_MAX, NPOINTS)
    # LATS = np.random.uniform(LAT_MIN, LAT_MAX, NPOINTS)

    # points = np.vstack([LONS, LATS]).T


    # --------------- Get data for windfalls ----------------------------

    # results = get_windfall_data(points, batch_size=40000, n_jobs=8, DS=DS, shapefile='./wfall_data_biomass/windfalls.shp')

    # wfall_mask = []
    # wfall_area = []
    # wfall_coordinates = []
    # for mask, area, batch in results:
    #     wfall_mask += mask[:]
    #     wfall_area += area[:]
    #     wfall_coordinates.append(np.array(batch))

    # del results
    # wfall_coordinates = np.vstack(wfall_coordinates)


    # wfall_df = pd.DataFrame({'wfall_mask': wfall_mask, 'wfall_area': wfall_area, 'lats': wfall_coordinates[:, 1], 'lons': wfall_coordinates[:, 0]})


    # common_df = wfall_df.copy()


    common_df=load('bugs_veg_200k.dat')

    # for wfall_data in get_masked_windfall(DS=DS):

    lons = common_df.lons.values
    lats = common_df.lats.values

    # wdata = wfall_data[2].ravel()
    # shape = wfall_data[2].shape
    # area = wfall_data[-1]

    # # read wfall data
    # # wfall_data, lats, lons = read_tiff(wfall_filename)
    # # lats= lats[::4, ::4]
    # # lons= lons[::4, ::4]
    # # wfall_data= wfall_data[::4,::4]
    # # wfall_na = np.isnan(wfall_data.ravel())
    # # wfall_mask = (wfall_data == 2).astype(int)
    # # #_, xsize, ysize, *args = GetGeoInfo(wfall_filename)

    # xsize, ysize = shape

    # print("Data w-fall loaded: ", wdata.shape, lats.shape, lons.shape)

    # read auxiliary features data
    features, _, _ = get_data_features(lats.copy(), lons.copy())

    # for key in features:
    #     features[key] = features[key].reshape(xsize, ysize).T.ravel()
    features['curvature'] = features['curvature']/(10**10)
    features['lats'] = lats
    features['lons'] = lons

   # choice = (features['aspect'] != -1) * \
   #     (~np.isnan(features['morphology']))

   # forest_mask = choice * features['tree_cover'] > 20

    #pr_chosen = np.array(wfall_mask)

    result_features = pd.merge(common_df.loc[:, ['wfall_mask','bugs_area', 'wfall_area', 'lats', 'lons']], pd.DataFrame(features), on=['lats', 'lons'], validate='one_to_one')

    print(result_features.head())

    print("All data were read")

   # features['patches'] = np.array(wfall_area)
   # features['wfall'] = pr_chosen

    # ---------- Saving resutls for postprocessing analysis
    dump(result_features, 'bugs_cut_veg_200k.dat')
    result_features.to_csv('cut_bugs.csv')
    sys.exit(0)
    # ---------- End of monte carlo evaluations



    # ============== compute patch size layer ==========================
    # wfall_shape = wfall_mask.shape
    # wfall_mask = wfall_mask.ravel()
    # wfall_mask[~choice] = 0
    # pr_chosen[~choice] = False
    # wfall_mask = wfall_mask.reshape(wfall_shape)
    # labels = label(wfall_mask, connectivity=2, background=0)
    # lat_size, lon_size = abs(lats[0][1] - lats[0][0]), abs(lons[1][0] - lons[0][0])
    # features['patches'] = np.zeros_like(features['morphology'])
    # for lab in np.unique(labels.ravel()):
    #     if lab > 0:
    #         features['patches'][labels.ravel() == lab] = lat_size *\
    #             lon_size * (labels.ravel() == lab).sum()
    # --------------------------------------------------------------------





    # ============ load joblib model and apply it ===========
    # clf = load('rf_clf.joblib')
    # y_true = pr_chosen[:]
    # X = np.vstack([val[:] for key, val in features.items()]).T
    # X_ = X[:, [True, False,False, True,True,True,True,True,False]]
    # y_pred = clf.predict(X_)
    # imsave('y_true.png', y_true.reshape(wfall_shape).astype(np.uint8)*255)
    # imsave('y_pred.png', y_pred.reshape(wfall_shape).astype(np.uint8)*255)
    # print(confusion_matrix(y_true, y_pred))
    # print(Counter(y_true))
    # sys.exit(0)
    # =======================================================
