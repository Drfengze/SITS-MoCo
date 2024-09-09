import logging
import threading
import multiprocessing
from pathlib import Path
from typing import Any, List, Tuple
from functools import partial
import os
import sys
import time
import ee
import requests
from rasterio.crs import CRS

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

os.chdir(sys.path[0])

counter = multiprocessing.Value('i', 0)
lock = multiprocessing.Lock()

def get_unique_id():
    with lock:
        counter.value += 1
        return counter.value

# 初始化 Earth Engine
try:
    ee.Initialize(project='ee-mrhuaize',opt_url='https://earthengine-highvolume.googleapis.com')
    log.info("Earth Engine initialized successfully.")
except Exception as e:
    log.error(f"Failed to initialize Earth Engine: {str(e)}")
    sys.exit(1)
class DownloadableGEEImage():
    lock = threading.Lock()

    def getdownload_url(self, image: ee.Image, bands: List[str], region: ee.Geometry, crs: str, scale: int) -> Tuple[requests.Response, str]:
        with self.lock:
            url = image.getDownloadURL(
                dict(format="GEO_TIFF", region=region, crs=crs, bands=bands, scale=scale)
            )
            return requests.get(url, stream=True), url

    def init(self, image: ee.Image):
        self.image = image

    def download(self, out: Path, region: ee.Geometry, crs: str, bands: List[str], scale: int, **kwargs: Any) -> None:
        max_retries = 10
        retry_delay = 20  # seconds

        for attempt in range(max_retries):
            try:
                log.info(f"Download attempt {attempt + 1} for {out}")
                crs = crs
                image = self.image.reproject(crs=crs, scale=scale).clip(region)
                response, url = self.getdownload_url(image, bands, region, crs, scale)
                
                download_size = int(response.headers.get("content-length", 0))
                log.info(f"Download size: {download_size} bytes")
                if download_size == 0 or not response.ok:
                    resp_dict = response.json()
                    if "error" in resp_dict and "message" in resp_dict["error"]:
                        msg = resp_dict["error"]["message"]
                        ex_msg = f"Error downloading tile: {msg}"
                    else:
                        ex_msg = str(response.json())
                    raise IOError(ex_msg)
                
                with open(out, "wb") as geojsonfile:
                    for data in response.iter_content(chunk_size=1024):
                        geojsonfile.write(data)
                log.info(f"Successfully downloaded image to {out}")
                break
            except Exception as e:
                log.error(f"Download error: {str(e)}", exc_info=True)
                if attempt < max_retries - 1:
                    log.warning(f"Download failed. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    log.error(f"Failed to download after {max_retries} attempts")
                    raise

def unique_values(collection, field):
    values = ee.Dictionary(collection.reduceColumns(ee.Reducer.frequencyHistogram(), [field]).get('histogram')).keys()
    return values

def daily_mosaics(imgs):
    def add_simple_time(img):
        d = ee.Date(img.get('system:time_start'))
        simple_date = ee.Date.fromYMD(d.get('year'), d.get('month'), d.get('day'))
        return img.set('simpleTime', simple_date.millis())

    imgs = imgs.map(add_simple_time)
    days = unique_values(imgs, 'simpleTime')

    def create_daily_mosaic(d):
        d = ee.Number.parse(d)
        d = ee.Date(d)
        t = imgs.filterDate(d, d.advance(1, 'day'))
        f = ee.Image(t.first())
        t = t.mosaic()
        t = t.set('system:time_start', d.millis())
        t = t.copyProperties(f)
        return t

    imgs = days.map(create_daily_mosaic)
    return ee.ImageCollection.fromImages(imgs)

def mask_s2_clouds(image):
    qa = image.select('QA60').int16()
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask)
def process_site(county, state, bound, years, data_source, scale):
    log.info(f"Processing site: {county}, {state}, {data_source}, {years}")
    county_feature = get_county(state, county)
    for year in years:
        start_day = ee.Date.fromYMD(year, 1, 1)
        end_day = ee.Date.fromYMD(year + 1, 1, 1)

        if data_source == 'S2':
            collection = ee.ImageCollection("COPERNICUS/S2_SR") \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80)) \
                .filterBounds(bound) \
                .filterDate(start_day, end_day)
            collection = process_s2(collection, bound)
            band_names = ['blue', 'green', 'red', 'red1', 'red2', 'red3', 'nir', 'red4', 'swir1', 'swir2']
            folder_name = f"{county}_S2_{year}"
        elif data_source == 'S1':
            collection = ee.ImageCollection("COPERNICUS/S1_GRD") \
                .filter(ee.Filter.eq('instrumentMode', 'IW')) \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
                .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')) \
                .filterBounds(bound) \
                .filterDate(start_day, end_day)
            collection = process_s1(collection, bound)
            band_names = ['VV', 'VH']
            folder_name = f"{county}_S1_{year}"
        else:
            log.error(f"Unsupported data source: {data_source}")
            return

        data = collection.map(lambda img: img.select(band_names).addBands(img.select(['doy'])) \
            .set('system:index', img.get('system:index')))

        i_size = data.size().getInfo()
        data = data.toList(i_size)

        for i in range(i_size):
            worker(i, data, bound, county, year, scale, folder_name)

def process_s2(collection, bound):
    def process_image(img):
        t = img.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']).divide(10000)
        t = t.addBands(img.select(['QA60']))
        out = t.copyProperties(img).copyProperties(img, ['system:time_start'])
        return out

    collection = collection.map(process_image) \
        .select(['QA60', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'], 
                ['QA60', 'aerosol', 'blue', 'green', 'red', 'red1', 'red2', 'red3', 'nir', 'red4', 'h2o', 'swir1', 'swir2']) \
        .map(lambda img: img.addBands(ee.Image.constant(ee.Date(img.get('system:time_start')).getRelative('day', 'year')).rename('doy').float()) \
            .set('doy', ee.Date(img.get('system:time_start')).getRelative('day', 'year'))) \
        .map(lambda img: img.clip(bound))

    collection = daily_mosaics(collection)
    collection = collection.map(mask_s2_clouds)

    return collection

def process_s1(collection, bound):
    return collection.select(['VV', 'VH']) \
        .map(lambda img: img.addBands(ee.Image.constant(ee.Date(img.get('system:time_start')).getRelative('day', 'year')).rename('doy').float()) \
            .set('doy', ee.Date(img.get('system:time_start')).getRelative('day', 'year'))) \
        .map(lambda img: img.clip(bound))

def worker(i, data, bound, county, year, scale, folder_name):
    try:
        format_name = f"{county}_{year}_{i + 1}"
        img = ee.Image(data.get(i))
        export_image(img, format_name, bound, scale, folder_name)
    except Exception as e:
        log.error(f"Error in worker function: {str(e)}")

def export_image(img, description, region, scale, folder):
    try:
        downloader = DownloadableGEEImage()
        downloader.init(img)
        
        output_path = Path(folder) / f"{description}_{get_unique_id()}.tif"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        crs = 'EPSG:4326'
        bands = img.bandNames().getInfo()
        
        downloader.download(out=output_path, region=region, crs=crs, bands=bands, scale=scale)
        log.info(f"Image downloaded to {output_path}")
    except Exception as e:
        log.error(f"Error exporting image: {str(e)}")

def process_grid(grid, county, state_fips, years, data_source, scale):
    try:
        bound = ee.Feature(grid).geometry()
        process_site(county, state_fips, bound, years, data_source, scale=scale)
    except Exception as e:
        log.error(f"Error processing grid: {str(e)}")
        return f"Error processing grid: {str(e)}"
def process_county_grids(county, state, data_source, years):
    try:
        log.info(f"Processing county grids: {county}, {state}, {data_source}, {years}")
        county_feature = get_county(state_fips_codes[state], county)
        grids = county_feature.geometry().coveringGrid(county_feature.geometry().projection(), 3000)
        grids_list = grids.toList(grids.size()).getInfo()
        
        process_grid_partial = partial(process_grid, 
                                       county=county, 
                                       state_fips=state_fips_codes[state],
                                       years=years, 
                                       data_source=data_source,
                                       scale=30)
        
        with multiprocessing.Pool(processes=5) as pool:
            results = pool.map(process_grid_partial, grids_list)
        
        for result in results:
            if isinstance(result, str) and result.startswith("Error"):
                log.error(result)
    except Exception as e:
        log.error(f"Error in process_county_grids: {str(e)}")
def get_county(state, county):
    counties = ee.FeatureCollection("TIGER/2018/Counties")
    return counties.filter(ee.Filter.And(
        ee.Filter.eq('STATEFP', state),
        ee.Filter.eq('NAME', county)
    ))
if __name__ == '__main__':
    try:
        years = [2020]
        county_names = ['Sumner']
        state_names = ['Kansas']
        state_fips_codes = {'Kansas': '20'}
        data_sources = ['S1']

        for county, state in zip(county_names, state_names):
            for data_source in data_sources:
                try:
                    process_county_grids(county, state, data_source, years)
                except Exception as e:
                    log.error(f"Error processing {county}, {state}, {data_source}: {str(e)}", exc_info=True)
    except Exception as e:
        log.error(f"An error occurred in main: {str(e)}", exc_info=True)