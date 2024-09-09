// ----------------------------------- //
// ------------ Variables ------------ //
// ----------------------------------- //
var years = [2019, 2020, 2021];
var field_name = 'Adams';  // Update this to process other counties as well
county_names = ['Garfield', 'Adams', 'Harvey', 'Haskell', 'Minidoka', 'Jerome', 'Cassia']
stateNames = ['Washington', 'North Dakota', 'Kansas', 'Texas', 'Idaho', 'Idaho', 'Idaho']
stateFipsCodes = {'Washington': '53', 'North Dakota': '38', 'Kansas': '20', 'Texas': '48', 'Idaho': '16'}
var bandnames = ['QA60', 'aerosol', 'blue', 'green', 'red','red1','red2','red3','nir','red4','h2o', 'swir1', 'swir2'];
var bandnames_ex = ['blue', 'green', 'red', 'red1', 'red2', 'red3', 'nir', 'red4', 'swir1', 'swir2'];

// Load the TIGER/USCensus dataset
var counties = ee.FeatureCollection("TIGER/2018/Counties");

// Function to get county features
function getCounty(state, county) {
  return counties.filter(ee.Filter.and(
    ee.Filter.eq('STATEFP', state),
    ee.Filter.eq('NAME', county)
  ));
}


// Function to process each site for each year
function processSite(county, state, years) {
  var countyFeature = getCounty(state, county);
  var bound = countyFeature.geometry();

  // Process each year
  years.forEach(function(year_int) {
    var year = year_int.toString();
    var startDay = ee.Date.fromYMD(year_int, 1, 1);
    var endDay = ee.Date.fromYMD(year_int + 1, 1, 1);

    var s2s = ee.ImageCollection("COPERNICUS/S2_SR")
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
      .filterBounds(bound)
      .filterDate(startDay, endDay);

    // Rename S2 band names
    s2s = s2s
      .map(function(img){
        var t = img.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']).divide(10000); // Rescale to 0-1
        t = t.addBands(img.select(['QA60']));
        var out = t.copyProperties(img).copyProperties(img, ['system:time_start']);
        return out;
      })
      .select(['QA60', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'], bandnames)
      .map(function(img){
        var doy = ee.Date(img.get('system:time_start')).getRelative('day', 'year');
        return img
          .addBands(ee.Image.constant(doy).rename('doy').float())
          .set('doy', doy);
      })
      .map(function(img){
        return img.clip(bound);
      });

    // Daily mosaic
    s2s = dailyMosaics(s2s);

    // Mask cloud
    s2s = s2s.map(maskS2clouds);

    // Download image
    var data = s2s.map(function(img){
      return img
        .select(bandnames_ex).multiply(10000)
        .addBands(img.select(['doy']))
        .set('system:index', img.get('system:index'));
    });

    var i_size = data.size();
    data = data.toList(i_size);

    for (var i = 0; i < num_dict_2022[field_name]; i++) {
      worker(i, data, bound, county, year);
    }
  });
}

// Function to handle the download process
function worker(i, data, bound, county, year) {
  var s2_format_name = county + "_" + year + "_" + (i + 1);

  i = ee.Number(i);
  var img = ee.Image(data.get(i));

  Export.image.toDrive({
    image: img.toUint16(),
    description: s2_format_name,
    region: bound,
    scale: 30,
    folder: county + "_S2_" + year,
    crs: "EPSG:4326",
    fileDimensions: 512,
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF'
  });
}

// Define the dailyMosaics function
function dailyMosaics(imgs) {
  // Simplify date to exclude time of day
  imgs = imgs.map(function(img) {
    var d = ee.Date(img.get('system:time_start'));
    var day = d.get('day');
    var m = d.get('month');
    var y = d.get('year');
    var simpleDate = ee.Date.fromYMD(y, m, day);
    return img.set('simpleTime', simpleDate.millis());
  });

  // Find the unique days
  var days = uniqueValues(imgs, 'simpleTime');

  imgs = days.map(function(d) {
    d = ee.Number.parse(d);
    d = ee.Date(d);
    var t = imgs.filterDate(d, d.advance(1, 'day'));
    var f = ee.Image(t.first());
    t = t.mosaic();
    t = t.set('system:time_start', d.millis());
    t = t.copyProperties(f);
    return t;
  });

  imgs = ee.ImageCollection.fromImages(imgs);
  return imgs;
}

// Function to mask clouds using the Sentinel-2 QA band.
function maskS2clouds(image) {
  var qa = image.select('QA60').int16();

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = Math.pow(2, 10);
  var cirrusBitMask = Math.pow(2, 11);

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0).and(
             qa.bitwiseAnd(cirrusBitMask).eq(0));

  // Return the masked and scaled data.
  return image.updateMask(mask);
}

// Function to find unique values of a field in a collection
function uniqueValues(collection, field) {
  var values = ee.Dictionary(collection.reduceColumns(ee.Reducer.frequencyHistogram(), [field]).get('histogram')).keys();
  return values;
}

// Process each site
countyNames.forEach(function(county, index) {
  processSite(county, stateFipsCodes[stateNames[index]], years);
});
// ----------------------------------- //
// ------------ Variables ------------ //
// ----------------------------------- //
var years = [2019, 2020, 2021];
var field_name = 'Adams';  // Update this to process other counties as well
var county_names = ['Garfield', 'Adams', 'Harvey', 'Haskell', 'Minidoka', 'Jerome', 'Cassia'];
var stateNames = ['Washington', 'North Dakota', 'Kansas', 'Texas', 'Idaho', 'Idaho', 'Idaho'];
var stateFipsCodes = {'Washington': '53', 'North Dakota': '38', 'Kansas': '20', 'Texas': '48', 'Idaho': '16'};
var bandNames = ['VV', 'VH'];

// Load the TIGER/USCensus dataset
var counties = ee.FeatureCollection("TIGER/2018/Counties");

// Function to get county features
function getCounty(state, county) {
  return counties.filter(ee.Filter.and(
    ee.Filter.eq('STATEFP', state),
    ee.Filter.eq('NAME', county)
  ));
}

// Function to process each site for each year
function processSite(county, state, years) {
  var countyFeature = getCounty(state, county);
  var bound = countyFeature.geometry();

  // Process each year
  years.forEach(function(year_int) {
    var year = year_int.toString();
    var startDay = ee.Date.fromYMD(year_int, 1, 1);
    var endDay = ee.Date.fromYMD(year_int + 1, 1, 1);

    var s1 = ee.ImageCollection("COPERNICUS/S1_GRD")
      .filter(ee.Filter.eq('instrumentMode', 'IW'))
      .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
      .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
      .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
      .filterBounds(bound)
      .filterDate(startDay, endDay);

    // Rename S1 band names and add DOY
    s1 = s1
      .select(['VV', 'VH'])
      .map(function(img){
        var doy = ee.Date(img.get('system:time_start')).getRelative('day', 'year');
        return img
          .addBands(ee.Image.constant(doy).rename('doy').float())
          .set('doy', doy);
      })
      .map(function(img){
        return img.clip(bound);
      });

    // Daily mosaic
    s1 = dailyMosaics(s1);

    // Download image
    var data = s1.map(function(img){
      return img
        .select(bandNames)
        .addBands(img.select(['doy']))
        .set('system:index', img.get('system:index'));
    });

    var i_size = data.size();

    data = data.toList(i_size);

    for (var i = 0; i < 100; i++) {
      worker(i, data, bound, county, year);
    }
  });
}

// Function to handle the download process
function worker(i, data, bound, county, year) {
  var s1_format_name = county + "_" + year + "_" + (i + 1);

  i = ee.Number(i);
  var img = ee.Image(data.get(i));

  Export.image.toDrive({
    image: img.toFloat(),
    description: s1_format_name,
    region: bound,
    scale: 30,
    folder: county + "_S1_" + year,
    crs: "EPSG:4326",
    fileDimensions: 512,
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF'
  });
}

// Define the dailyMosaics function
function dailyMosaics(imgs) {
  // Simplify date to exclude time of day
  imgs = imgs.map(function(img) {
    var d = ee.Date(img.get('system:time_start'));
    var day = d.get('day');
    var m = d.get('month');
    var y = d.get('year');
    var simpleDate = ee.Date.fromYMD(y, m, day);
    return img.set('simpleTime', simpleDate.millis());
  });

  // Find the unique days
  var days = uniqueValues(imgs, 'simpleTime');

  imgs = days.map(function(d) {
    d = ee.Number.parse(d);
    d = ee.Date(d);
    var t = imgs.filterDate(d, d.advance(1, 'day'));
    var f = ee.Image(t.first());
    t = t.mosaic();
    t = t.set('system:time_start', d.millis());
    t = t.copyProperties(f);
    return t;
  });

  imgs = ee.ImageCollection.fromImages(imgs);
  return imgs;
}

// Function to find unique values of a field in a collection
function uniqueValues(collection, field) {
  var values = ee.Dictionary(collection.reduceColumns(ee.Reducer.frequencyHistogram(), [field]).get('histogram')).keys();
  return values;
}

// Process each site
county_names.forEach(function(county, index) {
  processSite(county, stateFipsCodes[stateNames[index]], years);
});
// ----------------------------------- //
// ------------ Variables ------------ //
// ----------------------------------- //
var years = [2019, 2020, 2021];
var county_names = ['Garfield', 'Adams', 'Harvey', 'Haskell', 'Minidoka', 'Jerome', 'Cassia'];
var stateNames = ['Washington', 'North Dakota', 'Kansas', 'Texas', 'Idaho', 'Idaho', 'Idaho'];
var stateFipsCodes = {'Washington': '53', 'North Dakota': '38', 'Kansas': '20', 'Texas': '48', 'Idaho': '16'};
var bandNames = ['Nadir_Reflectance_Band1', 'Nadir_Reflectance_Band2', 'Nadir_Reflectance_Band3', 
                 'Nadir_Reflectance_Band4', 'Nadir_Reflectance_Band5', 'Nadir_Reflectance_Band6', 
                 'Nadir_Reflectance_Band7'];

// Load the TIGER/USCensus dataset
var counties = ee.FeatureCollection("TIGER/2018/Counties");

// Function to get county features
function getCounty(state, county) {
  return counties.filter(ee.Filter.and(
    ee.Filter.eq('STATEFP', state),
    ee.Filter.eq('NAME', county)
  ));
}

// Function to process each site for each year
function processSite(county, state, years) {
  var countyFeature = getCounty(state, county);
  var bound = countyFeature.geometry();

  // Process each year
  years.forEach(function(year_int) {
    var year = year_int.toString();
    var startDay = ee.Date.fromYMD(year_int, 1, 1);
    var endDay = ee.Date.fromYMD(year_int + 1, 1, 1);

    var modis = ee.ImageCollection("MODIS/061/MCD43A4")
      .filterBounds(bound)
      .filterDate(startDay, endDay)
      .select(bandNames);

    // Add DOY band
    modis = modis.map(function(img){
      var doy = ee.Date(img.get('system:time_start')).getRelative('day', 'year');
      return img
        .addBands(ee.Image.constant(doy).rename('doy').float())
        .set('doy', doy);
    });

    // Clip to county boundary
    modis = modis.map(function(img){
      return img.clip(bound);
    });

    var i_size = modis.size();
    var modis_list = modis.toList(i_size);

    // Use ee.List.sequence to create a list of indices
    var indices = ee.List.sequence(0, i_size.subtract(1));

    // Map over the indices to create export tasks
    indices.evaluate(function(indices) {
      indices.forEach(function(i) {
        worker(i, modis_list, bound, county, year);
      });
    });
  });
}

function worker(i, data, bound, county, year) {
  var modis_format_name = county + "_" + year + "_" + (i + 1);

  var img = ee.Image(data.get(i));

  Export.image.toDrive({
    image: img.toFloat(),
    description: modis_format_name,
    region: bound,
    scale: 500,  // MODIS resolution
    folder: county + "_MODIS_" + year,
    crs: "EPSG:4326",
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF'
  });
}

// Process each site
county_names.forEach(function(county, index) {
  processSite(county, stateFipsCodes[stateNames[index]], years);
});
// 定义基本变量
var years = [2019, 2020, 2021];
var county_names = ['Garfield', 'Adams', 'Harvey', 'Haskell', 'Minidoka', 'Jerome', 'Cassia'];
var stateNames = ['Washington', 'North Dakota', 'Kansas', 'Texas', 'Idaho', 'Idaho', 'Idaho'];
var stateFipsCodes = {'Washington': '53', 'North Dakota': '38', 'Kansas': '20', 'Texas': '48', 'Idaho': '16'};

// 加载ERA5和ERA5-Land数据集
// var era5 = ee.ImageCollection("ECMWF/ERA5/DAILY");
var era5Land = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR");

// 定义需要的变量
var era5Vars = [];
var era5LandVars = ['temperature_2m','total_precipitation_sum','volumetric_soil_water_layer_1'];

// 获取县级特征
var counties = ee.FeatureCollection("TIGER/2018/Counties");

// 获取指定县的特征
function getCounty(state, county) {
  return counties.filter(ee.Filter.and(
    ee.Filter.eq('STATEFP', state),
    ee.Filter.eq('NAME', county)
  ));
}

// 处理每个地点每个年份的数据
function processSite(county, state, years) {
  var countyFeature = getCounty(state, county);
  var bound = countyFeature.geometry();

  // 处理每个年份的数据
  years.forEach(function(year) {
    var startDay = ee.Date.fromYMD(year, 1, 1);
    var endDay = ee.Date.fromYMD(year + 1, 1, 1);

    // 分别处理每个变量
    era5Vars.concat(era5LandVars).forEach(function(variable) {
      var data;
      if (era5Vars.indexOf(variable) !== -1) {  // 使用indexOf代替includes
        data = era5.filterBounds(bound).filterDate(startDay, endDay).select([variable]);
      } else {
        data = era5Land.filterBounds(bound).filterDate(startDay, endDay).select([variable]);
      }

      // 将年度数据合并为单个图像
      var yearlyImage = data.toBands().rename(data.aggregate_array('system:index'));

      // 设置图像属性
      yearlyImage = yearlyImage.set({
        'year': year,
        'region': county,
        'variable': variable
      });

      // 导出图像
      Export.image.toDrive({
        image: yearlyImage.toFloat(),
        description: county + "_" + variable + "_" + year,
        region: bound,
        scale: 500,
        folder: county + "_ERA5_" + year,
        crs: "EPSG:4326",
        maxPixels: 1e13,
        fileFormat: 'GeoTIFF'
      });
    });
  });
}

// 处理每个县
county_names.forEach(function(county, index) {
  processSite(county, stateFipsCodes[stateNames[index]], years);
});
// 定义基本变量
var years = [2019, 2020, 2021];
var county_names = ['Garfield', 'Adams', 'Harvey', 'Haskell', 'Minidoka', 'Jerome', 'Cassia'];
var stateNames = ['Washington', 'North Dakota', 'Kansas', 'Texas', 'Idaho', 'Idaho', 'Idaho'];
var stateFipsCodes = {'Washington': '53', 'North Dakota': '38', 'Kansas': '20', 'Texas': '48', 'Idaho': '16'};

// 加载ERA5和ERA5-Land数据集
// var era5 = ee.ImageCollection("ECMWF/ERA5/DAILY");
var era5Land = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR");

// 定义需要的变量
var era5Vars = [];
var era5LandVars = ['temperature_2m','total_precipitation_sum','volumetric_soil_water_layer_1'];

// 获取县级特征
var counties = ee.FeatureCollection("TIGER/2018/Counties");

// 获取指定县的特征
function getCounty(state, county) {
  return counties.filter(ee.Filter.and(
    ee.Filter.eq('STATEFP', state),
    ee.Filter.eq('NAME', county)
  ));
}

// 处理每个地点每个年份的数据
function processSite(county, state, years) {
  var countyFeature = getCounty(state, county);
  var bound = countyFeature.geometry();

  // 处理每个年份的数据
  years.forEach(function(year) {
    var startDay = ee.Date.fromYMD(year, 1, 1);
    var endDay = ee.Date.fromYMD(year + 1, 1, 1);

    // 分别处理每个变量
    era5Vars.concat(era5LandVars).forEach(function(variable) {
      var data;
      if (era5Vars.indexOf(variable) !== -1) {  // 使用indexOf代替includes
        data = era5.filterBounds(bound).filterDate(startDay, endDay).select([variable]);
      } else {
        data = era5Land.filterBounds(bound).filterDate(startDay, endDay).select([variable]);
      }

      // 将年度数据合并为单个图像
      var yearlyImage = data.toBands().rename(data.aggregate_array('system:index'));

      // 设置图像属性
      yearlyImage = yearlyImage.set({
        'year': year,
        'region': county,
        'variable': variable
      });

      // 导出图像
      Export.image.toDrive({
        image: yearlyImage.toFloat(),
        description: county + "_" + variable + "_" + year,
        region: bound,
        scale: 500,
        folder: county + "_ERA5_" + year,
        crs: "EPSG:4326",
        maxPixels: 1e13,
        fileFormat: 'GeoTIFF'
      });
    });
  });
}

// 处理每个县
county_names.forEach(function(county, index) {
  processSite(county, stateFipsCodes[stateNames[index]], years);
});
