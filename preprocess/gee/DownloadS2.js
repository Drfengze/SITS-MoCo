// ----------------------------------- //
// ------------ Variables ------------ //
// ----------------------------------- //
var years = [2019, 2020, 2021];
var field_name = 'Adams';  // Update this to process other counties as well
var countyNames = ['Garfield', 'Adams', 'Randolph', 'Harvey', 'Coahoma', 'Haskell'];
var stateNames = ['Washington', 'North Dakota', 'Indiana', 'Kansas', 'Mississippi', 'Texas'];
var bandnames = ['QA60', 'aerosol', 'blue', 'green', 'red','red1','red2','red3','nir','red4','h2o', 'swir1', 'swir2'];
var bandnames_ex = ['blue', 'green', 'red', 'red1', 'red2', 'red3', 'nir', 'red4', 'swir1', 'swir2'];
var num_dict_2022 = {
  Adams: 112,
};

// Load the TIGER/USCensus dataset
var counties = ee.FeatureCollection("TIGER/2018/Counties");

// Function to get county features
function getCounty(state, county) {
  return counties.filter(ee.Filter.and(
    ee.Filter.eq('STATEFP', state),
    ee.Filter.eq('NAME', county)
  ));
}

// Convert state names to FIPS codes
var stateFipsCodes = {
  'Washington': '53',
  'North Dakota': '38',
  'Indiana': '18',
  'Kansas': '20',
  'Mississippi': '28',
  'Texas': '48'
};

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