// Load the TIGER/USCensus dataset
var counties = ee.FeatureCollection("TIGER/2018/Counties");

// Define the counties and states
var countyNames = ['Garfield', 'Adams', 'Randolph', 'Harvey', 'Coahoma', 'Haskell'];
var stateNames = ['Washington', 'North Dakota', 'Indiana', 'Kansas', 'Mississippi', 'Texas'];
var stateFipsCodes = ['53', '38', '18', '20', '28', '48'];

// Function to get county features
function getCounty(state, county) {
  var stateCounty = counties.filter(ee.Filter.and(
    ee.Filter.eq('STATEFP', state),
    ee.Filter.eq('NAME', county)
  ));
  print('County:', county, 'State:', state, 'Feature:', stateCounty);
  return stateCounty;
}

// Set the years for which to download the data
var years = [2019, 2020, 2021];

// Function to process each site for each year
function processSite(county, state, years) {
  var countyFeature = getCounty(state, county);
  var bound = countyFeature.geometry();

  // Iterate over each year and create an image with each year as a band
  var images = years.map(function(year_int) {
    var year = year_int.toString();
    var startDay = year + "-01-01";
    var endDay = year + "-12-31";

    // Main code for downloading CDL image
    var CDL = ee.ImageCollection("USDA/NASS/CDL")
      .filterBounds(bound)
      .filterDate(startDay, endDay);

    var imageCount = CDL.size().getInfo();
    print('Year:', year, 'Image count:', imageCount);
    if (imageCount === 0) {
      print('No CDL images found for ' + county + ' in ' + year);
      return null;
    }

    var cdl = CDL.first().clip(bound);
    return cdl.select('cropland').rename('cropland_' + year);  // Rename band to include year
  });

  // Filter out null images
  images = images.filter(function(img) { return img !== null; });

  if (images.length === 0) {
    print('No valid CDL images found for ' + county);
    return;
  }

  // Combine the images for each year into a single image
  var combinedImage = ee.ImageCollection(images).toBands();

  // Display the combined image on the map
  Map.addLayer(combinedImage, {}, county + ' Combined CDL');

  // Download the combined image
  var cdl_format_name = county + "_CDL";
  Export.image.toDrive({
    image: combinedImage.toUint16(),
    description: cdl_format_name,
    region: bound,
    scale: 30,
    folder: county + "_CDL",
    crs: "EPSG:4326",
    fileDimensions: 512,
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF'
  });
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

// Process each site
countyNames.forEach(function(county, index) {
  processSite(county, stateFipsCodes[stateNames[index]], years);
});