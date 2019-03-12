// ee_extract_landcover_by_country.js

// import the landcover dataset & the country shapefiles
var globcover = ee.FeatureCollection('ESA/GLOBCOVER_L4_200901_200912_V2_3')
var world_region = ee.FeatureCollection('ft:1tdSwUL7MVpOauSgRzqVTOwdfy17KDbw-1d9omPw')
var landcover = globcover.select('landcover');

// create Polygon for Ethiopia and Kenya
var eth = world_region.filterMetadata('Country','equals','Ethiopia').select(['landcover']);
var kenya = world_region.filterMetadata('Country','equals','Kenya');

// clip the landcover to the countries
var eth_lc = landcover.clip(eth);
var ken_lc = landcover.clip(kenya);

// export the landcover map
var scale  = 500;
var crs='EPSG:4326';

var task = Export.image.toDrive({
  image: eth_lc,
  description: 'ethiopia_landcover',
  scale: 1000,
  region: eth,
  folder: 'landcover'
})

var task_kenya = Export.image.toDrive({
  image: ken_lc,
  description: 'kenya_landcover',
  scale: 1000,
  region: kenya,
  folder: 'landcover'
})
