/**
 * This script would allow users to batch export MOD10A1 and MYD10A1 albedo data for the Greenland ice sheet from Google Earth Engine (GEE) to Google Drive.
 * 
 * shunan.feng@envs.au.dk 
 */


/**
 * Preparation
 */
var aoi = /* color: #ffc82d */ee.Geometry.Polygon(
  [[[-36.29516924635421, 83.70737243835941],
    [-51.85180987135421, 82.75597137647488],
    [-61.43188799635421, 81.99879137488564],
    [-74.08813799635422, 78.10103528196419],
    [-70.13305987135422, 75.65372336709613],
    [-61.08032549635421, 75.71891096312955],
    [-52.20337237135421, 60.9795530382023],
    [-43.41430987135421, 58.59235996703347],
    [-38.49243487135421, 64.70478286561182],
    [-19.771731746354217, 69.72271161037442],
    [-15.728762996354217, 76.0828635948066],
    [-15.904544246354217, 79.45091003031243],
    [-10.015872371354217, 81.62328742628017],
    [-26.627200496354217, 83.43179828852398],
    [-31.636966121354217, 83.7553561747887]]]); // whole greenland


// var greenlandmask = ee.Image('OSU/GIMP/2000_ICE_OCEAN_MASK')
//                     .select('ice_mask'); //'ice_mask', 'ocean_mask'
// var glims = ee.Image().paint(ee.FeatureCollection('GLIMS/current'), 1);          
// var iceMask = ee.ImageCollection([
//   greenlandmask,
//   glims.rename('ice_mask')
// ]).mosaic().eq(1);

/**
 * Export view time of MOD data
 */

var startDate = ee.Date.fromYMD(2024, 1, 1);
var endDate = startDate.advance(3, 'month'); // End of the same year

var mod10 = ee.ImageCollection('MODIS/061/MOD10A1')
                  .filter(ee.Filter.date(startDate, endDate))
                  .filterBounds(aoi)
                  .select("Snow_Albedo_Daily_Tile"); 
// var myd10 = ee.ImageCollection('MODIS/061/MYD10A1')
//                   .filter(ee.Filter.date(startDate, endDate))
//                   .filterBounds(aoi)
//                   .select("Snow_Albedo_Daily_Tile");

var batch = require('users/fitoprincipe/geetools:batch');

// batch export
batch.Download.ImageCollection.toDrive(mod10, 'MOD10', 
                {scale: 500,
                 crs: 'EPSG:3413', // Greenland Polar Stereographic projection
                 region: aoi, 
                 type: 'uint8',
                 name: 'MOD10A1_{system_date}',
                 maxPixels: 1e13,
                });               
                
// batch.Download.ImageCollection.toDrive(myd10, 'MYD10', 
//                 {scale: 500,
//                  crs: 'EPSG:3413', // Greenland Polar Stereographic projection
//                  region: aoi, 
//                  type: 'uint8',
//                  name: 'MYD10A1_{system_date}',
//                  maxPixels: 1e13,
//                 });                