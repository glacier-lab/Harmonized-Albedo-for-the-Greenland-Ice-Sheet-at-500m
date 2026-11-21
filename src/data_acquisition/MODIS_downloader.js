/**
 * This script would allow users to batch export MOD10A1 and MYD10A1 albedo data for the Greenland ice sheet from Google Earth Engine (GEE) to Google Drive.
 * 
 * shunan.feng@envs.au.dk 
 */


/**
 * Preparation
 */
// // arctic
// var roi = ee.Geometry.Polygon(
//   [[[-180, 60],
//     [-180, 84],
//     [180, 84],
//     [180, 60]]], null, false);
// Map.addLayer(roi, {color: 'red'}, 'ROI');


// // var greenlandmask = ee.Image('OSU/GIMP/2000_ICE_OCEAN_MASK')
// //                     .select('ice_mask'); //'ice_mask', 'ocean_mask'
// // var glims = ee.Image().paint(ee.FeatureCollection('GLIMS/current'), 1);          
// // var iceMask = ee.ImageCollection([
// //   greenlandmask,
// //   glims.rename('ice_mask')
// // ]).mosaic().eq(1);

// /**
//  * Export view time of MOD data
//  */

// var startDate = ee.Date.fromYMD(2024, 1, 1);
// var endDate = startDate.advance(3, 'month'); // End of the same year

// var mod10 = ee.ImageCollection('MODIS/061/MOD10A1')
//                   .filter(ee.Filter.date(startDate, endDate))
//                   .filterBounds(roi)
//                   .select("Snow_Albedo_Daily_Tile"); 
// // var myd10 = ee.ImageCollection('MODIS/061/MYD10A1')
// //                   .filter(ee.Filter.date(startDate, endDate))
// //                   .filterBounds(aoi)
// //                   .select("Snow_Albedo_Daily_Tile");

// var batch = require('users/fitoprincipe/geetools:batch');

// // batch export
// batch.Download.ImageCollection.toDrive(mod10, 'Arctic_MOD10', 
//                 {scale: 500,
//                  crs: 'EPSG:3413', // Greenland Polar Stereographic projection
//                  region: roi, 
//                  type: 'uint8',
//                  name: 'MOD10A1_{system_date}',
//                  maxPixels: 1e13,
//                 });               
                
// // batch.Download.ImageCollection.toDrive(myd10, 'Arctic_MYD10', 
// //                 {scale: 500,
// //                  crs: 'EPSG:3413', // Greenland Polar Stereographic projection
// //                  region: roi, 
// //                  type: 'uint8',
// //                  name: 'MYD10A1_{system_date}',
// //                  maxPixels: 1e13,
// //                 });                

// // antarctica
var roi = ee.FeatureCollection("projects/glacier-lab/assets/Coastline_Antarctica_v02").getBounds();
Map.addLayer(roi, {color: 'red'}, 'ROI');

var startDate = ee.Date.fromYMD(2024, 1, 1);
var endDate = startDate.advance(3, 'month'); // End of the same year

var mod10 = ee.ImageCollection('MODIS/061/MOD10A1')
                  .filter(ee.Filter.date(startDate, endDate))
                  .filterBounds(roi)
                  .select("Snow_Albedo_Daily_Tile"); 
// var myd10 = ee.ImageCollection('MODIS/061/MYD10A1')
//                   .filter(ee.Filter.date(startDate, endDate))
//                   .filterBounds(aoi)
//                   .select("Snow_Albedo_Daily_Tile");

var batch = require('users/fitoprincipe/geetools:batch');

// batch export
batch.Download.ImageCollection.toDrive(mod10, 'Antarctica_MOD10', 
                {scale: 500,
                 crs: 'EPSG:3031', // Antarctica Polar Stereographic projection
                 region: roi, 
                 type: 'uint8',
                 name: 'MOD10A1_{system_date}',
                 maxPixels: 1e13,
                });               
                
// batch.Download.ImageCollection.toDrive(myd10, 'Antarctica_MYD10', 
//                 {scale: 500,
//                  crs: 'EPSG:3413', // Greenland Polar Stereographic projection
//                  region: roi, 
//                  type: 'uint8',
//                  name: 'MYD10A1_{system_date}',
//                  maxPixels: 1e13,
//                 });                