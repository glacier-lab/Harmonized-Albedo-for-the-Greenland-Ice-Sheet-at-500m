/**
 * @file iceMask.js
 * @description This file contains the ice mask for Greenland and the GLIMS dataset. 
 * It will export the ice mask to Google Drive for further analysis.
 * ROI: north of 60N
 */

var greenlandmask = ee.Image('OSU/GIMP/2000_ICE_OCEAN_MASK')
                    .select('ice_mask'); //'ice_mask', 'ocean_mask'
var glimsmask = ee.Image().paint(ee.FeatureCollection('GLIMS/current'), 1);          
var iceMask = ee.ImageCollection([
  greenlandmask,
  glimsmask.rename('ice_mask')
]).mosaic().eq(1);
// convert to uint8 to reduce file size
var iceMask = iceMask.toUint8();

Map.addLayer(iceMask, {min: 0, max: 1, palette: ['white', 'blue']}, 'Ice Mask');
// define roi between 60N and 84N
var roi = ee.Geometry.Polygon(
  [[[-180, 60],
    [-180, 84],
    [180, 84],
    [180, 60]]], null, false);
Map.addLayer(roi, {color: 'red'}, 'ROI');

// export ice mask to drive
Export.image.toDrive({
    image: iceMask.clip(roi),
    description: 'panArctic_ice_mask',
    folder: 'export',
    region: roi,
    scale: 500,
    crs: 'EPSG:3413', // Greenland Polar Stereographic projection
    maxPixels: 1e13
});
// export roi to drive as shapefile
Export.table.toDrive({
    collection: ee.FeatureCollection(roi),
    description: 'panArctic_roi',
    folder: 'export',
    fileFormat: 'SHP'
});