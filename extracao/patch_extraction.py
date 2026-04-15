# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 17:50:58 2026

@author: Marcel
"""

# %% Import Libraries

from pathlib import Path
from osgeo import gdal, ogr, osr
import numpy as np
import math

# %%

class PatchExtractor:
    def __init__(self, patch_size=256, 
                 overlap_percent=0.25, padding=False):
        if not (0 <= overlap_percent < 1.0): raise ValueError("overlap_percent must be >= 0 and < 1")
            
        self.patch_size = patch_size
        self.overlap_percent = overlap_percent
        self.overlap_px = int(patch_size * overlap_percent)
        
        self.stride = self.patch_size - self.overlap_px
        
        # We always move forward
        if self.stride < 1:
            raise ValueError('Stride must be greater than or equal to 1. Check overlap_percent value.')
        
        self.padding = padding
        
    def extract(self, array, get_coords=True):
        """
        array: 3D array [height, width, channels]
        """
        if len(array.shape) != 3:
            raise ValueError('Image must be in shape (Height, Width, Channels)') # Image shape restriction

        h, w, c = array.shape

        # Symmetric padding
        if self.padding:
            # 1. Calculate 'SAME' Padding (TensorFlow style)
            out_h = math.ceil(h / self.stride)
            out_w = math.ceil(w / self.stride)
            
            # Get how much it overflowed the border (if it happens)
            pad_h_total = max(0, (out_h - 1) * self.stride + self.patch_size - h)
            pad_w_total = max(0, (out_w - 1) * self.stride + self.patch_size - w)
            
            # If odd, bottom and right stay with 1 more pixel
            top, left = pad_h_total // 2, pad_w_total // 2
            bottom, right = pad_h_total - top, pad_w_total - left
            
            # 2. Apply Symmetric Padding
            array = np.pad(array, ((top, bottom), (left, right), (0,0)), mode='symmetric')
            
            # Global pixel offset (relative to original top-left 0,0)
            # Starting at -top/-left because the patch window starts "outside" the original image
            start_y, start_x = -top, -left
        # Only valid pixels
        else:
            # VALID: No padding, calculate how many full patches fit
            out_h = (h - self.patch_size) // self.stride + 1
            out_w = (w - self.patch_size) // self.stride + 1
            
            # Valid pixels start in (0, 0)
            start_y, start_x = 0, 0
            
        # 2. Extract Patches (6D view) (Batch, Out Rows, Out Cols, Patch Height, Patch Width, Channels)
        s_h, s_w, s_c = array.strides
        shape = (out_h, out_w, self.patch_size, self.patch_size, c)
        strides = (self.stride * s_h, self.stride * s_w, s_h, s_w, s_c)
        patches_6d = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
        
        # 3. Flatten to (Total Patches, H, W, C)
        patches_flat = patches_6d.reshape(-1, self.patch_size, self.patch_size, c)
        
        # 4. Generate Coordinates (Bounding Boxes in pixels)
        # Format: [ymin, xmin, ymax, xmax] relative to original image
        if get_coords:
            coords = []
            for i in range(out_h):
                for j in range(out_w):
                    ymin = start_y + (i * self.stride)
                    xmin = start_x + (j * self.stride)
                    ymax = ymin + self.patch_size
                    xmax = xmin + self.patch_size
                    coords.append([ymin, xmin, ymax, xmax])
                    
            return patches_flat, np.array(coords)
        
        return patches_flat
    
# %%

def pixel_to_map(px, py, gt):
    """
    Transforms pixel coordinates to map coordinates using a GDAL geotransform.
    Can be called without instantiating the class.
    """
    if gt is None:
        return px, py
    
    map_x = gt[0] + px * gt[1] + py * gt[2]
    map_y = gt[3] + px * gt[4] + py * gt[5]
    
    return map_x, map_y
    
# %%

def coords_to_vector(coords, output_path, source_path=None, layer_name="patches", driver_name="GPKG"):
    output_path = Path(output_path)
    gt, srs = None, None
    
    if source_path:
        ds_src = gdal.Open(str(Path(source_path)))
        if ds_src:
            gt = ds_src.GetGeoTransform()
            srs = osr.SpatialReference(wkt=ds_src.GetProjection())
            ds_src = None

    driver = ogr.GetDriverByName(driver_name)
    
    if output_path.exists():
        if driver_name == "GPKG":
            ds = driver.Open(str(output_path), update=1)
            for i in range(ds.GetLayerCount()):
                if ds.GetLayerByIndex(i).GetName() == layer_name:
                    ds.DeleteLayer(i)
                    break
        else:
            driver.DeleteDataSource(str(output_path))
            ds = driver.CreateDataSource(str(output_path))
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ds = driver.CreateDataSource(str(output_path))

    # Speed Boost 1: Set GDAL config options for GeoPackage
    gdal.SetConfigOption("OGR_SQLITE_SYNCHRONOUS", "OFF")

    layer = ds.CreateLayer(layer_name, srs, ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))

    # Speed Boost 2: Wrap feature creation in a TRANSACTION
    layer.StartTransaction()

    defn = layer.GetLayerDefn()
    for i, (ymin, xmin, ymax, xmax) in enumerate(coords):
        # 1. Calculate corners
        p1 = pixel_to_map(xmin, ymin, gt)
        p2 = pixel_to_map(xmax, ymin, gt)
        p3 = pixel_to_map(xmax, ymax, gt)
        p4 = pixel_to_map(xmin, ymax, gt)
        
        # 2. Build geometry using explicit x, y arguments
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint_2D(p1[0], p1[1])
        ring.AddPoint_2D(p2[0], p2[1])
        ring.AddPoint_2D(p3[0], p3[1])
        ring.AddPoint_2D(p4[0], p4[1])
        ring.AddPoint_2D(p1[0], p1[1]) # Close the ring
        
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        
        # 3. Create feature
        feat = ogr.Feature(defn)
        feat.SetGeometry(poly)
        feat.SetField("id", int(i))
        layer.CreateFeature(feat)

    # Commit all features at once
    layer.CommitTransaction()
    
    ds = None
    print(f"Exported {len(coords)} patches to {output_path}")

