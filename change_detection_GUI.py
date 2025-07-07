#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from osgeo import gdal, osr
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk # Import ttk for Combobox
import threading

"""
This script performs change detection and classifies the results.
It supports analysis on a specific Area of ​​Interest (AOI).
It writes classified change detection raster statistics to console.

This script is the GUI for the change_detection.py script that 
you can find in the same repository. To read the comments and more 
refer to the change_detection.py script
"""

PREDEFINED_NATURAL_CLASSES = {
    "land_use": {11, 12, 13, 14, 16, 2, 5, 61, 62},
    "land_cover": {1210, 1220, 2111, 2112, 2120, 2211, 2212, 3100, 3200, 4000},
    "land_consumption": {2, 201, 202, 203, 204, 205},
}

PREDEFINED_ANTHROPIC_CLASSES = {
    "land_use": {3, 4},
    "land_cover": {1110, 1120, 1100},
    "land_consumption": {1, 11, 12, 111, 112, 113, 114, 115, 116, 117, 118, 121, 122, 123, 124, 125, 126},
}

def import_raster(filepath, output_text_widget):
    """
    Imports a georeferenced raster.

    Args:
        filepath (str): The path to the raster file.
        output_text_widget (tk.scrolledtext.ScrolledText): Widget to display messages.

    Returns:
        gdal.Dataset: The GDAL dataset object of the raster.
                      Returns None if the file cannot be opened.
    """
    if not os.path.exists(filepath):
        output_text_widget.insert(tk.END, f"Error: The file {filepath} does not exist.\n")
        output_text_widget.see(tk.END)
        return None
    
    dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
    if dataset is None:
        output_text_widget.insert(tk.END, f"Error: Unable to open raster file {filepath}. Check format or corruption.\n")
        output_text_widget.see(tk.END)
    return dataset

def get_raster_info(dataset, output_text_widget):
    """
    Reads data and metadata from a raster.

    Args:
        dataset (gdal.Dataset): The GDAL dataset object of the raster.
        output_text_widget (tk.scrolledtext.ScrolledText): Widget to display messages.

    Returns:
        dict: A dictionary containing raster information (pixel count, extent, resolution).
              Returns None if the dataset is None.
    """
    if dataset is None:
        return None

    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    transform = dataset.GetGeoTransform()
    
    if transform:
        x_min = transform[0]
        y_max = transform[3]
        x_res = transform[1]
        y_res = transform[5]

        x_max = x_min + cols * x_res
        y_min = y_max + rows * y_res

        extent = {
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max
        }
        resolution = (x_res, abs(y_res)) # Use absolute value for Y resolution
    else:
        extent = None
        resolution = None
        output_text_widget.insert(tk.END, "Warning: Unable to get georeferencing (transformation) information for raster.\n")
        output_text_widget.see(tk.END)

    info = {
        'rows': rows,
        'cols': cols,
        'num_pixels': rows * cols,
        'extent': extent,
        'resolution': resolution,
        'projection': dataset.GetProjection()
    }
    return info

def compare_rasters_pixel_by_pixel(data_t0, data_t1, output_text_widget):
    """
    Compares two raster arrays pixel by pixel and returns a difference array.
    Different pixels will have value 1, otherwise 0.

    Args:
        data_t0 (numpy.ndarray): Array of pixel values from raster T0.
        data_t1 (numpy.ndarray): Array of pixel values from raster T1.
        output_text_widget (tk.scrolledtext.ScrolledText): Widget to display messages.

    Returns:
        numpy.ndarray: A 2D array containing difference values (1 for change, 0 for no change).
                       Returns None if rasters are not comparable.
    """
    if data_t0.shape != data_t1.shape:
        output_text_widget.insert(tk.END, "Error: Raster arrays have different dimensions and cannot be compared pixel by pixel.\n")
        output_text_widget.insert(tk.END, f"Dimensions T0: {data_t0.shape}, Dimensions T1: {data_t1.shape}\n")
        output_text_widget.see(tk.END)
        return None
    
    difference_array = np.zeros_like(data_t0, dtype=np.uint8)
    difference_array[data_t0 != data_t1] = 1

    return difference_array

def classify_change_detection(data_t0, data_t1, natural_classes, anthropic_classes):
    """
    Classifies detected changes based on natural and anthropic area categories.

    Args:
        data_t0 (numpy.ndarray): Array of pixel values from raster T0.
        data_t1 (numpy.ndarray): Array of pixel values from raster T1.
        natural_classes (set): Set of integer class codes considered natural.
        anthropic_classes (set): Set of integer class codes considered anthropic.

    Returns:
        numpy.ndarray: A classified array where:
                        1: Unchanged Anthropic Areas
                        2: Unchanged Natural Areas
                        3: From Natural to Anthropic Areas
                        4: From Anthropic to Natural Areas
                        0: Other types of change (or NoData if applicable, not fitting categories)
    """
    
    # Initialize output array with zeros
    classified_change_array = np.zeros_like(data_t0, dtype=np.uint8)

    # Ensure arrays are integer type for class comparisons
    data_t0_int = data_t0.astype(int)
    data_t1_int = data_t1.astype(int)

    # Membership in the set is considered discriminant, not the pixel value itself.
    # This is because in some cases, pixels referring to the same location may report
    # different values for different years but still be traceable to the anthropic or natural class.

    # 1. Unchanged Anthropic Areas (value 1)
    mask_anthropic_invariate = np.isin(data_t0_int, list(anthropic_classes)) & \
                               np.isin(data_t1_int, list(anthropic_classes))
    classified_change_array[mask_anthropic_invariate] = 1

    # 2. Unchanged Natural Areas (value 2)
    mask_natural_invariate = np.isin(data_t0_int, list(natural_classes)) & \
                             np.isin(data_t1_int, list(natural_classes))
    classified_change_array[mask_natural_invariate] = 2

    # 3. From Natural to Anthropic Areas (value 3)
    mask_natural_to_anthropic = np.isin(data_t0_int, list(natural_classes)) & \
                                np.isin(data_t1_int, list(anthropic_classes))
    classified_change_array[mask_natural_to_anthropic] = 3

    # 4. From Anthropic to Natural Areas (value 4)
    mask_anthropic_to_natural = np.isin(data_t0_int, list(anthropic_classes)) & \
                                np.isin(data_t1_int, list(natural_classes))
    classified_change_array[mask_anthropic_to_natural] = 4

    return classified_change_array

def export_result_raster(array_to_export, output_filepath, geotransform, projection, output_text_widget):
    """
    Exports the resulting raster (difference or classified).

    Args:
        array_to_export (numpy.ndarray): The numpy array containing values to export.
        output_filepath (str): The path and name of the output raster file.
        geotransform (tuple): The 6-element geotransform tuple for the output raster.
        projection (str): The WKT string of the projection for the output raster.
        output_text_widget (tk.scrolledtext.ScrolledText): Widget to display messages.

    Returns:
        bool: True if export was successful, False otherwise.
    """
    rows, cols = array_to_export.shape
    driver = gdal.GetDriverByName('GTiff')
    
    # Determine GDAL data type based on NumPy array data type
    # For values 0-4, GDT_Byte (8-bit unsigned integer) is sufficient
    gdal_data_type = gdal.GDT_Byte    
    
    out_dataset = driver.Create(output_filepath, cols, rows, 1, gdal_data_type)
    
    if out_dataset is None:
        output_text_widget.insert(tk.END, f"Error: Unable to create output file {output_filepath}.\n")
        output_text_widget.see(tk.END)
        return False

    out_dataset.SetGeoTransform(geotransform)
    out_dataset.SetProjection(projection)

    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(array_to_export)
    
    out_band.SetNoDataValue(0)    

    out_band.FlushCache()
    out_dataset = None    

    output_text_widget.insert(tk.END, f"Raster successfully exported to: {output_filepath}\n")
    output_text_widget.see(tk.END)
    return True

def write_change_statistics(classified_change_array, output_text_widget):
    """
    Writes statistics of the classified change detection raster to console/GUI.

    Args:
        classified_change_array (numpy.ndarray): The numpy array containing classified change values.
        output_text_widget (tk.scrolledtext.ScrolledText): Widget to display messages.

    Returns:
        bool: True if writing was successful, False otherwise.
    """
    if classified_change_array is None:
        output_text_widget.insert(tk.END, "Error: The classified array is None; unable to calculate statistics.\n")
        output_text_widget.see(tk.END)
        return False

    total_pixels = classified_change_array.size
    
    # Count pixels for each class
    anthropic_invariate_pixels = np.sum(classified_change_array == 1)
    natural_invariate_pixels = np.sum(classified_change_array == 2)
    natural_to_anthropic_pixels = np.sum(classified_change_array == 3)
    anthropic_to_natural_pixels = np.sum(classified_change_array == 4)
    
    unclassified_pixels = np.sum(classified_change_array == 0)

    try:
        output_text_widget.insert(tk.END, "\n--- Classified Change Detection Raster Statistics ---\n")
        output_text_widget.insert(tk.END, f"Total number of pixels: {total_pixels}\n")
        output_text_widget.insert(tk.END, f"1) Unchanged Anthropic Areas: {anthropic_invariate_pixels} pixel ({((anthropic_invariate_pixels / total_pixels) * 100):.2f}%)\n")
        output_text_widget.insert(tk.END, f"2) Unchanged Natural Areas: {natural_invariate_pixels} pixel ({((natural_invariate_pixels / total_pixels) * 100):.2f}%)\n")
        output_text_widget.insert(tk.END, f"3) From Natural to Anthropic Areas: {natural_to_anthropic_pixels} pixel ({((natural_to_anthropic_pixels / total_pixels) * 100):.2f}%)\n")
        output_text_widget.insert(tk.END, f"4) From Anthropic to Natural Areas: {anthropic_to_natural_pixels} pixel ({((anthropic_to_natural_pixels / total_pixels) * 100):.2f}%)\n")
        output_text_widget.insert(tk.END, f"5) Unclassified pixels (other types of changes or areas not classified): {unclassified_pixels} pixel ({((unclassified_pixels / total_pixels) * 100):.2f}%)\n")
        output_text_widget.see(tk.END)
            
        return True
    except Exception as e:
        output_text_widget.insert(tk.END, f"Error writing statistics: {e}\n")
        output_text_widget.see(tk.END)
        return False

def world_to_pixel(geotransform, x, y):
    """
    Converts world coordinates (geographic or projected) to pixel/line coordinates.
    
    Args:
        geotransform (tuple): The 6-element geotransform tuple (gdal.GetGeoTransform()).
        x (float): The X coordinate of the point.
        y (float): The Y coordinate of the point.
        
    Returns:
        tuple: (column, row) - The corresponding pixel and row coordinates.
    """
    ulx, xres, xskew, uly, yskew, yres = geotransform
    
    if xskew == 0 and yskew == 0:
        px = (x - ulx) / xres
        py = (y - uly) / yres
    else:
        det = xres * yres - xskew * yskew
        if det == 0:
            raise ValueError("Cannot invert geotransform matrix (determinant is zero).")
        px = (yres * (x - ulx) - xskew * (y - uly)) / det
        py = (xres * (y - uly) - yskew * (x - ulx)) / det

    # Returns float, conversion to int occurs where necessary (e.g., slicing)
    return (px, py)

def check_aoi_overlap(full_data_t0_array, full_data_t1_array, dataset_t0, dataset_t1, aoi_coords, output_text_widget):
    """
    Checks if the AOI is within the overlap area of the two rasters, and if so, clips the raster data.

    Args:
        full_data_t0_array (np.array): The NumPy array of the complete T0 raster.
        full_data_t1_array (np.array): The NumPy array of the complete T1 raster.
        dataset_t0 (gdal.Dataset): The GDAL dataset object for raster T0.
        dataset_t1 (gdal.Dataset): The GDAL dataset object for raster T1.
        aoi_coords (tuple): A tuple (xmin, ymin, xmax, ymax) defining the Area of Interest (AOI) in the raster's coordinate system, or None if not specified.
        output_text_widget (tk.scrolledtext.ScrolledText): Widget to display messages.

    Returns:
        tuple: A tuple containing:
            - cropped_data_t0 (np.array): The clipped T0 array.
            - cropped_data_t1 (np.array): The clipped T1 array.
            - current_geotransform (tuple): The new geotransform for the clipped data.
            - current_projection (str): The raster's projection.
            - success (bool): True if the operation was successful and rasters are comparable, False otherwise.
            - message (str): A descriptive message of the outcome.
    """
    original_geotransform_t0 = dataset_t0.GetGeoTransform()
    original_projection_t0 = dataset_t0.GetProjection()

    # Initialize with full data. These will be the default values
    # if no AOI is specified or if the AOI does not lead to a valid and comparable clip.
    cropped_data_t0 = full_data_t0_array
    cropped_data_t1 = full_data_t1_array
    current_geotransform = original_geotransform_t0
    current_projection = original_projection_t0
    
    # Initially assume success, if there's no AOI or if AOI is valid and produces comparable output.
    success = True
    message = "No AOI specified. Full rasters will be used if they have the same dimensions."

    if aoi_coords:
        output_text_widget.insert(tk.END, f"\nAOI coordinates detected: {aoi_coords}. Attempt to crop raster on AOI.\n")
        output_text_widget.see(tk.END)
        xmin_aoi, ymin_aoi, xmax_aoi, ymax_aoi = aoi_coords

        # Get extent info for both rasters
        info_t0 = get_raster_info(dataset_t0, output_text_widget)
        info_t1 = get_raster_info(dataset_t1, output_text_widget)

        if not info_t0 or not info_t0['extent'] or not info_t1 or not info_t1['extent']:
            # Critical error: unable to determine raster extent.
            return full_data_t0_array, full_data_t1_array, original_geotransform_t0, original_projection_t0, False, "Error: Unable to get extension information for one or both rasters."

        # Calculate the overlap extent between the two input rasters
        raster_overlap_xmin = max(info_t0['extent']['x_min'], info_t1['extent']['x_min'])
        raster_overlap_ymin = max(info_t0['extent']['y_min'], info_t1['extent']['y_min'])
        raster_overlap_xmax = min(info_t0['extent']['x_max'], info_t1['extent']['x_max'])
        raster_overlap_ymax = min(info_t0['extent']['y_max'], info_t1['extent']['y_max'])

        if raster_overlap_xmin >= raster_overlap_xmax or raster_overlap_ymin >= raster_overlap_ymax:
            # No overlap between the two rasters, so the AOI cannot intersect a valid common area.
            success = False
            message = "Notice: T0 and T1 rasters do not have a common overlap area. No AOI clipping possible for comparison."
            return full_data_t0_array, full_data_t1_array, original_geotransform_t0, original_projection_t0, success, message

        # Determine the intersection between AOI and the raster overlap area
        intersect_xmin = max(xmin_aoi, raster_overlap_xmin)
        intersect_ymin = max(ymin_aoi, raster_overlap_ymin)
        intersect_xmax = min(xmax_aoi, raster_overlap_xmax)
        intersect_ymax = min(ymax_aoi, raster_overlap_ymax)

        if intersect_xmin >= intersect_xmax or intersect_ymin >= intersect_ymax:
            # The provided AOI does not intersect with the common overlap area of the rasters.
            success = False
            message = f"Warning: The provided AOI ({xmin_aoi}, {ymin_aoi}, {xmax_aoi}, {ymax_aoi}) does not intersect with the common overlapping area of ​​the rasters. No valid clipping operation."
            return full_data_t0_array, full_data_t1_array, original_geotransform_t0, original_projection_t0, success, message
        else:
            # The AOI intersects with the common overlap area. Proceed with clipping.
            # Calculate pixel/line coordinates for the intersection relative to raster T0
            top_left_col_t0, top_left_row_t0 = world_to_pixel(original_geotransform_t0, intersect_xmin, intersect_ymax)
            bottom_right_col_t0, bottom_right_row_t0 = world_to_pixel(original_geotransform_t0, intersect_xmax, intersect_ymin)

            # Calculate pixel/line coordinates for the intersection relative to raster T1
            original_geotransform_t1 = dataset_t1.GetGeoTransform()
            top_left_col_t1, top_left_row_t1 = world_to_pixel(original_geotransform_t1, intersect_xmin, intersect_ymax)
            bottom_right_col_t1, bottom_right_row_t1 = world_to_pixel(original_geotransform_t1, intersect_xmax, intersect_ymin)

            # Ensure indices are within raster limits and are integers for T0
            start_row_t0 = max(0, int(top_left_row_t0))
            end_row_t0 = min(dataset_t0.RasterYSize, int(bottom_right_row_t0))
            start_col_t0 = max(0, int(top_left_col_t0))
            end_col_t0 = min(dataset_t0.RasterXSize, int(bottom_right_col_t0))

            # Ensure indices are within raster limits and are integers for T1
            start_row_t1 = max(0, int(top_left_row_t1))
            end_row_t1 = min(dataset_t1.RasterYSize, int(bottom_right_row_t1))
            start_col_t1 = max(0, int(top_left_col_t1))
            end_col_t1 = min(dataset_t1.RasterXSize, int(bottom_right_col_t1))

            # Check that clip dimensions are valid for both rasters
            if (start_row_t0 >= end_row_t0 or start_col_t0 >= end_col_t0 or
                start_row_t1 >= end_row_t1 or start_col_t1 >= end_col_t1):
                success = False
                message = f"Warning: The AOI coordinates ({aoi_coords}) do not correspond to valid pixels in one or both rasters after conversion. Possible AOI too small or out of bounds. No valid clipping operation."
                return full_data_t0_array, full_data_t1_array, original_geotransform_t0, original_projection_t0, success, message
            else:
                # Perform array clipping
                cropped_data_t0 = full_data_t0_array[start_row_t0:end_row_t0, start_col_t0:end_col_t0]
                cropped_data_t1 = full_data_t1_array[start_row_t1:end_row_t1, start_col_t1:end_col_t1]

                if cropped_data_t0.size == 0 or cropped_data_t1.size == 0:
                    success = False
                    message = "Warning: The area of ​​interest has produced an empty clipped raster. No change operation."
                    return full_data_t0_array, full_data_t1_array, original_geotransform_t0, original_projection_t0, success, message
                
                # IMPORTANT: After clipping, ensure the shapes are identical for comparison.
                # If original raster resolutions are different, even clipping to the same AOI
                # might produce arrays of different sizes.
                if cropped_data_t0.shape != cropped_data_t1.shape:
                    success = False
                    message = f"Warning: AOI cropping produced rasters of different sizes ({cropped_data_t0.shape} vs {cropped_data_t1.shape}). Cannot proceed with pixel-by-pixel comparison. Full rasters returned."
                    return full_data_t0_array, full_data_t1_array, original_geotransform_t0, original_projection_t0, success, message

                # Calculate the new geotransform for the clipped raster (based on T0)
                x_res, y_res = original_geotransform_t0[1], original_geotransform_t0[5]
                new_ulx = original_geotransform_t0[0] + start_col_t0 * x_res
                new_uly = original_geotransform_t0[3] + start_row_t0 * y_res
                current_geotransform = (new_ulx, x_res, original_geotransform_t0[2], new_uly, original_geotransform_t0[4], y_res)
                
                output_text_widget.insert(tk.END, f"Raster cropped to size: {cropped_data_t0.shape}\n")
                output_text_widget.insert(tk.END, f"New GeoTransform for clipped raster: {current_geotransform}\n")
                output_text_widget.see(tk.END)
                success = True
                message = "AOI successfully cropped. The cropped rasters are compatible for comparison."
                return cropped_data_t0, cropped_data_t1, current_geotransform, current_projection, success, message
                
    # If aoi_coords is None, return full data as initialized.
    # Compatibility of dimensions will be checked in the calling function.
    return cropped_data_t0, cropped_data_t1, current_geotransform, current_projection, success, message


def main_change_detection_logic(raster_t0_path, raster_t1_path, output_classified_raster_path, 
                                natural_classes, anthropic_classes, aoi_coords, output_text_widget):
    """
    Main function to execute the classified change detection workflow.
    Adapted for GUI integration.

    Args:
        raster_t0_path (str): Path to raster at time T0.
        raster_t1_path (str): Path to raster at time T1.
        output_classified_raster_path (str): Path to save the classified difference raster.
        natural_classes (set): Set of integer class codes for natural areas.
        anthropic_classes (set): Set of integer class codes for anthropic areas.
        aoi_coords (tuple, optional): Tuple (xmin, ymin, xmax, ymax) of AOI coordinates. Default to None.
        output_text_widget (tk.scrolledtext.ScrolledText): Widget to display messages.
    """
    output_text_widget.insert(tk.END, "--- Starting the Classified Change Detection Process ---\n")
    output_text_widget.see(tk.END)

    # 1. Import Raster T0 and T1
    output_text_widget.insert(tk.END, f"\nImport Raster T0: {raster_t0_path}\n")
    output_text_widget.see(tk.END)
    dataset_t0 = import_raster(raster_t0_path, output_text_widget)
    if dataset_t0 is None:
        output_text_widget.insert(tk.END, "Process aborted due to error in importing T0.\n")
        output_text_widget.see(tk.END)
        return

    output_text_widget.insert(tk.END, f"\nImport Raster T1: {raster_t1_path}\n")
    output_text_widget.see(tk.END)
    dataset_t1 = import_raster(raster_t1_path, output_text_widget)
    if dataset_t1 is None:
        output_text_widget.insert(tk.END, "Process aborted due to error in importing T1.\n")
        output_text_widget.see(tk.END)
        dataset_t0 = None    
        return

    # Read full original pixel arrays
    full_data_t0_array = dataset_t0.GetRasterBand(1).ReadAsArray()
    full_data_t1_array = dataset_t1.GetRasterBand(1).ReadAsArray()

    # Call function to check AOI and get (potentially) clipped data
    cropped_data_t0, cropped_data_t1, current_geotransform, current_projection, aoi_op_success, message = \
        check_aoi_overlap(full_data_t0_array, full_data_t1_array, dataset_t0, dataset_t1, aoi_coords, output_text_widget)

    output_text_widget.insert(tk.END, f"AOI Management Result: {message}\n")
    output_text_widget.see(tk.END)

    # If check_aoi_overlap returned None for data, it means a critical error
    # (e.g., empty clip due to invalid AOI or extent issues).
    if cropped_data_t0 is None or cropped_data_t1 is None:
        output_text_widget.insert(tk.END, "Process aborted due to a critical error in AOI or raster data handling (e.g. empty crop).\n")
        output_text_widget.see(tk.END)
        dataset_t0 = None    
        dataset_t1 = None
        return

    # At this point, cropped_data_t0 and cropped_data_t1 contain:
    # Clipped data from AOI (if AOI was valid and produced a compatible clip).
    # Original full data (if no AOI was specified, or if AOI did not produce a valid/compatible clip).

    # Main condition to proceed: rasters (or their clipped portions) must have the same dimensions.
    if cropped_data_t0.shape != cropped_data_t1.shape:
        output_text_widget.insert(tk.END, "Error: The rasters (or their clipped portions) have different dimensions and cannot be compared pixel by pixel.\n")
        output_text_widget.insert(tk.END, f"Dimensions T0: {cropped_data_t0.shape}, Dimensions T1: {cropped_data_t1.shape}\n")
        output_text_widget.insert(tk.END, "The process stops.\n")
        output_text_widget.see(tk.END)
        dataset_t0 = None    
        dataset_t1 = None
        return
    
    # If we are here, rasters are compatible for comparison.
    output_text_widget.insert(tk.END, "\nRasters are compatible for pixel-by-pixel comparison.\n")
    output_text_widget.see(tk.END)

    # Compare rasters pixel by pixel or the potentially clipped arrays with AOI
    output_text_widget.insert(tk.END, "\nStart pixel by pixel comparison for change mask...\n")
    output_text_widget.see(tk.END)
    difference_mask_array = compare_rasters_pixel_by_pixel(cropped_data_t0, cropped_data_t1, output_text_widget)
    if difference_mask_array is None:
        output_text_widget.insert(tk.END, "Process aborted due to raster array comparison error.\n")
        output_text_widget.see(tk.END)
        dataset_t0 = None    
        dataset_t1 = None
        return

    output_text_widget.insert(tk.END, "\nClassification of change detection by categories...\n")
    output_text_widget.see(tk.END)
    classified_change_result = classify_change_detection(cropped_data_t0, cropped_data_t1, natural_classes, anthropic_classes)
    if classified_change_result is None:
        output_text_widget.insert(tk.END, "Process stopped due to classification error.\n")
        output_text_widget.see(tk.END)
        dataset_t0 = None    
        dataset_t1 = None
        return

    # Export the classified difference raster
    output_text_widget.insert(tk.END, "\nExporting the classified differences raster...\n")
    output_text_widget.see(tk.END)
    export_success = export_result_raster(classified_change_result, output_classified_raster_path,    
                                          current_geotransform, current_projection, output_text_widget)
    if not export_success:
        output_text_widget.insert(tk.END, "Process stopped due to error in exporting classified change raster.\n")
        output_text_widget.see(tk.END)
        dataset_t0 = None    
        dataset_t1 = None
        return

    # Write classified change detection statistics
    output_text_widget.insert(tk.END, "\nWriting classified change statistics...\n")
    output_text_widget.see(tk.END)
    write_change_statistics(classified_change_result, output_text_widget)

    # Close GDAL datasets to release resources
    dataset_t0 = None    
    dataset_t1 = None    

    output_text_widget.insert(tk.END, "\n--- Classified Change Detection Process Completed  ---\n")
    output_text_widget.insert(tk.END, f"Check the folder '{os.path.dirname(output_classified_raster_path)}' for the results.\n")
    output_text_widget.see(tk.END)


class ChangeDetectionGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Change Detection Tool")
        self.geometry("800x800") 

        self.create_widgets()
        # Set initial values for comboboxes and text areas
        # Set default selection for comboboxes to the first available option
        if PREDEFINED_NATURAL_CLASSES:
            first_natural_key = list(PREDEFINED_NATURAL_CLASSES.keys())[0]
            self.natural_class_combobox.set(first_natural_key)
            self.update_natural_classes_text(None) # Populate text area with default
        
        if PREDEFINED_ANTHROPIC_CLASSES:
            first_anthropic_key = list(PREDEFINED_ANTHROPIC_CLASSES.keys())[0]
            self.anthropic_class_combobox.set(first_anthropic_key)
            self.update_anthropic_classes_text(None) # Populate text area with default


    def create_widgets(self):
        # --- Input Files Frame ---
        input_frame = tk.LabelFrame(self, text="Input File", padx=10, pady=10)
        input_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        tk.Label(input_frame, text="Raster T0:").grid(row=0, column=0, sticky="w", pady=2)
        self.raster_t0_path = tk.StringVar()
        tk.Entry(input_frame, textvariable=self.raster_t0_path, width=50).grid(row=0, column=1, padx=5, pady=2)
        tk.Button(input_frame, text="Browse...", command=lambda: self.browse_file(self.raster_t0_path)).grid(row=0, column=2, padx=5, pady=2)

        tk.Label(input_frame, text="Raster T1:").grid(row=1, column=0, sticky="w", pady=2)
        self.raster_t1_path = tk.StringVar()
        tk.Entry(input_frame, textvariable=self.raster_t1_path, width=50).grid(row=1, column=1, padx=5, pady=2)
        tk.Button(input_frame, text="Browse...", command=lambda: self.browse_file(self.raster_t1_path)).grid(row=1, column=2, padx=5, pady=2)

        tk.Label(input_frame, text="Output Folder:").grid(row=2, column=0, sticky="w", pady=2)
        self.output_folder_path = tk.StringVar()
        tk.Entry(input_frame, textvariable=self.output_folder_path, width=50).grid(row=2, column=1, padx=5, pady=2)
        tk.Button(input_frame, text="Browse...", command=lambda: self.browse_directory(self.output_folder_path)).grid(row=2, column=2, padx=5, pady=2)

        # --- Classification Classes Frame ---
        classes_frame = tk.LabelFrame(self, text="Classification Classes", padx=10, pady=10)
        classes_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        # Natural Classes
        tk.Label(classes_frame, text="Select Natural Class Set:").grid(row=0, column=0, sticky="w", pady=2)
        self.natural_class_set_name = tk.StringVar()
        self.natural_class_combobox = ttk.Combobox(classes_frame, textvariable=self.natural_class_set_name, 
                                                   values=list(PREDEFINED_NATURAL_CLASSES.keys()), state="readonly", width=40)
        self.natural_class_combobox.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        self.natural_class_combobox.bind("<<ComboboxSelected>>", self.update_natural_classes_text)

        tk.Label(classes_frame, text="Natural Class Values ​​(editable):").grid(row=1, column=0, sticky="nw", pady=2)
        self.natural_classes_text_entry = scrolledtext.ScrolledText(classes_frame, wrap=tk.WORD, width=60, height=3)
        self.natural_classes_text_entry.grid(row=1, column=1, columnspan=2, padx=5, pady=2, sticky="ew")

        # Anthropic Classes
        # Corrected line: removed invalid 'pt' option
        tk.Label(classes_frame, text="Select Anthropic Class Set:").grid(row=2, column=0, sticky="w", pady=(10,2)) 
        self.anthropic_class_set_name = tk.StringVar()
        self.anthropic_class_combobox = ttk.Combobox(classes_frame, textvariable=self.anthropic_class_set_name, 
                                                     values=list(PREDEFINED_ANTHROPIC_CLASSES.keys()), state="readonly", width=40)
        self.anthropic_class_combobox.grid(row=2, column=1, padx=5, pady=2, sticky="ew")
        self.anthropic_class_combobox.bind("<<ComboboxSelected>>", self.update_anthropic_classes_text)

        tk.Label(classes_frame, text="Anthropic Class Values ​​(editable):").grid(row=3, column=0, sticky="nw", pady=2)
        self.anthropic_classes_text_entry = scrolledtext.ScrolledText(classes_frame, wrap=tk.WORD, width=60, height=3)
        self.anthropic_classes_text_entry.grid(row=3, column=1, columnspan=2, padx=5, pady=2, sticky="ew")


        # --- AOI Bounding Box Frame ---
        aoi_frame = tk.LabelFrame(self, text="Area of ​​Interest (AOI) - Optional", padx=10, pady=10)
        aoi_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        self.use_aoi = tk.BooleanVar(value=False)
        tk.Checkbutton(aoi_frame, text="Enable AOI", variable=self.use_aoi, command=self.toggle_aoi_entries).grid(row=0, column=0, columnspan=4, sticky="w", pady=5)

        tk.Label(aoi_frame, text="Xmin:").grid(row=1, column=0, sticky="w", pady=2)
        self.aoi_xmin = tk.StringVar()
        self.entry_xmin = tk.Entry(aoi_frame, textvariable=self.aoi_xmin, width=15, state="disabled")
        self.entry_xmin.grid(row=1, column=1, padx=5, pady=2)

        tk.Label(aoi_frame, text="Ymin:").grid(row=1, column=2, sticky="w", pady=2)
        self.aoi_ymin = tk.StringVar()
        self.entry_ymin = tk.Entry(aoi_frame, textvariable=self.aoi_ymin, width=15, state="disabled")
        self.entry_ymin.grid(row=1, column=3, padx=5, pady=2)

        tk.Label(aoi_frame, text="Xmax:").grid(row=2, column=0, sticky="w", pady=2)
        self.aoi_xmax = tk.StringVar()
        self.entry_xmax = tk.Entry(aoi_frame, textvariable=self.aoi_xmax, width=15, state="disabled")
        self.entry_xmax.grid(row=2, column=1, padx=5, pady=2)

        tk.Label(aoi_frame, text="Ymax:").grid(row=2, column=2, sticky="w", pady=2)
        self.aoi_ymax = tk.StringVar()
        self.entry_ymax = tk.Entry(aoi_frame, textvariable=self.aoi_ymax, width=15, state="disabled")
        self.entry_ymax.grid(row=2, column=3, padx=5, pady=2)

        # --- Run Button ---
        self.run_button = tk.Button(self, text="Run Change Detection", command=self.run_change_detection, height=2, bg="lightblue", fg="black")
        self.run_button.grid(row=3, column=0, pady=10, sticky="ew")

        # --- Output/Status Text Area ---
        output_frame = tk.LabelFrame(self, text="Log and Statistics", padx=10, pady=10)
        output_frame.grid(row=4, column=0, padx=10, pady=5, sticky="nsew")
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, width=90, height=15)
        self.output_text.pack(expand=True, fill="both")

        # Configure grid row and column weights for resizing
        self.grid_rowconfigure(4, weight=1)
        self.grid_columnconfigure(0, weight=1)
        input_frame.grid_columnconfigure(1, weight=1)
        classes_frame.grid_columnconfigure(1, weight=1)
        aoi_frame.grid_columnconfigure(1, weight=1)
        aoi_frame.grid_columnconfigure(3, weight=1)

    def toggle_aoi_entries(self):
        state = "normal" if self.use_aoi.get() else "disabled"
        self.entry_xmin.config(state=state)
        self.entry_ymin.config(state=state)
        self.entry_xmax.config(state=state)
        self.entry_ymax.config(state=state)

    def browse_file(self, var):
        filepath = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")])
        if filepath:
            var.set(filepath)

    def browse_directory(self, var):
        dirpath = filedialog.askdirectory()
        if dirpath:
            var.set(dirpath)

    def update_natural_classes_text(self, event=None):
        """Updates the natural classes text entry based on combobox selection."""
        selected_set_name = self.natural_class_set_name.get()
        classes_set = PREDEFINED_NATURAL_CLASSES.get(selected_set_name, set())
        self.natural_classes_text_entry.delete("1.0", tk.END)
        self.natural_classes_text_entry.insert(tk.END, ",".join(map(str, sorted(list(classes_set)))))

    def update_anthropic_classes_text(self, event=None):
        """Updates the anthropic classes text entry based on combobox selection."""
        selected_set_name = self.anthropic_class_set_name.get()
        classes_set = PREDEFINED_ANTHROPIC_CLASSES.get(selected_set_name, set())
        self.anthropic_classes_text_entry.delete("1.0", tk.END)
        self.anthropic_classes_text_entry.insert(tk.END, ",".join(map(str, sorted(list(classes_set)))))

    def parse_classes_from_string(self, class_str):
        """
        Parses a comma-separated string of class values into a set of integers.
        Returns None if parsing fails.
        """
        try:
            # Filter out empty strings that might result from multiple commas
            # or leading/trailing commas, then convert to int and put in a set.
            return set(int(c.strip()) for c in class_str.split(',') if c.strip())
        except ValueError:
            return None

    def run_change_detection(self):
        # Clear previous output
        self.output_text.delete(1.0, tk.END)
        self.run_button.config(state="disabled", text="Processing...")

        # Get inputs from GUI
        raster_t0_path = self.raster_t0_path.get()
        raster_t1_path = self.raster_t1_path.get()
        output_folder = self.output_folder_path.get()
        
        # Parse natural and anthropic classes from text entry fields (which might have been manually edited)
        natural_classes = self.parse_classes_from_string(self.natural_classes_text_entry.get("1.0", tk.END).strip())
        anthropic_classes = self.parse_classes_from_string(self.anthropic_classes_text_entry.get("1.0", tk.END).strip())

        if natural_classes is None:
            messagebox.showerror("Input Error", "Natural classes are not valid comma-separated numbers. Please check your input.")
            self.run_button.config(state="normal", text="Run Change Detection")
            return
        if anthropic_classes is None:
            messagebox.showerror("Input Error", "Anthropic classes are not valid comma-separated numbers. Please check your input.")
            self.run_button.config(state="normal", text="Run Change Detection")
            return

        aoi_coords = None
        if self.use_aoi.get():
            try:
                xmin = float(self.aoi_xmin.get())
                ymin = float(self.aoi_ymin.get())
                xmax = float(self.aoi_xmax.get())
                ymax = float(self.aoi_ymax.get())
                if not (xmin < xmax and ymin < ymax):
                    messagebox.showerror("Error AOI", "Invalid AOI: xmin must be < xmax and ymin must be < ymax.")
                    self.run_button.config(state="normal", text="Run Change Detection")
                    return
                aoi_coords = (xmin, ymin, xmax, ymax)
            except ValueError:
                messagebox.showerror("Input Error", "AOI coordinates must be valid numbers.")
                self.run_button.config(state="normal", text="Run Change Detection")
                return
        
        if not raster_t0_path or not raster_t1_path or not output_folder:
            messagebox.showerror("Input Error", "Please select both rasters and the output folder.")
            self.run_button.config(state="normal", text="Run Change Detection")
            return

        output_classified_raster_path = os.path.join(output_folder, "change_detection_classified_result.tif")

        # Run the change detection in a separate thread
        thread = threading.Thread(target=self._run_change_detection_thread, 
                                  args=(raster_t0_path, raster_t1_path, output_classified_raster_path, 
                                        natural_classes, anthropic_classes, aoi_coords, self.output_text))
        thread.start()

    def _run_change_detection_thread(self, raster_t0_path, raster_t1_path, output_classified_raster_path, 
                                     natural_classes, anthropic_classes, aoi_coords, output_text_widget):
        """
        Wrapper function to run the main change detection logic in a separate thread.
        """
        try:
            main_change_detection_logic(raster_t0_path, raster_t1_path, output_classified_raster_path, 
                                        natural_classes, anthropic_classes, aoi_coords, output_text_widget)
        except Exception as e:
            output_text_widget.insert(tk.END, f"\nUnexpected error while running: {e}\n")
            output_text_widget.see(tk.END)
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")
        finally:
            self.run_button.config(state="normal", text="Run Change Detection")


if __name__ == "__main__":
    app = ChangeDetectionGUI()
    app.mainloop()
