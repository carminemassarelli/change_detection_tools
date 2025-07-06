#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from osgeo import gdal, osr
import numpy as np
import sys
import argparse

"""
This script performs change detection and classifies the results.
It supports analysis on a specific Area of ​​Interest (AOI).
It writes classified change detection raster statistics to console.

Usage example::
To view help:
  python3 change_detection.py --help

Examples of use:
1. Without AOI, with predefined classes (e.g. Land Use):
   python3 change_detection.py <data_folder>/ <raster_t0.tif> <raster_t1.tif> --natural-type land_use --anthropic-type land_use

2. With AOI, with predefined classes (e.g. Land Cover):
   python3 change_detection.py <data_folder>/ <raster_t0.tif> <raster_t1.tif> --aoi 401000.0 4501000.0 403000.0 4503000.0 --natural-type land_cover --anthropic-type land_cover

3. Without AOI, with manually indicated natural and anthropic classes:
   python3 change_detection.py <data_folder>/ <raster_t0.tif> <raster_t1.tif> --natural-manual "11,12,13,14" --anthropic-manual "3,4"

4. With AOI and mixed classes (predefined natural, manual anthropic):
   python3 change_detection.py <data_folder>/ <raster_t0.tif> <raster_t1.tif> --aoi 401000.0 4501000.0 403000.0 4503000.0 --natural-type land_consumption --anthropic-manual "1,11,12"

Note: If you use the AOI mode, the coordinates must be in the raster reference system.
Manual class values ​​must be separated by commas and enclosed in quotes (e.g. "1,2,3").
"""

# --- Default Classification Classes ---
# These dictionaries contain the different sets of natural and anthropic classes.
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


def import_raster(filepath):
    """
    Import a georeferenced raster.

    Args:
        filepath (str): The path to the raster file.

    Returns:
        gdal.Dataset: The GDAL dataset object of the raster.
                      Returns None if the file cannot be opened.
    """
    if not os.path.exists(filepath):
        print(f"Errore: Il file {filepath} non esiste.")
        return None
    
    dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
    if dataset is None:
        print(f"Error: Unable to open raster file {filepath}. Check format or corruption.")
    return dataset

def get_raster_info(dataset):
    """
    Reads data and metadata from a raster.

    Args:
        dataset (gdal.Dataset): The GDAL dataset object of the raster.

    Returns:
        dict: A dictionary containing the raster information (number of pixels, extent, resolution).
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
        resolution = (x_res, abs(y_res)) # Using the absolute value for the Y resolution
    else:
        extent = None
        resolution = None
        print("Warning: Unable to get georeferencing (transformation) information for raster.")

    info = {
        'rows': rows,
        'cols': cols,
        'num_pixels': rows * cols,
        'extent': extent,
        'resolution': resolution,
        'projection': dataset.GetProjection()
    }
    return info

def compare_rasters_pixel_by_pixel(data_t0, data_t1):
    """
    Compares two raster arrays pixel by pixel and returns an array of differences.
    Different pixels will have value 1, otherwise 0.

    Args:
        data_t0 (numpy.ndarray): Array of raster pixel values T0.
        data_t1 (numpy.ndarray): Array of raster pixel values T1.

    Returns:
        numpy.ndarray: A 2D array containing the difference values ​​(1 for change, 0 for no change).
                       Returns None if the rasters are not comparable.
    """
    if data_t0.shape != data_t1.shape:
        print("Error: Raster arrays have different dimensions and cannot be compared pixel by pixel.")
        print(f"Dimensions T0: {data_t0.shape}, Dimensions T1: {data_t1.shape}")
        return None
    
    difference_array = np.zeros_like(data_t0, dtype=np.uint8)
    difference_array[data_t0 != data_t1] = 1

    return difference_array

def classify_change_detection(data_t0, data_t1, natural_classes, anthropic_classes):
    """
    Classify detected changes according to natural and anthropic area categories.

    Args:
        data_t0 (numpy.ndarray): Array of raster pixel values T0.
        data_t1 (numpy.ndarray): Array of raster pixel values T1.
        natural_classes (set): Set of integer class codes considered natural.
        anthropic_classes (set): Set of integer class codes considered anthropic.

    Returns:
        numpy.ndarray: A classified array where:
                        1: Unchanged Anthropic Areas
                        2: Unchanged Natural Areas
                        3: From Natural to Anthropic Areas
                        4: From Anthropic to Natural Areas
                        0: Other types of change (or NoData if applicable, not falling into the categories)
    """
    
    # Initialize the output array with zeros
    classified_change_array = np.zeros_like(data_t0, dtype=np.uint8)

    # Make sure arrays are of integer type for class comparisons
    data_t0_int = data_t0.astype(int)
    data_t1_int = data_t1.astype(int)

    # The membership to the set is considered as the discriminant and not the absolute
    # value of the pixel, this is because in some cases pixels referring to the same 
    # place report different values ​​for different years but always attributable to the
    # anthropic or natural class
    
    # 1. Unchanged anthropic areas (value 1)
    mask_anthropic_invariate = np.isin(data_t0_int, list(anthropic_classes)) & \
                               np.isin(data_t1_int, list(anthropic_classes))
    classified_change_array[mask_anthropic_invariate] = 1

    # 2. Unchanged natural areas (value 2)
    mask_natural_invariate = np.isin(data_t0_int, list(natural_classes)) & \
                             np.isin(data_t1_int, list(natural_classes))
    classified_change_array[mask_natural_invariate] = 2

    # 3. From natural to anthropic areas (value 3)
    mask_natural_to_anthropic = np.isin(data_t0_int, list(natural_classes)) & \
                                np.isin(data_t1_int, list(anthropic_classes))
    classified_change_array[mask_natural_to_anthropic] = 3

    # 4. Da aree antropiche a naturali (valore 4)
    mask_anthropic_to_natural = np.isin(data_t0_int, list(anthropic_classes)) & \
                                np.isin(data_t1_int, list(natural_classes))
    classified_change_array[mask_anthropic_to_natural] = 4

    return classified_change_array

def export_result_raster(array_to_export, output_filepath, geotransform, projection):
    """
    Export the resulting raster (of differences or classified).

    Args:
        array_to_export (numpy.ndarray): The numpy array containing the values ​​to export.
        output_filepath (str): The path and name of the output raster file.
        geotransform (tuple): The 6-tuple of the geotransformation for the output raster.
        projection (str): The projection WKT string for the output raster.

    Returns:
        bool: True if the export was successful, False otherwise.
    """
    rows, cols = array_to_export.shape
    driver = gdal.GetDriverByName('GTiff')
    
    # Determine GDAL data type based on NumPy array data type
    # For values ​​0-4, GDT_Byte (8-bit unsigned integer) is sufficient.
    gdal_data_type = gdal.GDT_Byte    
    
    out_dataset = driver.Create(output_filepath, cols, rows, 1, gdal_data_type)
    
    if out_dataset is None:
        print(f"Error: Unable to create output file {output_filepath}.")
        return False

    out_dataset.SetGeoTransform(geotransform)
    out_dataset.SetProjection(projection)

    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(array_to_export)
    
    out_band.SetNoDataValue(0)    

    out_band.FlushCache()
    out_dataset = None    

    print(f"Raster successfully exported to: {output_filepath}")
    return True

def write_change_statistics(classified_change_array):
    """
    Writes classified change detection raster statistics to console.

    Args:
        classified_change_array (numpy.ndarray): The numpy array containing the classified change values.

    Returns:
        bool: True if writing was successful, False otherwise.
    """
    if classified_change_array is None:
        print("Error: The classified array is None; unable to calculate statistics.")
        return False

    total_pixels = classified_change_array.size
    
    # We count pixels for each class
    anthropic_invariate_pixels = np.sum(classified_change_array == 1)
    natural_invariate_pixels = np.sum(classified_change_array == 2)
    natural_to_anthropic_pixels = np.sum(classified_change_array == 3)
    anthropic_to_natural_pixels = np.sum(classified_change_array == 4)
    
    unclassified_pixels = np.sum(classified_change_array == 0)

    try:
        print("\n--- Classified Change Detection Raster Statistics ---")
        print(f"Total number of pixels: {total_pixels}")
        print(f"1) Unchanged Anthropic Areas: {anthropic_invariate_pixels} pixel ({((anthropic_invariate_pixels / total_pixels) * 100):.2f}%)")
        print(f"2) Unchanged Natural Areas: {natural_invariate_pixels} pixel ({((natural_invariate_pixels / total_pixels) * 100):.2f}%)")
        print(f"3) From Natural to Anthropic Areas: {natural_to_anthropic_pixels} pixel ({((natural_to_anthropic_pixels / total_pixels) * 100):.2f}%)")
        print(f"4) From Anthropic to Natural Areas: {anthropic_to_natural_pixels} pixel ({((anthropic_to_natural_pixels / total_pixels) * 100):.2f}%)")
        print(f"5) Unclassified pixels (other types of changes or areas not classified): {unclassified_pixels} pixel ({((unclassified_pixels / total_pixels) * 100):.2f}%)")
            
        return True
    except Exception as e: # Catch a more generic exception for printing
        print(f"Error writing statistics: {e}")
        return False

def world_to_pixel(geotransform, x, y):
    """
    Converts world coordinates (geographic or projected) to pixel/line coordinates.
    
    Args:
        geotransform (tuple): The 6-tuple of the geotransform (gdal.GetGeoTransform()).
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

    return (px, py)

def check_aoi_overlap(full_data_t0_array, full_data_t1_array, dataset_t0, dataset_t1, aoi_coords):
    """
    Checks whether the AOI is within the overlapping area of ​​the two rasters and, if so, clips the raster data.

    Args:
        full_data_t0_array (np.array): The NumPy array of the complete T0 raster.
        full_data_t1_array (np.array): The NumPy array of the complete T1 raster
        dataset_t0 (gdal.Dataset): The GDAL dataset object for the T0 raster.
        dataset_t1 (gdal.Dataset): The GDAL dataset object for the T1 raster.
        aoi_coords (tuple): A tuple (xmin, ymin, xmax, ymax) that defines the Area of ​​Interest (AOI) in the raster coordinate system, or None if not specified.

    Returns:
        tuple: A tuple containing:
            - cropped_data_t0 (np.array): The trimmed T0 array.
            - cropped_data_t1 (np.array): The trimmed T1 array.
            - current_geotransform (tuple): The new geotransform for clipped data.
            - current_projection (str): The raster projection.
            - success (bool): True if the operation was successful and the rasters are comparable, False otherwise.
            - message (str): A descriptive message of the outcome.
    """
    original_geotransform_t0 = dataset_t0.GetGeoTransform()
    original_projection_t0 = dataset_t0.GetProjection()

    # Initialize with full data. These will be the default values ​​if no AOI is specified
    # or if the AOI does not yield a valid and comparable crop.
    cropped_data_t0 = full_data_t0_array
    cropped_data_t1 = full_data_t1_array
    current_geotransform = original_geotransform_t0
    current_projection = original_projection_t0
    
    # Success is initially assumed if there is no AOI or if the AOI is valid and produces comparable output.
    success = True
    message = "No AOI specified. Full rasters will be used if they have the same dimensions."

    if aoi_coords:
        print(f"\nAOI coordinates detected: {aoi_coords}. Attempt to crop raster on AOI.")
        xmin_aoi, ymin_aoi, xmax_aoi, ymax_aoi = aoi_coords

        # Get extension info for both rasters
        info_t0 = get_raster_info(dataset_t0)
        info_t1 = get_raster_info(dataset_t1)

        if not info_t0 or not info_t0['extent'] or not info_t1 or not info_t1['extent']:
            # Critical error: Unable to determine raster extent.
            return full_data_t0_array, full_data_t1_array, original_geotransform_t0, original_projection_t0, False, "Error: Unable to get extension information for one or both rasters."

        # Calculates the extent of overlap between the two input rasters
        raster_overlap_xmin = max(info_t0['extent']['x_min'], info_t1['extent']['x_min'])
        raster_overlap_ymin = max(info_t0['extent']['y_min'], info_t1['extent']['y_min'])
        raster_overlap_xmax = min(info_t0['extent']['x_max'], info_t1['extent']['x_max'])
        raster_overlap_ymax = min(info_t0['extent']['y_max'], info_t1['extent']['y_max'])

        if raster_overlap_xmin >= raster_overlap_xmax or raster_overlap_ymin >= raster_overlap_ymax:
            # There is no overlap between the two rasters, so the AOI cannot intersect a valid common area.
            success = False
            message = "Notice: T0 and T1 rasters do not have a common overlap area. No AOI clipping possible for comparison."
            return full_data_t0_array, full_data_t1_array, original_geotransform_t0, original_projection_t0, success, message

        # Determines the intersection between AOI and raster overlap area
        intersect_xmin = max(xmin_aoi, raster_overlap_xmin)
        intersect_ymin = max(ymin_aoi, raster_overlap_ymin)
        intersect_xmax = min(xmax_aoi, raster_overlap_xmax)
        intersect_ymax = min(ymax_aoi, raster_overlap_ymax)

        if intersect_xmin >= intersect_xmax or intersect_ymin >= intersect_ymax:
            # The provided AOI does not intersect with the common overlap area of ​​the rasters.
            success = False
            message = f"Warning: The provided AOI ({xmin_aoi}, {ymin_aoi}, {xmax_aoi}, {ymax_aoi}) does not intersect with the common overlapping area of ​​the rasters. No valid clipping operation."
            return full_data_t0_array, full_data_t1_array, original_geotransform_t0, original_projection_t0, success, message
        else:
            # The AOI intersects with the common overlap area. Let's proceed with the clipping.
            # Calculate pixel/line coordinates for intersection with raster T0
            top_left_col_t0, top_left_row_t0 = world_to_pixel(original_geotransform_t0, intersect_xmin, intersect_ymax)
            bottom_right_col_t0, bottom_right_row_t0 = world_to_pixel(original_geotransform_t0, intersect_xmax, intersect_ymin)

            # Calculate pixel/line coordinates for intersection with raster T1
            original_geotransform_t1 = dataset_t1.GetGeoTransform()
            top_left_col_t1, top_left_row_t1 = world_to_pixel(original_geotransform_t1, intersect_xmin, intersect_ymax)
            bottom_right_col_t1, bottom_right_row_t1 = world_to_pixel(original_geotransform_t1, intersect_xmax, intersect_ymin)

            # Make sure the indices are within the raster bounds and are integers for T0
            start_row_t0 = max(0, int(top_left_row_t0))
            end_row_t0 = min(dataset_t0.RasterYSize, int(bottom_right_row_t0))
            start_col_t0 = max(0, int(top_left_col_t0))
            end_col_t0 = min(dataset_t0.RasterXSize, int(bottom_right_col_t0))

            # Make sure the indices are within the raster bounds and are integers for T1
            start_row_t1 = max(0, int(top_left_row_t1))
            end_row_t1 = min(dataset_t1.RasterYSize, int(bottom_right_row_t1))
            start_col_t1 = max(0, int(top_left_col_t1))
            end_col_t1 = min(dataset_t1.RasterXSize, int(bottom_right_col_t1))

            # Check that the crop dimensions are valid for both rasters
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
                
                # IMPORTANT: After cropping, make sure the shapes are identical for comparison.
                # If the resolutions of the original rasters are different, even a crop on the same AOI may produce arrays of different sizes.
                if cropped_data_t0.shape != cropped_data_t1.shape:
                    success = False
                    message = f"Warning: AOI cropping produced rasters of different sizes ({cropped_data_t0.shape} vs {cropped_data_t1.shape}). Cannot proceed with pixel-by-pixel comparison. Full rasters returned."
                    return full_data_t0_array, full_data_t1_array, original_geotransform_t0, original_projection_t0, success, message

                # Calculate the new geotransformation for the clipped raster (based on T0)
                x_res, y_res = original_geotransform_t0[1], original_geotransform_t0[5]
                new_ulx = original_geotransform_t0[0] + start_col_t0 * x_res
                new_uly = original_geotransform_t0[3] + start_row_t0 * y_res
                current_geotransform = (new_ulx, x_res, original_geotransform_t0[2], new_uly, original_geotransform_t0[4], y_res)
                
                print(f"Raster cropped to size: {cropped_data_t0.shape}")
                print(f"New GeoTransform for clipped raster: {current_geotransform}")
                success = True
                message = "AOI successfully cropped. The cropped rasters are compatible for comparison."
                return cropped_data_t0, cropped_data_t1, current_geotransform, current_projection, success, message
                
    # If aoi_coords is None, return the complete data as initialized
    # Size compatibility will be checked in the calling function.
    return cropped_data_t0, cropped_data_t1, current_geotransform, current_projection, success, message


def main_change_detection(raster_t0_path, raster_t1_path, output_classified_raster_path, 
                          natural_classes, anthropic_classes, aoi_coords=None):
    """
    Main function to perform the classified change detection workflow.

    Args:
        raster_t0_path (str): Raster path at time T0.
        raster_t1_path (str): Raster path at time T1.
        output_classified_raster_path (str): Path to save the classified differences raster.
        natural_classes (set): Set of integer class codes for natural areas.
        anthropic_classes (set): Set of integer class codes for anthropic areas.
        aoi_coords (tuple, optional): Tuple (xmin, ymin, xmax, ymax) of the AOI coordinates. Default to None.
    """
    print("--- Starting the Classified Change Detection Process ---")

    # Import Raster T0 and T1
    print(f"\nImport Raster T0: {raster_t0_path}")
    dataset_t0 = import_raster(raster_t0_path)
    if dataset_t0 is None:
        print("Process aborted due to error in importing T0.")
        return

    print(f"\nImport Raster T1: {raster_t1_path}")
    dataset_t1 = import_raster(raster_t1_path)
    if dataset_t1 is None:
        print("Process aborted due to error in importing T1.")
        dataset_t0 = None    
        return

    # Read the complete original pixel arrays
    full_data_t0_array = dataset_t0.GetRasterBand(1).ReadAsArray()
    full_data_t1_array = dataset_t1.GetRasterBand(1).ReadAsArray()

    # Call function to check AOI and get (potentially) clipped data
    cropped_data_t0, cropped_data_t1, current_geotransform, current_projection, aoi_op_success, message = check_aoi_overlap(full_data_t0_array, full_data_t1_array,
    dataset_t0, dataset_t1, aoi_coords)

    print(f"AOI Management Result: {message}")

    # If the check_aoi_overlap function returned None for the data, it means 
    # a critical error (e.g. empty clipping due to invalid AOI or extension problems).
    if cropped_data_t0 is None or cropped_data_t1 is None:
        print("Process aborted due to a critical error in AOI or raster data handling (e.g. empty crop).")
        dataset_t0 = None    
        dataset_t1 = None
        return

    # At this point, cropped_data_t0 and cropped_data_t1 contain:
    # The data cropped by the AOI (if the AOI was valid and produced a compatible crop).
    # The original complete data (if no AOI was specified, or if the AOI did not produce a valid/compatible crop).

    # Main condition to proceed: the rasters (or their cut portions) must have the same dimensions.
    if cropped_data_t0.shape != cropped_data_t1.shape:
        print("Error: The rasters (or their cropped portions) have different dimensions and cannot be compared pixel by pixel.")
        print(f"Dimensions T0: {cropped_data_t0.shape}, Dimensions T1: {cropped_data_t1.shape}")
        print("The process stops.")
        dataset_t0 = None    
        dataset_t1 = None
        return
    
    # If we are here, the rasters are compatible for comparison.
    print("\nRasters are compatible for pixel-by-pixel comparison.")

    # Compare rasters pixel by pixel or arrays possibly clipped with AOI
    print("\nStart pixel by pixel comparison for change mask...")
    difference_mask_array = compare_rasters_pixel_by_pixel(cropped_data_t0, cropped_data_t1)
    if difference_mask_array is None:
        print("Process aborted due to raster array comparison error.")
        dataset_t0 = None    
        dataset_t1 = None
        return

    print("\nClassification of change detection by categories...")
    # Pass the natural and anthropogenic classes to the classification function
    classified_change_result = classify_change_detection(cropped_data_t0, cropped_data_t1, natural_classes, anthropic_classes)
    if classified_change_result is None:
        print("Process stopped due to classification error.")
        dataset_t0 = None    
        dataset_t1 = None
        return

    # Export the classified differences raster
    print("\nExporting the classified differences raster...")
    # We pass the updated geotransformation and projection (of the clipped raster if present)
    export_success = export_result_raster(classified_change_result, output_classified_raster_path, current_geotransform, current_projection)
    if not export_success:
        print("Process stopped due to error in exporting classified change raster.")
        dataset_t0 = None    
        dataset_t1 = None
        return

    # Write the statistics of the classified change detection raster
    print("\nWriting classified change statistics...")
    write_change_statistics(classified_change_result)

    # Close GDAL datasets to release resources
    dataset_t0 = None    
    dataset_t1 = None    

    print("\n--- Classified Change Detection Process Completed ---")
    print(f"Check the folder '{os.path.dirname(output_classified_raster_path)}' for the results.")

def parse_classes_from_string(class_str):
    """
    Parse a comma-separated string of class values ​​into a set of integers.
    Raises a ValueError if parsing fails.
    """
    if not class_str:
        return set()
    try:
        return set(int(c.strip()) for c in class_str.split(',') if c.strip())
    except ValueError:
        raise ValueError(f"Error: The values ​​of the classes '{class_str}' are not valid comma separated numbers.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Performs change detection and classifies the results.\n"
                    "Allows you to specify natural and anthropogenic classes via predefined sets or manual input.",
        formatter_class=argparse.RawTextHelpFormatter # For better help formatting
    )

    # Mandatory arguments
    parser.add_argument("data_dir", help="Path to the folder containing the raster data and where to save the results.")
    parser.add_argument("raster_t0_filename", help="Raster file name at time T0.")
    parser.add_argument("raster_t1_filename", help="Raster file name at time T1.")

    # Optional AOI topics
    parser.add_argument("--aoi", nargs=4, metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'), type=float,
                        help="Area of ​​Interest (AOI) coordinates in the raster reference system.\n"
                             "Formato: XMIN YMIN XMAX YMAX (es. 401000.0 4501000.0 403000.0 4503000.0)")

    # Class Selection Topics (Mutually Exclusive Groups for Clarity)
    class_group_natural = parser.add_mutually_exclusive_group()
    class_group_natural.add_argument("--natural-type", choices=list(PREDEFINED_NATURAL_CLASSES.keys()),
                                     help="Select a predefined set of natural classes: " + ", ".join(PREDEFINED_NATURAL_CLASSES.keys()))
    class_group_natural.add_argument("--natural-manual", type=str,
                                     help="Manually provide natural classes as a comma-separated string of numbers (e.g. '1,2,3,10').")

    class_group_anthropic = parser.add_mutually_exclusive_group()
    class_group_anthropic.add_argument("--anthropic-type", choices=list(PREDEFINED_ANTHROPIC_CLASSES.keys()),
                                      help="Select a predefined set of anthropic classes: " + ", ".join(PREDEFINED_ANTHROPIC_CLASSES.keys()))
    class_group_anthropic.add_argument("--anthropic-manual", type=str,
                                      help="Please manually provide the anthropological classes as a comma-separated string of numbers (e.g. '4,5,6,20').")

    args = parser.parse_args()

    # Determine the natural classes
    if args.natural_manual:
        try:
            natural_classes = parse_classes_from_string(args.natural_manual)
        except ValueError as e:
            print(e)
            sys.exit(1)
    elif args.natural_type:
        natural_classes = PREDEFINED_NATURAL_CLASSES[args.natural_type]
    else:
        print("Error: You must specify natural classes using --natural-type or --natural-manual.")
        parser.print_help()
        sys.exit(1)

    # Determine the anthropic classes
    if args.anthropic_manual:
        try:
            anthropic_classes = parse_classes_from_string(args.anthropic_manual)
        except ValueError as e:
            print(e)
            sys.exit(1)
    elif args.anthropic_type:
        anthropic_classes = PREDEFINED_ANTHROPIC_CLASSES[args.anthropic_type]
    else:
        print("Error: You must specify anthropic classes using --anthropic-type or --anthropic-manual.")
        parser.print_help()
        sys.exit(1)

    # Prepare file paths and AOI
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True) # Assicura che la cartella di output esista

    raster_t0_filepath = os.path.join(data_dir, args.raster_t0_filename)
    raster_t1_filepath = os.path.join(data_dir, args.raster_t1_filename)

    output_classified_raster_filepath = os.path.join(data_dir, "change_detection_classified_result.tif")

    aoi_coords = tuple(args.aoi) if args.aoi else None
    if aoi_coords:
        if not (aoi_coords[0] < aoi_coords[2] and aoi_coords[1] < aoi_coords[3]):
            print("Errore: AOI non valida. XMIN deve essere < XMAX e YMIN deve essere < YMAX.")
            sys.exit(1)

    # Call the main change detection function with the given classes
    main_change_detection(raster_t0_filepath, raster_t1_filepath,
                          output_classified_raster_filepath, 
                          natural_classes=natural_classes, 
                          anthropic_classes=anthropic_classes,
                          aoi_coords=aoi_coords)

    print("\nScript execution completed.")