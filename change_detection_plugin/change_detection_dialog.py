import os
import sys
import numpy as np
from osgeo import gdal, osr

from qgis.PyQt import uic
from qgis.PyQt.QtWidgets import QDialog, QMessageBox, QProgressBar, QLineEdit
from qgis.PyQt.QtCore import QThread, pyqtSignal, QCoreApplication
from qgis.core import QgsRasterLayer, QgsMessageLog, Qgis, QgsMapLayer, QgsMapLayerProxyModel, QgsRectangle, QgsCoordinateReferenceSystem, QgsProject 
from qgis.gui import QgsMapLayerComboBox, QgsFileWidget 
from .change_detection_dialog_base import Ui_Dialog as FORM_CLASS

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

def import_raster(filepath, log_func):
    """
    Import a georeferenced raster.
    """
    if not os.path.exists(filepath):
        log_func(f"Error: The file {filepath} does not exist.", Qgis.Critical)
        return None
    
    dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
    if dataset is None:
        log_func(f"Error: Unable to open raster file {filepath}. Check format or corruption.", Qgis.Critical)
    return dataset

def get_raster_info(dataset, log_func):
    """
    Reads data and metadata from a raster.
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
        resolution = (x_res, abs(y_res))
    else:
        extent = None
        resolution = None
        log_func("Warning: Unable to get georeferencing (transformation) information for raster.", Qgis.Warning)

    info = {
        'rows': rows,
        'cols': cols,
        'num_pixels': rows * cols,
        'extent': extent,
        'resolution': resolution,
        'projection': dataset.GetProjection()
    }
    return info

def compare_rasters_pixel_by_pixel(data_t0, data_t1, log_func):
    """
    Compares two raster arrays pixel by pixel and returns an array of differences.
    """
    if data_t0.shape != data_t1.shape:
        log_func("Error: Raster arrays have different dimensions and cannot be compared pixel by pixel.", Qgis.Critical)
        log_func(f"Dimensions T0: {data_t0.shape}, Dimensions T1: {data_t1.shape}", Qgis.Critical)
        return None
    
    difference_array = np.zeros_like(data_t0, dtype=np.uint8)
    difference_array[data_t0 != data_t1] = 1

    return difference_array

def classify_change_detection(data_t0, data_t1, natural_classes, anthropic_classes):
    """
    Classify detected changes according to natural and anthropic area categories.
    """
    classified_change_array = np.zeros_like(data_t0, dtype=np.uint8)
    data_t0_int = data_t0.astype(int)
    data_t1_int = data_t1.astype(int)

    mask_anthropic_invariate = np.isin(data_t0_int, list(anthropic_classes)) & \
                               np.isin(data_t1_int, list(anthropic_classes))
    classified_change_array[mask_anthropic_invariate] = 1

    mask_natural_invariate = np.isin(data_t0_int, list(natural_classes)) & \
                             np.isin(data_t1_int, list(natural_classes))
    classified_change_array[mask_natural_invariate] = 2

    mask_natural_to_anthropic = np.isin(data_t0_int, list(natural_classes)) & \
                                np.isin(data_t1_int, list(anthropic_classes))
    classified_change_array[mask_natural_to_anthropic] = 3

    mask_anthropic_to_natural = np.isin(data_t0_int, list(anthropic_classes)) & \
                                np.isin(data_t1_int, list(natural_classes))
    classified_change_array[mask_anthropic_to_natural] = 4

    return classified_change_array

def export_result_raster(array_to_export, output_filepath, geotransform, projection, log_func):
    """
    Export the resulting raster (of differences or classified).
    """
    rows, cols = array_to_export.shape
    driver = gdal.GetDriverByName('GTiff')
    gdal_data_type = gdal.GDT_Byte    
    
    out_dataset = driver.Create(output_filepath, cols, rows, 1, gdal_data_type)
    
    if out_dataset is None:
        log_func(f"Error: Unable to create output file {output_filepath}.", Qgis.Critical)
        return False

    out_dataset.SetGeoTransform(geotransform)
    out_dataset.SetProjection(projection)

    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(array_to_export)
    
    out_band.SetNoDataValue(0)    

    out_band.FlushCache()
    out_dataset = None    

    log_func(f"Raster successfully exported to: {output_filepath}", Qgis.Info)
    return True

def write_change_statistics(classified_change_array, log_func):
    """
    Writes classified change detection raster statistics to log.
    """
    if classified_change_array is None:
        log_func("Error: The classified array is None; unable to calculate statistics.", Qgis.Critical)
        return False

    total_pixels = classified_change_array.size
    
    anthropic_invariate_pixels = np.sum(classified_change_array == 1)
    natural_invariate_pixels = np.sum(classified_change_array == 2)
    natural_to_anthropic_pixels = np.sum(classified_change_array == 3)
    anthropic_to_natural_pixels = np.sum(classified_change_array == 4)
    unclassified_pixels = np.sum(classified_change_array == 0)

    try:
        log_func("\n--- Classified Change Detection Raster Statistics ---", Qgis.Info)
        log_func(f"Total number of pixels: {total_pixels}", Qgis.Info)
        log_func(f"1) Unchanged Anthropic Areas: {anthropic_invariate_pixels} pixel ({((anthropic_invariate_pixels / total_pixels) * 100):.2f}%)", Qgis.Info)
        log_func(f"2) Unchanged Natural Areas: {natural_invariate_pixels} pixel ({((natural_invariate_pixels / total_pixels) * 100):.2f}%)", Qgis.Info)
        log_func(f"3) From Natural to Anthropic Areas: {natural_to_anthropic_pixels} pixel ({((natural_to_anthropic_pixels / total_pixels) * 100):.2f}%)", Qgis.Info)
        log_func(f"4) From Anthropic to Natural Areas: {anthropic_to_natural_pixels} pixel ({((anthropic_to_natural_pixels / total_pixels) * 100):.2f}%)", Qgis.Info)
        log_func(f"5) Unclassified pixels (other types of changes or areas not classified): {unclassified_pixels} pixel ({((unclassified_pixels / total_pixels) * 100):.2f}%)", Qgis.Info)
            
        return True
    except Exception as e:
        log_func(f"Error writing statistics: {e}", Qgis.Critical)
        return False

def world_to_pixel(geotransform, x, y):
    """
    Converts world coordinates (geographic or projected) to pixel/line coordinates.
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

def check_aoi_overlap(full_data_t0_array, full_data_t1_array, dataset_t0, dataset_t1, aoi_coords, log_func):
    """
    Checks whether the AOI is within the overlapping area of ​​the two rasters and, if so, clips the raster data.
    """
    original_geotransform_t0 = dataset_t0.GetGeoTransform()
    original_projection_t0 = dataset_t0.GetProjection()

    cropped_data_t0 = full_data_t0_array
    cropped_data_t1 = full_data_t1_array
    current_geotransform = original_geotransform_t0
    current_projection = original_projection_t0
    
    success = True
    message = "No AOI specified. Full rasters will be used if they have the same dimensions."

    if aoi_coords:
        log_func(f"\nOI coordinates detected: {aoi_coords}. Attempt to crop raster on AOI.", Qgis.Info)
        xmin_aoi, ymin_aoi, xmax_aoi, ymax_aoi = aoi_coords

        info_t0 = get_raster_info(dataset_t0, log_func)
        info_t1 = get_raster_info(dataset_t1, log_func)

        if not info_t0 or not info_t0['extent'] or not info_t1 or not info_t1['extent']:
            return full_data_t0_array, full_data_t1_array, original_geotransform_t0, original_projection_t0, False, "Error: Unable to get extension information for one or both rasters."

        raster_overlap_xmin = max(info_t0['extent']['x_min'], info_t1['extent']['x_min'])
        raster_overlap_ymin = max(info_t0['extent']['y_min'], info_t1['extent']['y_min'])
        raster_overlap_xmax = min(info_t0['extent']['x_max'], info_t1['extent']['x_max'])
        raster_overlap_ymax = min(info_t0['extent']['y_max'], info_t1['extent']['y_max'])

        if raster_overlap_xmin >= raster_overlap_xmax or raster_overlap_ymin >= raster_overlap_ymax:
            success = False
            message = "Warning: T0 and T1 rasters do not have a common overlap area. No AOI clipping possible for comparison."
            return full_data_t0_array, full_data_t1_array, original_geotransform_t0, original_projection_t0, success, message

        intersect_xmin = max(xmin_aoi, raster_overlap_xmin)
        intersect_ymin = max(ymin_aoi, raster_overlap_ymin)
        intersect_xmax = min(xmax_aoi, raster_overlap_xmax)
        intersect_ymax = min(ymax_aoi, raster_overlap_ymax)

        if intersect_xmin >= intersect_xmax or intersect_ymin >= intersect_ymax:
            success = False
            message = f"Warning: The provided AOI ({xmin_aoi}, {ymin_aoi}, {xmax_aoi}, {ymax_aoi}) does not intersect with the common overlapping area of ​​the rasters. No valid clipping operation."
            return full_data_t0_array, full_data_t1_array, original_geotransform_t0, original_projection_t0, success, message
        else:
            top_left_col_t0, top_left_row_t0 = world_to_pixel(original_geotransform_t0, intersect_xmin, intersect_ymax)
            bottom_right_col_t0, bottom_right_row_t0 = world_to_pixel(original_geotransform_t0, intersect_xmax, intersect_ymin)

            original_geotransform_t1 = dataset_t1.GetGeoTransform()
            top_left_col_t1, top_left_row_t1 = world_to_pixel(original_geotransform_t1, intersect_xmin, intersect_ymax)
            bottom_right_col_t1, bottom_right_row_t1 = world_to_pixel(original_geotransform_t1, intersect_xmax, intersect_ymin)

            start_row_t0 = max(0, int(top_left_row_t0))
            end_row_t0 = min(dataset_t0.RasterYSize, int(bottom_right_row_t0))
            start_col_t0 = max(0, int(top_left_col_t0))
            end_col_t0 = min(dataset_t0.RasterXSize, int(bottom_right_col_t0))

            start_row_t1 = max(0, int(top_left_row_t1))
            end_row_t1 = min(dataset_t1.RasterYSize, int(bottom_right_row_t1))
            start_col_t1 = max(0, int(top_left_col_t1))
            end_col_t1 = min(dataset_t1.RasterXSize, int(bottom_right_col_t1))

            if (start_row_t0 >= end_row_t0 or start_col_t0 >= end_col_t0 or
                start_row_t1 >= end_row_t1 or start_col_t1 >= end_col_t1):
                success = False
                message = f"Warning: The AOI coordinates ({aoi_coords}) do not correspond to valid pixels in one or both rasters after conversion. Possible AOI too small or out of bounds. No valid clipping operation."
                return full_data_t0_array, full_data_t1_array, original_geotransform_t0, original_projection_t0, success, message
            else:
                cropped_data_t0 = full_data_t0_array[start_row_t0:end_row_t0, start_col_t0:end_col_t0]
                cropped_data_t1 = full_data_t1_array[start_row_t1:end_row_t1, start_col_t1:end_col_t1]

                if cropped_data_t0.size == 0 or cropped_data_t1.size == 0:
                    success = False
                    message = "Warning: The area of ​​interest has produced an empty clipped raster. No change operation."
                    return full_data_t0_array, full_data_t1_array, original_geotransform_t0, original_projection_t0, success, message
                
                if cropped_data_t0.shape != cropped_data_t1.shape:
                    success = False
                    message = f"Warning: AOI cropping produced rasters of different sizes ({cropped_data_t0.shape} vs {cropped_data_t1.shape}). Cannot proceed with pixel-by-pixel comparison. Full rasters returned."
                    return full_data_t0_array, full_data_t1_array, original_geotransform_t0, original_projection_t0, success, message

                x_res, y_res = original_geotransform_t0[1], original_geotransform_t0[5]
                new_ulx = original_geotransform_t0[0] + start_col_t0 * x_res
                new_uly = original_geotransform_t0[3] + start_row_t0 * y_res
                current_geotransform = (new_ulx, x_res, original_geotransform_t0[2], new_uly, original_geotransform_t0[4], y_res)
                
                log_func(f"Raster cropped to size: {cropped_data_t0.shape}", Qgis.Info)
                log_func(f"New GeoTransform for clipped raster: {current_geotransform}", Qgis.Info)
                success = True
                message = "AOI successfully cropped. The cropped rasters are compatible for comparison."
                return cropped_data_t0, cropped_data_t1, current_geotransform, current_projection, success, message
                
    return cropped_data_t0, cropped_data_t1, current_geotransform, current_projection, success, message


class ChangeDetectionWorker(QThread):
    """Worker thread to perform change detection analysis without blocking the GUI."""
    finished = pyqtSignal(str)
    progress = pyqtSignal(int)
    log_message = pyqtSignal(str, Qgis.MessageLevel)

    def __init__(self, raster_t0_path, raster_t1_path, output_classified_raster_path,
                 natural_classes, anthropic_classes, aoi_coords):
        super().__init__()
        self.raster_t0_path = raster_t0_path
        self.raster_t1_path = raster_t1_path
        self.output_classified_raster_path = output_classified_raster_path
        self.natural_classes = natural_classes
        self.anthropic_classes = anthropic_classes
        self.aoi_coords = aoi_coords

    def run(self):
        try:
            self.log_message.emit("--- Starting the Classified Change Detection Process ---", Qgis.Info)
            self.progress.emit(5)

            # 1. Importa Raster T0 e T1
            self.log_message.emit(f"\nImport Raster T0: {self.raster_t0_path}", Qgis.Info)
            dataset_t0 = import_raster(self.raster_t0_path, self.log_message.emit)
            if dataset_t0 is None:
                self.finished.emit("Process aborted due to error in importing T0.")
                return
            self.progress.emit(15)

            self.log_message.emit(f"\nImport Raster T1: {self.raster_t1_path}", Qgis.Info)
            dataset_t1 = import_raster(self.raster_t1_path, self.log_message.emit)
            if dataset_t1 is None:
                self.finished.emit("Process aborted due to error in importing T1.")
                dataset_t0 = None
                dataset_t1 = None
                return
            self.progress.emit(25)

            full_data_t0_array = dataset_t0.GetRasterBand(1).ReadAsArray()
            full_data_t1_array = dataset_t1.GetRasterBand(1).ReadAsArray()

            self.progress.emit(35)
            cropped_data_t0, cropped_data_t1, current_geotransform, current_projection, aoi_op_success, message = \
                check_aoi_overlap(full_data_t0_array, full_data_t1_array, dataset_t0, dataset_t1, self.aoi_coords, self.log_message.emit)

            self.log_message.emit(f"AOI Management Result: {message}", Qgis.Info)

            if cropped_data_t0 is None or cropped_data_t1 is None:
                self.finished.emit("Process aborted due to a critical error in AOI or raster data handling (e.g. empty crop).")
                dataset_t0 = None
                dataset_t1 = None
                return
            self.progress.emit(45)

            if cropped_data_t0.shape != cropped_data_t1.shape:
                self.log_message.emit(f"Error: The rasters (or their clipped portions) have different dimensions and cannot be compared pixel by pixel.", Qgis.Critical)
                self.log_message.emit(f"Dimensions T0: {cropped_data_t0.shape}, Dimensions T1: {cropped_data_t1.shape}", Qgis.Critical)
                self.finished.emit("The process stops.")
                dataset_t0 = None
                dataset_t1 = None
                return
            self.log_message.emit("\nRasters are compatible for pixel-by-pixel comparison.", Qgis.Info)
            self.progress.emit(55)

            self.log_message.emit("\nStart pixel by pixel comparison for change mask...", Qgis.Info)
            difference_mask_array = compare_rasters_pixel_by_pixel(cropped_data_t0, cropped_data_t1, self.log_message.emit)
            if difference_mask_array is None:
                self.finished.emit("Process aborted due to raster array comparison error.")
                dataset_t0 = None
                dataset_t1 = None
                return
            self.progress.emit(65)

            self.log_message.emit("\nClassification of change detection by categories...", Qgis.Info)
            classified_change_result = classify_change_detection(cropped_data_t0, cropped_data_t1, self.natural_classes, self.anthropic_classes)
            if classified_change_result is None:
                self.finished.emit("Process stopped due to classification error.")
                dataset_t0 = None
                dataset_t1 = None
                return
            self.progress.emit(75)

            self.log_message.emit("\nExporting the classified differences raster...", Qgis.Info)
            export_success = export_result_raster(classified_change_result, self.output_classified_raster_path,    
                                                  current_geotransform, current_projection, self.log_message.emit)
            if not export_success:
                self.finished.emit("Process stopped due to error in exporting classified change raster.")
                dataset_t0 = None
                dataset_t1 = None
                return
            self.progress.emit(85)

            self.log_message.emit("\nWriting classified change statistics...", Qgis.Info)
            write_change_statistics(classified_change_result, self.log_message.emit)
            self.progress.emit(95)

            dataset_t0 = None    
            dataset_t1 = None    

            self.log_message.emit("\n--- Classified Change Detection Process Completed ---", Qgis.Info)
            self.log_message.emit(f"Check the folder '{os.path.dirname(self.output_classified_raster_path)}' for the results.", Qgis.Info)
            self.finished.emit("Successfully completed!")
            self.progress.emit(100)

        except Exception as e:
            self.log_message.emit(f"\nUnexpected error while running: {e}", Qgis.Critical)
            self.finished.emit(f"Error: {e}")
            self.progress.emit(0)


class ChangeDetectionDialog(QDialog, FORM_CLASS):
    def __init__(self, iface, parent=None):
        """Dialog Box Builder."""
        super(ChangeDetectionDialog, self).__init__(parent)
        self.setupUi(self)
        self.iface = iface
        self.worker_thread = None

        self.setup_ui_elements()
        self.connect_signals()
        self.set_initial_values()
        
    def setup_ui_elements(self):
        """Configure QGIS specific UI elements and other details."""
        # Configure combo boxes for raster layers
        self.mMapLayerComboBox_t0.setFilters(QgsMapLayerProxyModel.RasterLayer) 
        self.mMapLayerComboBox_t1.setFilters(QgsMapLayerProxyModel.RasterLayer)

        # Configure the output folder selection widget
        self.mFileWidget_outputFolder.setStorageMode(QgsFileWidget.StorageMode.GetDirectory) 

        # Configure the widget for the AOI extension
        self.mExtentWidget_aoi.setMapCanvas(self.iface.mapCanvas())

        # Populates comboboxes for default classes
        self.comboBox_natural_type.addItems(list(PREDEFINED_NATURAL_CLASSES.keys()))
        self.comboBox_anthropic_type.addItems(list(PREDEFINED_ANTHROPIC_CLASSES.keys()))

        # Sets the initial state of the AOI inputs
        self.toggle_aoi_entries()

    def connect_signals(self):
        """Connects UI element signals to slo."""
        self.pushButton_run.clicked.connect(self.run_change_detection)
        self.checkBox_aoi.toggled.connect(self.toggle_aoi_entries)
        self.comboBox_natural_type.currentIndexChanged.connect(self.update_natural_classes_text)
        self.comboBox_anthropic_type.currentIndexChanged.connect(self.update_anthropic_classes_text)
        
    def set_initial_values(self):
        """Set initial values ​​for widgets."""
        # Set a default selection for class comboboxes
        if PREDEFINED_NATURAL_CLASSES:
            self.comboBox_natural_type.setCurrentIndex(0) 
            self.update_natural_classes_text()
        if PREDEFINED_ANTHROPIC_CLASSES:
            self.comboBox_anthropic_type.setCurrentIndex(0) 
            self.update_anthropic_classes_text()

    def toggle_aoi_entries(self):
        """Enable/disable AOI input fields based on checkbox state."""
        enabled = self.checkBox_aoi.isChecked()
        self.mExtentWidget_aoi.setEnabled(enabled)

    def update_natural_classes_text(self, event=None):
        """Updates the natural classes text field based on the combobox selection."""
        selected_set_name = self.comboBox_natural_type.currentText()
        classes_set = PREDEFINED_NATURAL_CLASSES.get(selected_set_name, set())
        self.plainTextEdit_natural_manual.clear()
        self.plainTextEdit_natural_manual.insertPlainText(",".join(map(str, sorted(list(classes_set)))))

    def update_anthropic_classes_text(self, event=None):
        """Updates the human classes text field based on the combobox selection."""
        selected_set_name = self.comboBox_anthropic_type.currentText()
        classes_set = PREDEFINED_ANTHROPIC_CLASSES.get(selected_set_name, set())
        self.plainTextEdit_anthropic_manual.clear()
        self.plainTextEdit_anthropic_manual.insertPlainText(",".join(map(str, sorted(list(classes_set)))))

    def parse_classes_from_string(self, class_str):
        """
        Parsing of a a comma-separated string of class values ​​into a set of integers.
        Raises a ValueError if parsing fails.
        """
        if not class_str:
            return set()
        try:
            return set(int(c.strip()) for c in class_str.split(',') if c.strip())
        except ValueError:
            raise ValueError(f"Error: The class values '{class_str}' are not valid comma separated numbers.")

    def log_to_text_browser(self, message, level=Qgis.Info):
        """Emits a message to the log textBrowser and the QGIS log."""
        self.textBrowser_log.append(message)
        QgsMessageLog.logMessage(message, 'Change Detection Plugin', level=level)

    def update_progress_bar(self, value):
        """Refresh the progress bar."""
        self.progressBar.setValue(value)

    def on_worker_finished(self, message):
        """Handles the worker thread end signal."""
        self.log_to_text_browser(f"\n--- Process Completed ---", Qgis.Info)
        self.log_to_text_browser(message, Qgis.Info)
        self.pushButton_run.setEnabled(True)
        self.pushButton_run.setText("Run Change Detection")
        if "Error" in message:
            QMessageBox.critical(self, "Change Detection Error", message)
        else:
            QMessageBox.information(self, "Change Detection Completed", message)
            # Carica il raster risultante in QGIS
            output_path = os.path.join(self.mFileWidget_outputFolder.filePath(), "change_detection_classified_result.tif")
            if os.path.exists(output_path):
                layer_name = os.path.basename(output_path)
                rlayer = QgsRasterLayer(output_path, layer_name)
                if rlayer.isValid():
                    self.iface.addRasterLayer(output_path, layer_name)
                    self.log_to_text_browser(f"Layer '{layer_name}' added to map.", Qgis.Info)
                else:
                    self.log_to_text_browser(f"Error loading raster layer: {output_path}", Qgis.Critical)
            else:
                self.log_to_text_browser(f"The output file does not exist: {output_path}", Qgis.Warning)


    def run_change_detection(self):
        """Start the change detection process in a separate thread."""
        self.textBrowser_log.clear()
        self.progressBar.setValue(0)
        self.pushButton_run.setEnabled(False)
        self.pushButton_run.setText("Processing...")

        raster_t0_layer = self.mMapLayerComboBox_t0.currentLayer()
        raster_t1_layer = self.mMapLayerComboBox_t1.currentLayer()
        output_folder = self.mFileWidget_outputFolder.filePath()

        if not raster_t0_layer or not raster_t1_layer:
            self.log_to_text_browser("Error: Select both raster layers T0 and T1.", Qgis.Critical)
            QMessageBox.critical(self, "Input Error", "Please, select both raster layers T0 and T1.")
            self.on_worker_finished("Process stopped due to missing inputs.")
            return

        if not output_folder:
            self.log_to_text_browser("Error: Select an output folder.", Qgis.Critical)
            QMessageBox.critical(self, "Input Error", "Please, select an output foldert.")
            self.on_worker_finished("Process stopped due to missing inputs.")
            return

        # --- start of CRS checks ---
        raster_t0_crs = raster_t0_layer.crs().authid()
        raster_t1_crs = raster_t1_layer.crs().authid()
        project_crs = QgsProject.instance().crs().authid()

        self.log_to_text_browser(f"CRS Raster T0: {raster_t0_crs}", Qgis.Info)
        self.log_to_text_browser(f"CRS Raster T1: {raster_t1_crs}", Qgis.Info)
        self.log_to_text_browser(f"CRS Project QGIS: {project_crs}", Qgis.Info)

        # 1. Check: if the rasters have different reference systems between them
        if raster_t0_crs != raster_t1_crs:
            message = "Warning: The two rasters have different reference systems, change detection will not be performed."
            self.log_to_text_browser(message, Qgis.Critical)
            QMessageBox.critical(self, "Error CRS", message)
            self.on_worker_finished("Process stopped due to different CRS between rasters.")
            return

        # 2. Check: if the two rasters have the same reference system but different from the project one
        if raster_t0_crs != project_crs:
            message = "Warning: The rasters have a different reference system than the project one, the results may not be consistent."
            self.log_to_text_browser(message, Qgis.Critical)
            QMessageBox.critical(self, "Warning CRS", message)
            self.on_worker_finished("Process stopped due to different CRS between rasters and project.")
            return            

        # --- stop of CRS checks ---
        
        raster_t0_path = raster_t0_layer.source()
        raster_t1_path = raster_t1_layer.source()
        output_classified_raster_path = os.path.join(output_folder, "change_detection_classified_result.tif")

        try:
            natural_classes = self.parse_classes_from_string(self.plainTextEdit_natural_manual.toPlainText().strip())
            anthropic_classes = self.parse_classes_from_string(self.plainTextEdit_anthropic_manual.toPlainText().strip())
        except ValueError as e:
            self.log_to_text_browser(str(e), Qgis.Critical)
            QMessageBox.critical(self, "Input Class Error", str(e))
            self.on_worker_finished("Process aborted due to invalid classes.")
            return

        aoi_coords = None
        if self.checkBox_aoi.isChecked():
            # Instead of using currentExtent(), we read directly from the text fields to overcome a communication problem (fallback)
            try:
                # The default names of the QLineEdit inside the QgsExtentWidget are assumed
                xmin_str = self.mExtentWidget_aoi.findChild(QLineEdit, "mXMinLineEdit").text()
                ymin_str = self.mExtentWidget_aoi.findChild(QLineEdit, "mYMinLineEdit").text()
                xmax_str = self.mExtentWidget_aoi.findChild(QLineEdit, "mXMaxLineEdit").text()
                ymax_str = self.mExtentWidget_aoi.findChild(QLineEdit, "mYMaxLineEdit").text()

                # Convert to float
                xmin = float(xmin_str.replace(',', '.'))
                ymin = float(ymin_str.replace(',', '.'))
                xmax = float(xmax_str.replace(',', '.'))
                ymax = float(ymax_str.replace(',', '.'))

                # Build a QgsRectangle
                extent = QgsRectangle(xmin, ymin, xmax, ymax)

                if extent.isNull():
                    self.log_to_text_browser("Error: AOI enabled but invalid extension (text fields are empty or invalid).", Qgis.Critical)
                    QMessageBox.critical(self, "Error AOI", "AOI enabled but extension invalid. Make sure fields are populated.")
                    self.on_worker_finished("Process aborted due to invalid AOI.")
                    return

                aoi_coords = (extent.xMinimum(), extent.yMinimum(), extent.xMaximum(), extent.yMaximum())
                
                # Additional check for AOI validity (xmin < xmax and ymin < ymax)
                if not (aoi_coords[0] < aoi_coords[2] and aoi_coords[1] < aoi_coords[3]):
                    self.log_to_text_browser("Error: Invalid AOI. Xmin must be < Xmax and Ymin must be < Ymax.", Qgis.Critical)
                    QMessageBox.critical(self, "Error AOI", "Invalid AOI: Xmin must be < Xmax and Ymin must be < Ymax.")
                    self.on_worker_finished("Process aborted due to invalid AOI.")
                    return

            except (AttributeError, ValueError) as e:
                self.log_to_text_browser(f"Error reading AOI coordinates from text fields: {e}. Make sure the fields are populated correctly.", Qgis.Critical)
                QMessageBox.critical(self, "Error AOI", f"Unable to read AOI coordinates: {e}. Fields may be blank or contain non-numeric text.")
                self.on_worker_finished("Process aborted due to invalid AOI.")
                return



        self.worker_thread = ChangeDetectionWorker(
            raster_t0_path, raster_t1_path, output_classified_raster_path,
            natural_classes, anthropic_classes, aoi_coords
        )
        self.worker_thread.finished.connect(self.on_worker_finished)
        self.worker_thread.progress.connect(self.update_progress_bar)
        self.worker_thread.log_message.connect(self.log_to_text_browser)
        self.worker_thread.start()
