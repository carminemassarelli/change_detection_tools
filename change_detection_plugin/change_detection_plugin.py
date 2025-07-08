import os
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMessageBox

# Importa il file delle risorse compilato
from . import resources

class ChangeDetectionPlugin:
    """QGIS Plugin to perform change detection and classifies the results.
       It supports analysis on a specific Area of ​​Interest (AOI).
       It writes classified change detection raster statistics to console."""

    def __init__(self, iface):
        """Plugin Builder.

        :param iface: A reference to the QGIS interface.
        :type iface: QgsInterface
        """
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)

        # Initialize the dialog to None
        self.dialog = None

        # Create the action that will launch the plugin dialog
        self.action = QAction(
            QIcon(":/plugins/change_detection_plugin/change_detection.png"),
            self.tr("Change Detection Tool"), self.iface.mainWindow())
        self.action.triggered.connect(self.run)

        # Add the action to the QGIS toolbar and menu
        self.menu = self.tr("&Change Detection")
        self.toolbar = self.iface.addToolBar(self.tr("Change Detection"))
        self.toolbar.setObjectName("ChangeDetectionToolbar")
        self.toolbar.addAction(self.action)

    def initGui(self):
        """Create the plugin menu and toolbar."""
        self.iface.addPluginToMenu(self.menu, self.action)

    def unload(self):
        """Removes the plugin menu and toolbar.
           The toolbar is automatically removed from QGIS when the plugin is deactivated.
        """
        self.iface.removePluginMenu(self.tr("&Change Detection"), self.action)

    def run(self):
        """Launch the plugin dialog."""
        if self.dialog is None:
            # Import dialog class here to avoid circular imports
            from .change_detection_dialog import ChangeDetectionDialog
            self.dialog = ChangeDetectionDialog(self.iface)
        self.dialog.show()

    def tr(self, message):
        """Gets the string translated with QGIS translation system.

        :param message: The string to translate.
        :type message: str
        :returns: The traslated string.
        :rtype: str
        """
        return QCoreApplication.translate("ChangeDetectionPlugin", message)
