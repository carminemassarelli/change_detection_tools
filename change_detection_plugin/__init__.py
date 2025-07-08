# __init__.py
def classFactory(iface):
    """Load ChangeDetectionPlugin from its class."""
    from .change_detection_plugin import ChangeDetectionPlugin
    return ChangeDetectionPlugin(iface)
