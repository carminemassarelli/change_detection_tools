# Change Detection Tools
Tools to perform change detection on encoded rasters in GIS environment
  
## Description
This repository hosts tools dedicated to the analysis of change detection on raster data, specifically designed to operate with classified maps related to the theme of Soil in Italy, exploiting the available time series, such as those provided by ISPRA (Higher Institute for Environmental Protection and Research).
  
## Purpose  
The main objective is to identify and classify the transformations that have occurred over time on the Italian territory, analysing the differences between two classified raster maps (e.g. relating to different years). The methodology is based on a pixel-by-pixel comparison between the two inputs, allowing for quantification and mapping the dynamics of change, with particular attention to the transitions between natural and anthropogenic areas.
  
## Data Source and References  
The reference data for the application of these tools are the classified maps on the theme of Soil in Italy, available on the ISPRA portal. It is recommended to consult the "Soil and Territory" section to access the time series and complete metadata: ISPRA - Soil and Territory: https://www.isprambiente.gov.it/it/attivita/suolo-e-territorio/suolo  
You can use any raster map classified with integer numeric values to represent the different classes or categories.
  
## How does that work  
The heart of the script and tools is the function that compares the values of each pixel between the raster at time T0 and that at time T1. The changes are then classified into significant categories (e.g. unchanged natural areas, unchanged anthropogenic areas, transformation from natural to anthropogenic, transformation from anthropogenic to natural), providing a clear view of territorial dynamics.
  
## Available Tools  
This repository offers a suite of tools designed to facilitate the analysis of change detection, aimed at both users who prefer the command line interface and those who prefer a visual and integrated approach in the GIS environment.

The tools included are the following:

- Python script from the command line ([change_detection.py](https://github.com/carminemassarelli/change_detection_tools/blob/main/change_detection.py)), which represents the heart of the project. A versatile Python script that allows you to run your entire change detection workflow directly from the terminal. It offers complete control over all parameters, including class selection, Area of Interest (AOI) definition, and input/output path management. Ideal for automation, integration into data processing pipelines, or for power users.

- Desktop application with Graphical User Interface ([change_detection_GUI.py](https://github.com/carminemassarelli/change_detection_tools) coming soon) for users who prefer a more intuitive and visual approach. A Python script is available that integrates a Tkinter-based User Interface. This desktop application allows you to interact with change detection capabilities through a graphical window, making it easy to select files, enter parameters, and view progress, without the need to type complex commands.

- Plugin for QGIS 3.x ([QGIS 3.x Plugin](https://github.com/carminemassarelli/change_detection_tools) coming soon). To enable full integration into the geospatial workflow, a dedicated plugin for QGIS has been developed. This plugin brings change detection capabilities directly into the QGIS environment, allowing users to perform analysis on layers loaded into the map, taking advantage of all GIS features for data pre-processing and post-processing. It provides a familiar user experience and direct access to GIS tools to visualise and analyse results.
  
## How to Cite These Tools  
If you use or modify these tools for your research, publications or if you integrate them into your software/project, I kindly ask you to cite them. Citation helps to give visibility to the work done and allows others to trace the source.  
  
**For scientific publications, theses, reports**  
If the results obtained through these tools or one of them are part of a scientific publication, dissertation, technical report or other formal document, the following formatting is suggested (adaptable to the citation style of your discipline):  
- Massarelli, Carmine (2025). Change Detection Tools. GitHub repository https://github.com/carminemassarelli/change_detection_tools
  
**For Integration into Your Software or Project**  
If you integrate portions of these tools or use them as a module within your software, application, or other code-based project, please include clear attribution in your project documentation (README, license, "Credits" section, or in-code comments), for example:

Credits  
This project uses, or was inspired by, "Change Detection Tools" available on GitHub. Link to the original repository: https://github.com/carminemassarelli/change_detection_tools  
  
## Code Development Support  
The development of this script was supported and facilitated by the use of artificial intelligence models. As an environmental technologist with skills that are not primarily in computer science, I used Google Gemini to receive assistance in optimising parts of the code and solving "deep problems" related to UI development.
