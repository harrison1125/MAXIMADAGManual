"""
Input paths for XRD and XRF analysis workflows

This module centralizes the filepaths used in automated XRD analysis workflows for further use. This discretizes input values from further analysis scripts, which increases cleanliness down the line as we separate manual vs automated workflows (manual workflows use this input folder, automated workflows naturally stream new results without requiring explicitly defining filepaths). 

Attributes
----------
root_dir : str
    Path to directory containing all files to analyze.
poni_file : str
    Path to calibration file for XRD. Generated through PyFAI.
config_path : str
    Path to configuration file for XRF. Generated through PYMCA.
"""
root_dir = '/Users/hpark108/Desktop/Piyush Rohit Solutionized CuTi Final/Day 2 Batch 2'
poni_file = '/Users/hpark108/Desktop/Piyush Rohit Solutionized CuTi Final/Day 2 Batch 2/JHACRD00011_69d3cc54b1ba2b821d691427_0_1340_2026-05-21_21-11-13/JHACRD00011_69d3cc54b1ba2b821d691427_0_1340_2026-05-21_21-11-13.poni'
config_path = '/Users/hpark108/Desktop/Piyush Rohit Solutionized CuTi Final/Piyush Rohit Solutionized CuTi Final.cfg'
