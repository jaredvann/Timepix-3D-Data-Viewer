"""
ARIADNE Experiment, Department of Physics, University of Liverpool

event-viewer/constants.py

Authors: Jared Vann
"""

TOA_CLOCK_TO_NS = 1.5625
TOT_ADU_TO_NS = 25

XY_CONVERSION_FACTOR = 0.703125    # pixels -> mm

Z_CONVERSION_FACTOR = 0.001674 * TOA_CLOCK_TO_NS    # clocks -> mm

PIXELS_TO_HEIGHT_NS_RATIO = Z_CONVERSION_FACTOR / XY_CONVERSION_FACTOR
