# SpectralHull
Implementation of quick hull algorithm for spectal data

Import cHullRemover.py module using:
```
from cHullRemover import cHullRemover
```
See https://github.com/mminin/SpectralHull/blob/master/removeHullTutorial.ipynb for more information.

# CHQ_numpy.py

This script will perform Convex Hull Quotent on an entire coverage.
This was written for use on moderately sized areas of interest obtained from projected L2 M3 coverages.
The script is written entirely in numpy, its already quite fast, but there's lots of room for improvement and further optimizations.

Note that the input spectrum is smoothed by 3-element-wide median filter in order to avoid creating artifacts when a single bad pixel value is encountered (these would produce false positives for pyroxenes).
Quotent is calculated by dividing unsmoothed source data by this hull (so you can still see all the detail)
The script also outputs a copy of the source image with first two bands removed (so you could load both into QGIS at the same time and use spectral-temporal plugin to plot both at once).

