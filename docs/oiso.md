# Wavefront measurements for OISO data

Import all needed packages:

```py
import astrofocus as focus
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import scipy
```

Input and output files for the sequence of observations (adapt to your case):

```py
bkg_files = glob.glob('full/bkg/2026110*[0-9].fits.bz2') # input background files
bkg_stat_file = 'background_stat.fits.bz2' # output background statistics
img_files = glob.glob('full/2026110*[0-9].fits.bz2') # input image files
img_stat_file = 'debiased_image_stat.fits.bz2' # output image statistics
```

Integrate background statisticse (or read it from existing file):

```py
if os.path.isfile(bkg_stat_file):
    bkg_stat = focus.SampleStatistics.readfrom(bkg_stat_file)
else:
    bkg_stat = focus.SampleStatistics()
    for file in bkg_files:
        bkg_stat.update(focus.readfits(file))
    bkg_stat.writeto(bkg_stat_file)
bkg_avg = bkg_stat.mean()
bkg_std = bkg_stat.std(corrected=True)
bkg_rms = np.median(bkg_std)
print(f"Median of background level: {round(np.median(bkg_avg), 3)}")
print(f"Median of background standard deviation: {round(bkg_rms, 3)}")
```

Integrate debiased full image (or read it from existing file):

```py
if os.path.isfile(img_stat_file):
    img_stat = focus.SampleStatistics.readfrom(img_stat_file)
else:
    img_stat = focus.SampleStatistics()
    for file in img_files:
        img_stat.update(focus.readfits(file) - bkg_avg)
    img_stat.writeto(img_stat_file)
img_avg = img_stat.mean()
```

Display mean image `img_avg` (with `SAOImage/DS9` or `plt.imshow`) and choose
a region of interest (ROI) for the wavefront sensor (WFS):

```py
roi = focus.ImageROI(xcen=2192, ycen=526, width=301, height=299)
```

Define a mask to hide bad pixels in the ROI:

```py
wfs = roi.crop(img_avg) # extract WFS part in mean debiased image
ker = focus.ball_2d_window(3)
ker[3,3] = False # exclude central point
med = scipy.ndimage.median_filter(wfs, footprint=ker, mode='constant', cval=0.0)
msk = (wfs - med) < 35.0 # mask of good pixels
```

Process WFS images:

```py
wfs_bkg_avg = roi.crop(bkg_avg)
for file in img_files:
    # Read WFS image (remove bias and multiply by mask of good pixels).
    wfs = (roi.crop(focus.readfits(file)) - wfs_bkg_avg)*msk
    data = focus.detect_grid_of_spots(wfs, centroiding=True, fig=None, verbosity=1,
                                      spot_fwhm=5, grid_shape=(8,8),
                                      detect_level=5)
    prefix = re.match('^(.*?)\\.fits(\\.[^/]*)?$', file).group(1)
    data.writeto(prefix + '-wfs.fits.bz2', overwrite=True)
```
