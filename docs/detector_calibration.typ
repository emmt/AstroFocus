#import "@preview/mosaic:0.0.1" as mosaic
#import mosaic.themes.metropolis as m
#show: m.setup.with(
  title: [Basics of Image Processing],
  subtitle: [with Python examples],
  authors: [Éric Thiébaut, Jean-Philippe Berger, Ferréol Soulez],
  date: [novembre 2026, OHP],
)

#m.slide(layout: "title")

= Prerequisites

== Required Software

- Python with packages:
  - `NumPy` for fast computations of numerical arrays (images);
  - `Matplotlib` for plots;
  - `Astropy` for reading and writing FITS files;
  - `SciPy` for optimization (https://scipy.org/);

- Custom package `AstroFocus` (https://github.com/emmt/AstroFocus), 2 possibilities:
  - `git clone https://github.com/emmt/AstroFocus`
  - download https://github.com/emmt/AstroFocus/archive/refs/heads/main.zip

== Installation of AstroFocus

After cloning Git repository with:

```bash
git clone https://github.com/emmt/AstroFocus
```

or downloading and unpacking archive at

```
https://github.com/emmt/AstroFocus/archive/refs/heads/main.zip
```

Start Python and execute the following snippet after replacing `some_dir` by the path to the cloned Git repository or unpacked archive:

```python
import sys
sys.path.append("some_dir/astrofocus/src")
import astrofocus as focus
```

== Code

Typically your Python code will looks like:

```python
import numpy as np # for arrays
import matplotlib.pyplot as plt # for plotting
from astrofocus import * # for useful functions
from pathlib import Path
datadir = '/home/eric/data/FOCUS/detector_calibration/SBIG-STF-8300M/'
bias = [readfits(f) for f in Path(datadir + "/bias").iterdir() if f.is_file()]
len(bias)
sum1 = zeros(bias[0].shape);
for b in bias: sum1 += b;
avg = sum1/len(bias);
figure(2); plt.clf(); plt.imshow(avg,vmin=1000,vmax=1050);
plt.title("Mean bias ({} frame)");
sum2 = zeros(bias[0].shape);
for img in bias: sum2 += (img - avg)**2;
var = sum2/(len(bias) - 1);
figure(2); plt.clf(); plt.imshow(avg,vmin=1000,vmax=1050);

```

= Task 1: Detector calibration

== Captured electrons

Number of electrons captured by a pixel well:
$
  p = cal(P)((q thin r + c) thin Delta t)
$
- $cal(P)(mu)$ denotes the realization of a Poisson process of mean $mu$,
- $q$ is the quantum efficiency (QE),
- $r$ is the irradiance (in photons per unit of time),
- $c$ is the dark current (in electrons par unit of time),
- $Delta t$ is the exposure time.

There are two different physical processes:
- photo-electric effect with $q thin r thin Delta t$ the mean number of resulting photo-electrons,
- dark current with $c thin Delta t$ the mean number of resulting electrons.

== Output signal

Output signal in _Analog Digital Units_ (ADU):
$
  y approx round(p/g + b) = round((cal(P)((q thin r + c) thin Delta t))/g + b)
$
- $round(...)$ denotes conversion to ADU by the _Analog to Digital Converter_ (ADC),
- $g$ is the *gain* of the ADC (in electrons per ADU),
- $b$ is the *bias* of the ADC (in ADU).

Remarks:
- $≈$ accounts for readout noise (shot noise already taken into account by $p$),
- _saturation_ not taken into account,
- _blooming_ and _leakage_ not taken into account.

== Affine approximation

According to the model, the output signal $y$ is approximately an affine function of the irradiance $r$:
$
  y ≈ ((q  thin r + c) thin Delta t)/g + b
  = (x/a + d) thin Delta t + b
$
- $≈$ reminds that all noises (shot noise, read-out noise and quantization noise) have been neglected,
- $x$ is the _signal of interest_, which can be anything proportional to the irradiance $r$, here we assume that $x = (frac(a thin q, g, style: "horizontal")) thin r$,
- $d = frac(c, g, style: "horizontal")$ is the contribution of the dark current in ADU per unit of time.

Remark: unmixing the effects of $q$ and $g$ require an absolute calibration source (usually not available on sky), this is why we introduced $x$ (and thus $a$).

== Affine correction

Considering the previous affine approximation, the extracted signal is given by an affine correction of the form:
$
  hat(x) = a thin ((y - b)/(Delta t) - d)
$
Since the exposure time $Delta t$ is known, detector calibration amounts to estimating $a$, $b$, and $d$ from a set set of calibration data frames acquired under different conditions.

Remarks:
- $a$, $b$, and $d$ depend on the pixel  (i.e, $a_(j)$, $b_(j)$, and $d_(j)$ for the $j$-th pixel) but, when not necessary, we do not explicitly indicate the pixel index,
- the exposure time $Delta t$ is the same for all pixels of an image.

== Signal of interest statistics

The estimator:
$
  hat(x) = a thin ((y - b)/(Delta t) - d)
$
is not sufficient, the level of uncertainty is also needed.

Neglecting saturation, blooming, etc., the first statistical moments of $hat(x)$ are:
$
  EE(hat(x)) &= a thin ((EE(y) - b)/(Delta t) - d) = (a q)/g thin r\
  "Var"(hat(x)) &= (a/(Delta t))^2 thin "Var"(y)
$
i.e. the expectation $EE(y)$ and variance $"Var"(y)$ of the output signal $y$ are needed.

== Output signal statistics

For a Poisson process, $EE(cal(P)(mu)) = mu$ and $"Var"(cal(P)(mu)) = mu$.

Hence, expectation and variance of the output signal are:
$
  EE(y)
    &= EE(p)/g + b
    = ((q thin r + c) thin Delta t)/g + b\
    &= (x/a + d) thin Delta t + b\
  "Var"(y)
    &= "Var"(p)/g^2 + sigma^2_("ron")
    = ((q thin r + c) thin Delta t)/g^2 + sigma^2_("ron")\
    &= underbrace((x/a + d) thin (Delta t)/g,"shot noise") + underbrace(sigma^2_("ron"),"read-out noise")
$

== Summary

The signal of interest and its variance are given by:
$
  hat(x) &= a thin ((y - b)/(Delta t) - d)\
  "Var"(hat(x))
  &≈ (a/(Delta t))^2 thin ((y - b)_(+)/g + sigma_("ron")^2)\
$
with 5 detector parameters to calibrate:
- $g$ and $b$ the *gain* and *bias* of the DAC (in e$""^-$/ADU and ADU),
- $d$ the *dark current* (in ADU per unit of time),
- $a$ a *conversion factor* (in units of $x$ per ADU),
- $sigma_("ron")$ the standard deviation of the read-out + quantization noises (in ADU).

== Bias images

Bias images ($Delta t = 0$, usually $r=0$ but can be anything):
$
  hat("Avg")(y^("bias")) &≈ EE(y^("bias")) = b\
  hat("Var")(y^("bias")) &≈ "Var"(y^("bias")) = sigma^2_("ron")\
$
with $hat("Avg")(y^("bias"))$ and $hat("Var")(y^("bias"))$ the sample mean and variance of the $n^("bias")$ bias images (taken pixel-wise). This gives the following estimators:
$
  hat(b)
    &= hat("Avg")(y^("bias"))
    = 1/n^("bias") thin sum_(k=1)^(n^("bias")) y^("bias")_(k)\
  hat(sigma)_("ron")^2
    &= hat("Var")(y^("bias"))
    = 1/n^("bias") thin sum_(k=1)^(n^("bias")) (y^("bias")_(k) - hat("Avg")(y^("bias")))^2\
$

== Dark images

Dark images ($Delta t > 0$ and $r = 0$):
$
  hat("Avg")(y^("dark")) &≈ EE(y^("dark")) = d thin Delta t + b\
  hat("Var")(y^("dark")) &≈ "Var"(y^("dark")) = (d thin Delta t)/g + sigma^2_("ron")\
$
If dark images have been acquired with a single exposure time, then possible estimators are:
$
  hat(d)
    &= (hat("Avg")(y^("dark")) - hat(b))/(Delta t)
    = (hat("Avg")(y^("dark")) - hat("Avg")(y^("bias")))/(Delta t),\
  hat(g)
    &= (hat("Avg")(y^("dark")) - hat(b))/(hat("Var")(y^("dark")) - hat(sigma)_("ron")^2)
    = (hat("Avg")(y^("dark")) - hat("Avg")(y^("bias")))/(hat("Var")(y^("dark")) - hat("Var")(y^("bias")))
$

== Flat fields

Flat field images are acquired with a uniform illumination, say $x_(j) = u$ for any pixel $j$ (and $Delta t != 0$). The sample mean and variance of the flat field images are:
$
  hat("Avg")(y^("flat")) &≈ EE(y^("flat")) = (u/a + d) thin Delta t + b\
  hat("Var")(y^("flat")) &≈ "Var"(y^("flat")) = (u/a + d) thin (Delta t)/g + sigma^2_("ron")\
$
which yields the following possible estimator for $a$:
$
  hat(a)
    &= (u thin Delta t)/(hat("Avg")(y^("flat")) - hat(d) thin Delta t - hat(b))
$

If $u$ is unknown (e.g., because a calibrated source is not available), assuming $u = 1$ amounts to having $x$ expressed in units of the (unknown) flat field irradiance.


== Remarks

- If dark images have been acquired with different exposure times, $d$ can be estimated by fitting the model.

- What are the best weights for fitting the calibration data?

- In the calibration process, the quantities of interest are the (pixel-wise) sample mean and sample variance. We develop the software to estimate these quantities in the following.

== Functions

- `np.mean(data, axis=d)` or `data.mean(axis=d)` return the mean of `data` along axis `d`;
- `np.std(data, axis=d)` or `data.std(axis=d)` return the sample standard deviation of `data` along axis `d`;

== Inline statistics

When the data set consists in many large images, it may not be possible to read all the data in memory.

$
hat(m)_n &= 1/n sum_(t=1)^(n) x_t = (n-1)/n hat(m)_(n-1) + 1/n x_n\
hat(v)_n &= 1/(n - 1) sum_(t=1)^(n) (x_t - hat(m)_n)^2 \
&= 1/(n - 1) sum_(t=1)^(n) ((x_t - hat(m)_(n-1)) + (hat(m)_(n-1) - hat(m)_(n)))^2
$

== Exercices

- Write a code sample that update the sample mean and sample variance for each new image.

- What are the most relevant quantities to memorize in order to reduce the number of operations? How does inline statistics code based on recurrence formulae compares to the simple 2-pass method in terms of number of operations?

- Improve the code to take into account a mask that indicates the bad (or the good at your choice) pixels in each given image. The mask may be different for each image.

- Is it possible that the sample variance computed by recursion has some negative values? Consider that numerical operations in floating-point have limited precision and are subject to (small) rounding errors.

== Code

Typically your Python code will looks like:

```python
import numpy as np # for arrays
import matplotlib.pyplot as plt # for plotting
from astrofocus import * # for useful functions
from pathlib import Path
datadir = '/home/eric/data/FOCUS/detector_calibration/SBIG-STF-8300M/'
bias = [readfits(f) for f in Path(datadir + "/bias").iterdir() if f.is_file()]
len(bias)
sum1 = zeros(bias[0].shape);
for b in bias: sum1 += b;
avg = sum1/len(bias);
figure(2); plt.clf(); plt.imshow(avg,vmin=1000,vmax=1050);
plt.title("Mean bias ({} frame)");
sum2 = zeros(bias[0].shape);
for img in bias: sum2 += (img - avg)**2;
var = sum2/(len(bias) - 1);
figure(2); plt.clf(); plt.imshow(avg,vmin=1000,vmax=1050);

```
