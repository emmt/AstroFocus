#import "@preview/mosaic:0.0.1" as mosaic
#import mosaic.themes.metropolis as m
#show: m.setup.with(
  title: [Basics of Image Processing],
  subtitle: [with Python examples],
  authors: [Éric Thiébaut, Jean-Philippe Berger & Ferréol Soulez],
  date: [November 2026, Observatoire de Haute Provence],
)

// Reset raw blocks to the same size as normal text,
// but keep inline raw at the reduced size.
#show raw.where(block: true): set text(1em / 0.8)

// Make inline raw text colored.
#show raw.where(block: false): set text(eastern)

#m.slide(layout: "title")

= Prerequisites

== Required Software

- Python with packages:
  - `NumPy` for fast computations of numerical arrays (images);
  - `Matplotlib` for plots;
  - `Astropy` for reading and writing FITS files;
  - `SciPy` for optimization (https://scipy.org/);

- Custom package `AstroFocus` (https://github.com/emmt/AstroFocus), 2 possibilities:
  - ```sh git clone https://github.com/emmt/AstroFocus```
  - download and unpack code at:\ https://github.com/emmt/AstroFocus/archive/refs/heads/main.zip

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

Typically your Python code will looks like (replace `some_dir` by a suitable path):

```python
import sys
sys.path.append("some_dir/astrofocus/src")
import numpy as np # for arrays
import matplotlib.pyplot as plt # for plotting
import astrofocus as focus # for useful functions
...
```

In the following examples, the two first lines will be omitted (just as if `AstroFocus` was a normal package).

= Task 1: Detector calibration

== Captured electrons

Number of electrons captured by a pixel well:
$
  p = cal(P)((q thin r + c) thin Delta t)
$
- $cal(P)(mu)$ denotes the realization of a Poisson process of mean $mu$,
- $q$ is the *quantum efficiency* (QE),
- $r$ is the *irradiance* (in photons per unit of time),
- $c$ is the *dark current* (in electrons par unit of time),
- $Delta t$ is the *exposure time*.

There are two different physical processes:
- *photoelectric effect* with $q thin r thin Delta t$ the mean number of resulting photo-electrons,
- *dark current* with $c thin Delta t$ the mean number of resulting electrons.

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

Remark: unmixing the effects of $q$ and $g$ would require an absolute calibration source (usually not available on sky), this is why we introduced $x$ (and thus $a$).

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
- $g$ and $b$ are the *gain* and *bias* of the DAC (in e$""^-$/ADU and ADU),
- $d$ is the *dark current* (in ADU per unit of time),
- $a$ is a *conversion factor* (in units of $x$ per ADU),
- $sigma_("ron")$ is the standard deviation of the read-out + quantization noises (in ADU).

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

== Remarks

+ If $u$, the flat field irradiance, is unknown (e.g., because a calibrated source is not available), assuming $u = 1$ amounts to having $x$ expressed in units of the (unknown) flat field irradiance.

+ If dark images have been acquired with different exposure times, $d$ can be estimated by fitting the model. What does it means?

+ What are the best weights for fitting the calibration data?

+ In the calibration process, the building blocks are the (pixel-wise) *sample mean* and *sample variance*. We will develop the software to estimate these quantities in the following.

= Tools: Sample statistics

== Sample statistics

The pixel-wise sample mean and variance of a sequence of $n$ images $[y_(k)]_(k=1,...,n)$ are defined by:
$
hat(m)_n &= 1/n thin sum_(k=1)^(n) y_k\
hat(v)_n &= 1/n thin sum_(k=1)^(n) (y_k - hat(m)_n)^2
$

Remarks:

- $y_(k)$ denotes the $k$-th image of the sequence (with all its pixels).
- Arithmetic operations (addition, subtraction, division, taking the square, etc.) in the above formulae are meant to be applied *element-wise*.

== A first Python implementation for the sample mean

In Python, an image is typically stored as an `numpy.ndarray` provided by the `NumPy` package, the _de facto_ standard for dealing with large numerical arrays.

The convention that operations be applied *element-wise* matches the implementation of operations and elementary functions in `NumPy`. It is then straightforward to write Python code similar to the equations and with no explicit loops on the pixel indices:

```python
s1 = 0
for y in seq:
    s1 += y
n = len(seq) # number of images
avg = s1/n # sample mean
```

where `seq` is a Python `list` of images (each is an `numpy.ndarray`).

== A first Python implementation for the sample variance

Computing the sample variance (after the sample mean) follows the same pattern:

```python
s1 = 0
for y in seq:
    s1 += y
n = len(seq) # number of images
avg = s1/n # sample mean
s2 = 0
for y in seq:
    s2 += (y - avg)**2
var = s2/n # sample variance
```

Exercises (addressed in the next slides):

+ Improve the code to save the first (element-wise) additions.
+ How to build the sequence of images?

== Saving computations for the sample mean and variance

#[
  #set text(0.9em)

Saving the first pixel-wise addition in each loop can be done as follows:

```python
# helper function:
def increment(accumulator, term):
    if accumulator is None:
        return term
    else:
        return accumulator + term

s1 = None
for y in seq:
    s1 = increment(s1, y)
n = len(seq) # number of images
avg = s1/n # sample mean
s2 = None
for y in seq:
    s2 = increment(s2, (y - avg)**2)
var = s2/n # sample variance
```
]

== Reading the sequence of data

Suppose that the bias images are saved as (compressed) FITS files in some directory, say `data/bias`:

```bash
$ ls data/bias
bias-0001.fits.gz  bias-0002.fits.gz  bias-0003.fits.gz  ...
```

Reading the sequence of bias images can be done by:

```python
import astrofocus as focus # for useful functions like readfits
import glob
files = glob.glob('data/bias/bias-*.fits.gz')
seq = list(focus.readfits(file) for file in files)
```

*Exercise:* Write a simple script based on the previous snippets to compute the sample mean and variance of your bias images stored in a Python list as above.


== Sequence of data as a single `NumPy` array

#[
  #set text(0.9em)

`NumPy` provides functions to compute the sample statistics of an array `A`:
- `np.mean(A, axis=d)` or `A.mean(axis=d)` $->$ mean of `A` along axis `d`;
- `np.var(A, axis=d)` or `A.var(axis=d)` $->$ sample variance of `A` along axis `d`.
But this requires to have the data sequence in a single `numpy.ndarray`:

```python
import numpy as np # for arrays
import astrofocus as focus # for useful functions like readfits
import glob
files = glob.glob('data/bias/bias-*.fits.gz')
n = len(files) # number of images
first_image = focus.readfits(files[0]) # read the 1st image
data = np.empty((n,) + first_image.shape, first_image.dtype)
data[0, ...] = first_image
for k in range(1, n):
    data[k, ...] = focus.readfits(files[k])
```
]

== Practice

1. If each image of the sequence has $m_1 × m_2$ pixels, what is the shape of `data`, the `num,py.ndarray` build and filled in the previous piece of code?
+ What is the axis number to use with keyword `axis` in `np.mean` and `np.var`?
+ Justify the choice of the position of the additional dimension.
+ Write two scripts to compute the sample mean and variance of your bias images: one with a list of images, the other for all data in a single `numpy.ndarray`. Ensure that the two give the same results.
+ Which of the two methods require the more computations?
+ What is the drawback of these methods when the sequence consists in a large number of images each with a large number of pixels?

For the 2 last questions, think in terms of number of operations, commonly named *flops* (for floating-point operations), and in terms of occupied memory.

== A naïve single pass method: math and code

#[
  #set text(0.85em)

/*It is in fact not necessary to store all the data sequence in memory. */ Using the following property of the sample variance:

$
    hat(v)_(n) = 1/n thin sum_(k=1)^(n) (y_(k) - hat(m)_(n))^2 = (1/n thin sum_(k=1)^(n) y_(k)^2) - hat(m)_(n)^2
$
which is the analog of $"Var"(y) = EE((y - EE(y))^2) = EE(y^2) - EE(y)^2$, we can do:

```python
import astrofocus as focus
import glob
files = glob.glob('data/bias/bias-*.fits.gz')
s1, s2 = None, None
for file in files:
    y = focus.readfits(file)
    s1 = increment(s1, y)
    s2 = increment(s2, y**2)
n = len(files) # number of images
avg = s1/n
var = s2/n - avg**2
```
]

== A naïve single pass method: exercises

1. Justify the name of *single pass method* for this kind of code?

   The previous versions of the code to compute the sample statistics were
   *two-pass methods*: they can be written so as to avoid memorizing all the
   images simultaneously to the cost of reading the images twice.

+ What are the memory requirements of the naïve single pass method? Does it
  depend on the length of the sequence?

+ Prove the property of the sampled variance used in the naïve single pass
  method. (Hint: just develop the square and simplify.)

== A naïve single pass method: the failure

The naïve single pass method may be *unstable* and *grossly wrong* due to
rounding errors. *It may even be the case that the sampled variance has
negative values!*

Example:
```python
Python 3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy as np
>>> y = 1 + np.array([2.827, -3.435, -3.423])*1e-9
>>> (y**2).mean() - (y.mean())**2
-1.1102230246251565e-16
```

Can you explain why? (You may notice that $EE(y)^2 >> "Var"(y)$.)

This issue is solved in the following thanks to *stable recurrence formulae*.

== Recurrence for the sample mean

The sample mean for $n + 1$ observations can be expanded as follows:
$
  hat(m)_(n+1)
  = 1/(n+1) sum_(k=1)^(n+1) y_k
  &= 1/(n+1) sum_(k=1)^(n) y_k + y_(n+1)/(n+1) \
  &= n/(n+1) thin hat(m)_(n) + y_(n+1)/(n+1) \
  &= hat(m)_(n) + 1/(n+1) thin (y_(n+1) - hat(m)_(n)) \
$
to obtain the recurrence:
$
    hat(m)_1 &= y_1 \
    hat(m)_(n+1) &= hat(m)_(n) + u_(n)
    quad "with" quad u_(n) = 1/(n + 1) thin (y_(n+1) - hat(m)_(n))
    quad "for" quad n ≥ 1 \
$

== Recurrence for the sample variance

The sample variance for $n + 1$ observations:
$
  hat(v)_(n+1)
    &= 1/(n+1) sum_(k=1)^(n+1) (y_k - hat(m)_(n+1))^2
$
can be rewritten to obtain the recurrence:
$
    hat(v)_1 &= 0 \
    hat(v)_(n+1) &=  n/(n+1) thin hat(v)_(n) + n thin u_(n)^2
    quad "with" quad u_(n) = 1/(n + 1) thin (y_(n+1) - hat(m)_(n))
    quad "for" quad n ≥ 1 \
$
where $u_n^2$ denotes $u_n$ squared element by element.

*Exercise:* Prove the above identity for $hat(v)_(n+1)$.

== Proof of the recurrence for the sample variance

We use $u_n = hat(m)_(n+1) - hat(m)_(n) = frac((y_(n+1) - hat(m)_(n)),(n + 1),style: "horizontal")$ to expand the sample variance for $n +1 $ observations:
$
  hat(v)_(n+1)
    &= 1/(n+1) sum_(k=1)^(n+1) (y_k - hat(m)_(n+1))^2
    = 1/(n+1) sum_(k=1)^(n+1) (y_k - hat(m)_(n) - u_(n))^2 \
    &= 1/(n+1) sum_(k=1)^(n) [(y_k - hat(m)_(n)) - u_(n)]^2
      + (y_(n+1) - hat(m)_(n) - u_(n))^2/(n+1) \
    &= 1/(n+1) sum_(k=1)^(n) (y_k - hat(m)_(n))^2
      - (2 thin u_(n))/(n+1) underbrace(sum_(k=1)^(n) (y_k - hat(m)_(n)),0)
      + (n thin u_(n)^2)/(n+1)
      + (n^2 thin u_(n)^2)/(n + 1) \
    &= n/(n+1) thin v_(n) + n thin u_n^2 quad qed\
$

== Online statistics formulae

The sample mean and sample variance of independent observations $(y_k)_(k=1,2,...)$ can be updated by the following recurrence rules:

$
  hat(m)_(n+1) &= cases(
    y_(1) & "if" n = 0,
    hat(m)_(n) + u_(n) & "if" n > 0) \
  hat(v)_(n+1) &= cases(
    0 & "if" n = 0,
    n/(n+1) thin hat(v)_(n) + n thin u_(n)^2 & "if" n > 0) \
  u_n &= 1/(n + 1) thin (y_(n+1) - hat(m)_(n)) \
$

== Online statistics code

```python
import astrofocus as focus
import glob
n, avg, var = 0, None, None
for file in glob.glob('data/bias/bias-*.fits.gz'):
    y = focus.readfits(file)
    if n == 0:
        n = 1
        avg = y
        var = 0
    else:
        np1 = n + 1
        u = (y - avg)/np1
        avg += u
        var = (n/np1)*var + n*u**2
```

== Online statistics with AstroFocus

The `AstroFocus` package provides a simple `SampleStatistics` class to compute
the sample mean and variance by recurrence. The previous example becomes:

```python
import astrofocus as focus
import glob
stat = focus.SampleStatictics() # initial object with no observations
for file in glob.glob('data/bias/bias-*.fits.gz'):
    stat.update(focus.readfits(file))
n = stat.count()
avg = stat.mean()
var = stat.var()
```

You may save the sample statistics to a (compressed) FITS file:

```python
stat.writeto('data/bias-stat.fits.gz')
```

and, later, recover the sample statistics by:

```python
stat = focus.SampleStatistics('data/bias-stat.fits.gz')
```

== Exercices

+ Is it possible that the sample variance computed by recursion has some
  negative values? Consider that numerical operations in floating-point have
  limited precision and are subject to (small) rounding errors.

+ Write a code sample that update the sample mean and sample variance for each
  new image.

+ What are the most relevant quantities to memorize in order to reduce the
  number of operations? How does inline statistics code based on recurrence
  formulae compares to the simple two-pass method in terms of number of
  operations?

+ Improve the code to take into account a mask that indicates the bad (or the
  good at your choice) pixels in each given image. The mask may be different
  for each image.

== Unbiased variance

#[#set text(0.8em)
The sample variance is a biased estimator of the variance:
$
  EE(hat(v)_n) = (n - 1)/n thin "Var"(y)
$
an unbiased estimator is thus given by:
$
  hat(s)_(n) = n/(n -1) thin hat(v)_n
  = 1/(n -1) thin sum_(k=1)^(n) (y_k - hat(m)_n)^2
$

With `AstroFocus`, the unbiased variance of an instance of
`focus.SampleStatistics` is given by:

```python
var = stat.var(corrected=True) # get unbiased estimator of the variance
```

+ Prove the above expectation of the sample variance.
+ To calibrate a detectors which of $hat(v)_n$ or $hat(s)_n$ shall be used?
+ Modify your scripts to compute or update the _unbiased_ sample variance.

]

== Online statistics: alternate formulation

#[#set text(0.9em)

Originally Welford (1962) proposed to update the sample mean $hat(m)_n$ and the
sum of squared residuals:

$
  hat(q)_(n) = sum_(k=1)^(n) (y_k - hat(m)_n)^2
$

by the following recurrence rules (with $r_(n) = y_(n+1) - hat(m)_n$):
$
  hat(m)_(n+1) &= hat(m)_n + r_n/(n + 1) \
  hat(q)_(n+1) &= hat(q)_n + r_n thin (y_(n+1) - hat(m)_(n + 1))
  = hat(q)_n + n/(n + 1) thin r_(n)^2 \
$

+ How to initialize the recurrence?

+ Prove these rules and, hence, that $hat(v)_(n)= frac(hat(q)_n, n, style:
  "horizontal")$ and $hat(s)_(n)= frac(hat(q)_n, (n - 1), style:
  "horizontal")$ hold.

+ Is there any advantages or drawbacks to this approach?
]

== Online statistics references

#[#set text(0.8em)

Updating formulae for computing the sample mean and central moments of
independent observations have many applications and have been studied and
developed by many authors:

- B. P. Welford, 1962, _"Note on a Method for Calculating Corrected Sums of
  Squares and Products"_, Technometrics *4*, pp. 419-420
  (https://amstat.tandfonline.com/doi/abs/10.1080/00401706.1962.10490022)

  An early publication of the recurrence formulae for the sample mean and
  (unbiased) variance.

- West

- Chan et al.

- Pébay et al.
]

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
