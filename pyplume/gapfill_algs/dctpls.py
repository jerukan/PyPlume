import copy
import warnings

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from numpy.linalg import norm
from numpy.random import random, rand, randn
import scipy
from scipy.fft import dct, idct, dctn, idctn
from scipy.optimize import fminbound


def smoothn(
    *y,
    s=None,
    W=None,
    smoothOrder=2,
    spacing=None,
    isrobust=False,
    TolZ=1e-3,
    MaxIter=100,
    Initial=None,
    weight="bisquare",
    full_output=False,
):
    """
    Ported directly from the MATLAB smoothn function:
    https://www.mathworks.com/matlabcentral/fileexchange/25634-smoothn

    The below is the documentation written by the MATLAB implementation
    but adapted for Python.

    SMOOTHN provides a fast, automatized and robust discretized spline
    smoothing for data of arbitrary dimension.

    Args:
        *y (np.ndarray): Variable arguments representing the fields to be smoothed. Each
            array Y can be any N-D noisy array (time series, images, 3D data,...). Non
            finite data (NaN or Inf) are treated as missing values. If you want to smooth
            a vector field or multicomponent data, Y can be supplied multiple arguments.
            For example, if you need to smooth a 3-D vectorial flow (Vx,Vy,Vz), use
            Zx,Zy,Zz = smoothn(Vx, Vy, Vz, ...).
        s (scalar): Smoothes the array Y using the smoothing parameter S. S must be
            a real positive scalar. The larger S is, the smoother the output will be.
            If the smoothing parameter S is omitted or None, it is automatically
            determined by minimizing the generalized cross-validation (GCV) score.
        W (np.ndarray): Array W of positive values, which must have the same size as
            Y. Note that a nil weight corresponds to a missing value. If None, all
            weights are assumed to be 1 and nil values are assigned 0.
        smoothOrder (0, 1, or 2): Integer value to determine how the bounds of the
            smoothing parameter s are found. Default value is 2.
        spacing (array_like): SMOOTHN, by default, assumes that the spacing increments
            are constant and equal in all the directions (i.e. dx = dy = dz = ...).
            This means that the smoothness parameter is also similar for each direction.
            If the increments differ from one direction to the other, it can be useful
            to adapt these smoothness parameters. You can thus use the following
            keyword argument:
                spacing = [d1, d2, d3, ...]
            where dI represents the spacing between points in the Ith dimension.
        isrobust (bool): If True, carries out a robust smoothing that minimizes
            the influence of outlying data (default = False).
        TolZ (scalar): Robust smoothing option. Termination tolerance on Z
            (default = 1e-3), TolZ must be in ]0,1[
        MaxIter (integer): Robust smoothing option. Maximum number of iterations
            allowed (default = 100).
        Initial (list of np.ndarray): Robust smoothing option. Initial guess of values
            for fields in *y. If None (which is default), guess initial value automatically
            (which is just the original data *y).
        weight: Robust smoothing option. Weight function for robust smoothing:
            'bisquare' (default), 'talworth' or 'cauchy'
        full_output: If True, the function also returns the calculated smoothness
            parameter S, as well as a boolean EXITFLAG that describes the exit condition
            of smoothn. Default False.

    Returns:
        smoothed Z (np.ndarray): The smoothed fields. If a single array was passed in,
            a single array is returned. If multiple arrays (multicomponent data) were
            passed in, then a list of the arrays is returned.
        S (scalar): Returns if full_output is True. The calculated smoothness parameter.
        EXITFLAG (bool): Returns if full_output is True. Describes the exit condition
            of SMOOTHN:
            1       SMOOTHN converged.
            0       Maximum number of iterations was reached.

    Notes
    -----
    scipy is required for the N-D (inverse) discrete cosine transform functions.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dctn.html
    and
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.idctn.html


    REFERENCES (please refer to the two following papers)
    ---------
    1) Garcia D, Robust smoothing of gridded data in one and higher
    dimensions with missing values. Computational Statistics & Data
    Analysis, 2010;54:1167-1178.
    http://www.biomecardio.com/publis/csda10.pdf
    2) Garcia D, A fast all-in-one method for automated post-processing of
    PIV data. Exp Fluids, 2011;50:1247-1259.
    http://www.biomecardio.com/publis/expfluids11.pdf

    -- Damien Garcia -- 2009/03, last update 2020/06
    website: http://www.biomecardio.com/en
    """
    y = list(copy.deepcopy(y))
    sizy = y[0].shape
    ny = len(y)

    noe = np.size(y)  # number of elements
    if noe < 2:
        z = y[0]
        exitflag = True
        if full_output:
            return z, s, exitflag
        return z
    # ---
    # Smoothness parameter and weights
    # if s != None:
    #    s = []
    if W is None:
        W = np.ones(sizy)

    # ---
    # "Maximal number of iterations" criterion
    if MaxIter is None:
        MaxIter = 100
    if not isinstance(MaxIter, int) or MaxIter < 1:
        raise ValueError("MaxIter must be an integer >=1")

    # ---
    # "Tolerance on smoothed output" criterion
    if TolZ is None:
        TolZ = 1e-3
    if TolZ >= 1 or TolZ <= 0:
        raise ValueError("TolZ must be in ]0,1[")

    # ---
    # "Initial Guess" critereon
    if Initial is None:
        isinitial = False
    else:
        z0 = Initial
        if np.shape(z0) != np.shape(y):
            raise ValueError("Initial must contain a valid initial guess for Z")

    # ---
    # "Weight function" criterion (for robust smoothing)
    if weight is None:
        weight = "bisquare"
    else:
        if not isinstance(weight, str):
            raise TypeError("A valid weight function (weight) must be chosen")
        weight = weight.lower()
        if weight not in {"bisquare", "talworth", "cauchy"}:
            raise ValueError(
                "The weight function must be 'bisquare', 'cauchy' or 'talworth'."
            )

    # ---
    # "Order" criterion (by default m = 3)
    # Note: m = 0 is of course not recommended!
    if smoothOrder is None:
        m = 2
    else:
        m = smoothOrder
        if m not in {0, 1, 2}:
            raise ValueError(
                "MATLAB:smoothn:IncorrectOrder\nThe order (smoothOrder) must be 0, 1 or 2."
            )

    # ---
    # "Spacing" criterion
    d = y[0].ndim
    if spacing is None:
        dI = np.ones(d)
    else:
        dI = np.array(spacing)
        # add type and dimension check here
    dI = dI / np.max(dI)

    # ---
    # Weights. Zero weights are assigned to not finite values (Inf or NaN),
    # (Inf/NaN values = missing data).
    IsFinite = np.all(np.isfinite(y), axis=0)
    nof = IsFinite.sum()  # number of finite elements
    W *= IsFinite
    if np.any(W < 0):
        raise ValueError("smoothn:NegativeWeights\nWeights must all be >=0")
    W /= np.max(W)
    # ---
    # Weighted or missing data?
    isweighted = np.any(W < 1)
    # ---
    # Robust smoothing?
    # isrobust
    # ---
    # Automatic smoothing?
    isauto = not s

    ## Creation of the Lambda tensor
    # ---
    # Lambda contains the eingenvalues of the difference matrix used in this
    # penalized least squares process (see CSDA paper for details).
    Lambda = np.zeros(sizy)
    for i in range(d):
        siz0 = np.ones(d, dtype=int)
        siz0[i] = sizy[i]
        Lambda += (
            2
            - 2
            * np.cos(np.pi * (np.arange(1, sizy[i] + 1).reshape(siz0) - 1) / sizy[i])
        ) / dI[i] ** 2
    # Gamma is recalculated after the first iteration anyway
    # if not isauto:
    #     Gamma = 1.0 / (1 + s * Lambda ** m)

    ## Upper and lower bound for the smoothness parameter
    # The average leverage (h) is by definition in [0 1]. Weak smoothing occurs
    # if h is close to 1, while over-smoothing appears when h is near 0. Upper
    # and lower bounds for h are given to avoid under- or over-smoothing. See
    # equation relating h to the smoothness parameter (Equation #12 in the
    # referenced CSDA paper).
    N = np.sum(np.array(sizy) != 1)  # tensor rank of the y-array
    hMin = 1e-6
    hMax = 0.99
    if m == 0:  # Not recommended. For mathematical purpose only.
        sMinBnd = 1 / hMax ** (1 / N) - 1
        sMinBnd = 1 / hMin ** (1 / N) - 1
    elif m == 1:
        sMinBnd = (1 / hMax ** (2 / N) - 1) / 4
        sMaxBnd = (1 / hMin ** (2 / N) - 1) / 4
    elif m == 2:
        sMinBnd = (
            ((1 + np.sqrt(1 + 8 * hMax ** (2.0 / N))) / 4.0 / hMax ** (2.0 / N)) ** 2
            - 1
        ) / 16.0
        sMaxBnd = (
            ((1 + np.sqrt(1 + 8 * hMin ** (2.0 / N))) / 4.0 / hMin ** (2.0 / N)) ** 2
            - 1
        ) / 16.0
    ## Initialize before iterating
    # ---
    Wtot = W
    # --- Initial conditions for z
    if isweighted:
        # --- With weighted/missing data
        # An initial guess is provided to ensure faster convergence. For that
        # purpose, a nearest neighbor interpolation followed by a coarse
        # smoothing are performed.
        # ---
        if isinitial:  # an initial guess (z0) has been already given
            z = z0
        else:
            z = InitialGuess(y, IsFinite)
    else:
        z = [np.zeros(sizy) for _ in range(ny)]
    # ---
    z0 = copy.deepcopy(z)
    for yi in y:
        yi[~IsFinite] = 0  # arbitrary values for missing y-data
    # ---
    tol = 1.0
    RobustIterativeProcess = True
    RobustStep = 1
    nit = 0
    DCTy = [np.empty(sizy) for _ in range(ny)]
    # --- Error on p. Smoothness parameter s = 10^p
    errp = 0.1
    # --- Relaxation factor RF: to speedup convergence
    RF = 1 + 0.75 * isweighted
    ## Main iterative process
    # ---
    if not isauto:
        # auto will run GCV on first iteration, xpost will be set by then
        xpost = np.log10(s)
    while RobustIterativeProcess:
        # --- "amount" of weights (see the function GCVscore)
        aow = np.sum(Wtot) / noe  # 0 < aow <= 1
        # ---
        while tol > TolZ and nit < MaxIter:
            nit = nit + 1
            # DCTy = dctND(Wtot * (y - z) + z, f=dct)
            for i in range(ny):
                DCTy[i] = dctn(Wtot * (y[i] - z[i]) + z[i])
            if isauto and not np.remainder(np.log2(nit), 1):
                # ---
                # The generalized cross-validation (GCV) method is used.
                # We seek the smoothing parameter S that minimizes the GCV
                # score i.e. S = Argmin(GCVscore).
                # Because this process is time-consuming, it is performed from
                # time to time (when the step number - nit - is a power of 2)
                # ---
                gcv_f = lambda p: gcv(
                    p, Lambda, aow, DCTy, IsFinite, Wtot, y, nof, noe, m, ny
                )
                xpost = fminbound(
                    gcv_f, np.log10(sMinBnd), np.log10(sMaxBnd), xtol=errp
                )
            s = 10**xpost
            Gamma = 1.0 / (1 + s * Lambda**m)

            for i in range(ny):
                z[i] = RF * idctn(Gamma * DCTy[i]) + (1 - RF) * z[i]

            # z = RF * dctND(Gamma * DCTy, f=idct) + (1 - RF) * z
            # z = RF * idctn(Gamma * DCTy) + (1 - RF) * z
            # if no weighted/missing data => tol=0 (no iteration)
            tol = (
                isweighted
                * norm((np.array(z0) - np.array(z)).flatten())
                / norm(np.array(z).flatten())
            )

            z0 = copy.deepcopy(z)  # re-initialization
        exitflag = nit < MaxIter

        if isrobust:  # -- Robust Smoothing: iteratively re-weighted process
            # --- average leverage
            h = 1
            for i in range(N):
                if m == 0:  # not recommended - only for numerical purpose
                    h0 = 1 / (1 + s / dI[i] ** (2**m))
                elif m == 1:
                    h0 = 1 / np.sqrt(1 + 4 * s / dI[i] ** (2**m))
                elif m == 2:
                    h0 = np.sqrt(1 + 16 * s / dI[i] ** (2**m))
                    h0 = np.sqrt(1 + h0) / np.sqrt(2) / h0
                h *= h0
            # --- take robust weights into account
            Wtot = W * RobustWeights(y, z, IsFinite, h, weight)
            # --- re-initialize for another iterative weighted process
            isweighted = True
            tol = 1
            nit = 0
            # ---
            RobustStep = RobustStep + 1
            RobustIterativeProcess = RobustStep < 4  # 3 robust steps are enough.
        else:
            RobustIterativeProcess = False  # stop the whole process

    ## Warning messages
    # ---
    if isauto:
        if np.abs(np.log10(s) - np.log10(sMinBnd)) < errp:
            warnings.warn(
                f"MATLAB:smoothn:SLowerBound\ns = {s:.3f} : the lower bound for s has been reached. Put s as an input variable if required."
            )
        elif np.abs(np.log10(s) - np.log10(sMaxBnd)) < errp:
            warnings.warn(
                f"MATLAB:smoothn:SUpperBound\ns = {s:.3f} : the upper bound for s has been reached. Put s as an input variable if required."
            )
    if not exitflag:
        warnings.warn(
            f"MATLAB:smoothn:MaxIter\nMaximum number of iterations (' {MaxIter} ') has been exceeded. Increase MaxIter option or decrease TolZ value."
        )
    if len(z) == 1:
        z = z[0]
    if full_output:
        return z, s, exitflag
    return z


## GCV score
# ---
# function GCVscore = gcv(p)
def gcv(p, Lambda, aow, DCTy, IsFinite, Wtot, y, nof, noe, m, ny):
    """
    Search the smoothing parameter s that minimizes the GCV score.

    The parameter s is based on the input p.
    """
    # ---
    s = 10**p
    Gamma = 1.0 / (1 + s * Lambda**m)
    # --- RSS = Residual sum-of-squares
    RSS = 0
    if aow > 0.9:  # aow = 1 means that all of the data are equally weighted
        # very much faster: does not require any inverse DCT
        for i in range(ny):
            RSS += norm(DCTy[i].flatten() * (Gamma.flatten() - 1.0)) ** 2
    else:
        # take account of the weights to calculate RSS:
        # yhat = dctND(Gamma * DCTy, f=idct)
        for i in range(ny):
            yhat = idctn(Gamma * DCTy[i])
            RSS += (
                norm(np.sqrt(Wtot[IsFinite]) * (y[i][IsFinite] - yhat[IsFinite])) ** 2
            )
    # ---
    TrH = np.sum(Gamma)
    GCVscore = RSS / nof / (1.0 - TrH / noe) ** 2
    return GCVscore


## Robust weights
# function W = RobustWeights(y,z,I,h,wstr)
def RobustWeights(y, z, I, h, wstr):
    # One seeks the weights for robust smoothing...
    ABS = lambda x: np.sqrt(np.sum(np.abs(x) ** 2, axis=0))
    r = np.array(y) - np.array(z)  # residuals
    Icnt = I.sum()
    Imask = np.broadcast_to(I, r.shape)
    rI = r[Imask].reshape(-1, Icnt)
    MMED = np.median(rI, axis=1)  # marginal median
    AD = ABS(rI - MMED[:, np.newaxis])  # absolute deviation
    MAD = np.median(AD, axis=0)  # median absolute deviation

    # -- studentized residuals
    u = ABS(r) / (1.4826 * MAD) / np.sqrt(1 - h)
    u = u.reshape(I.shape)

    if wstr == "cauchy":
        c = 2.385
        W = 1.0 / (1 + (u / c) ** 2)  # Cauchy weights
    elif wstr == "talworth":
        c = 2.795
        W = u < c  # Talworth weights
    elif wstr == "bisquare":
        c = 4.685
        W = (1 - (u / c) ** 2) ** 2.0 * ((u / c) < 1)  # bisquare weights
    else:
        raise ValueError(
            "MATLAB:smoothn:IncorrectWeights\nA valid weighting function must be chosen"
        )
    W[np.isnan(W)] = 0
    return W


## Initial Guess with weighted/missing data
# function z = InitialGuess(y,I)
def InitialGuess(y, I):
    """The MATLAB code is hard to read, so I gave up on porting this function."""
    z = copy.deepcopy(y)
    for zi in z:
        zi[~I] = 0  # arbitrary values for missing data
    return z

    # ny = len(y)
    # # -- nearest neighbor interpolation (in case of missing values)
    # z = copy.deepcopy(y)
    # if np.any(~I):
    #     from scipy.ndimage.morphology import distance_transform_edt

    #     # if license('test','image_toolbox')
    #     # [z,L] = bwdist(I)
    #     _, L = distance_transform_edt(1 - I, return_indices=True)
    #     for i in range(ny):
    #         z[i][~I] = y[i][L[~I]]
    # # coarse fast smoothing
    # # z = dctND(z, f=dct)

    # for i in range(ny):
    #     z[i] = dctn(z[i])
    #     k = np.array(z.shape)
    #     m = np.ceil(k / 10) + 1
    #     d = []
    #     for i in range(len(k)):
    #         d.append(np.arange(m[i], k[i]))
    #     d = np.array(d, dtype=int)
    #     z[i][d] = 0.0
    #     # z = dctND(z, f=idct)
    #     z[i] = idctn(z[i])
    # return z
    # # -- coarse fast smoothing using one-tenth of the DCT coefficients
    # # siz = z.shape
    # # z = dct(z,norm='ortho',type=2)
    # # for k in np.arange(len(z.shape)):
    # #        z[ceil(siz[k]/10)+1:-1] = 0
    # #        ss = tuple(roll(array(siz),1-k))
    # #        z = z.reshape(ss)
    # #        z = np.roll(z.T,1)
    # # z = idct(z,norm='ortho',type=2)


# NB: filter is 2*I - (np.roll(I,-1) + np.roll(I,1))


def dctND(data, f=dct):
    """Multidimensional DCT (unneeded due to the existence of dctn and idctn in scipy)"""
    nd = len(data.shape)
    if nd == 1:
        return f(data, norm="ortho", type=2)
    elif nd == 2:
        return f(f(data, norm="ortho", type=2).T, norm="ortho", type=2).T
    elif nd == 3:
        return f(
            f(f(data, norm="ortho", type=2, axis=0), norm="ortho", type=2, axis=1),
            norm="ortho",
            type=2,
            axis=2,
        )


def peaks(n):
    """
    Mimic basic of matlab peaks fn
    """
    xp = np.arange(n)
    [x, y] = np.meshgrid(xp, xp)
    z = np.zeros_like(x).astype(float)
    for i in range(int(n / 5)):
        x0 = random() * n
        y0 = random() * n
        sdx = random() * n / 4.0
        sdy = sdx
        c = random() * 2 - 1.0
        f = np.exp(
            -(((x - x0) / sdx) ** 2)
            - ((y - y0) / sdy) ** 2
            - (((x - x0) / sdx)) * ((y - y0) / sdy) * c
        )
        # f /= f.sum()
        f *= random()
        z += f
    return z


def test1():
    plt.figure(1)
    plt.clf()
    # 1-D example
    x = np.linspace(0, 100, 2**8)
    y = np.cos(x / 10) + (x / 50) ** 2 + randn(np.size(x)) / 10
    y[[70, 75, 80]] = [5.5, 5, 6]
    z = smoothn(y)  # Regular smoothing
    zr = smoothn(y, isrobust=True)  # Robust smoothing
    plt.subplot(121)
    plt.plot(x, y, "r.")
    plt.plot(x, z, "k")
    plt.title("Regular smoothing")
    plt.subplot(122)
    plt.plot(x, y, "r.")
    plt.plot(x, zr, "k")
    plt.title("Robust smoothing")


def test2():
    # 2-D example
    plt.figure(2)
    plt.clf()
    xp = np.arange(0, 1, 0.02)
    [x, y] = np.meshgrid(xp, xp)
    f = np.exp(x + y) + np.sin((x - 2 * y) * 3)
    fn = f + (randn(f.size) * 0.5).reshape(f.shape)
    fs = smoothn(fn, isrobust=True)
    plt.subplot(121)
    plt.imshow(fn, interpolation="Nearest")  # axis square
    plt.subplot(122)
    plt.imshow(fs, interpolation="Nearest")  # axis square


def test3():
    # 2-D example with missing data
    plt.figure(3)
    plt.clf()
    n = 256
    y0 = peaks(n)
    y = (y0 + random(np.shape(y0)) * 2 - 1.0).flatten()
    I = np.random.permutation(range(n**2))
    y[I[1 : int(n**2 * 0.5)]] = np.nan  # lose 50% of data
    y = y.reshape(y0.shape)
    y[40:90, 140:190] = np.nan  # create a hole
    yData = y.copy()
    z0 = smoothn(yData)  # smooth data
    yData = y.copy()
    z = smoothn(yData, isrobust=True)  # smooth data
    y = yData
    vmin = np.min([np.min(z), np.min(z0), np.min(y0)])
    vmax = np.max([np.max(z), np.max(z0), np.max(y0)])
    plt.subplot(221)
    plt.imshow(y, interpolation="Nearest", vmin=vmin, vmax=vmax)
    plt.title("Noisy corrupt data")
    plt.subplot(222)
    plt.imshow(z0, interpolation="Nearest", vmin=vmin, vmax=vmax)
    plt.title("Recovered data")
    plt.subplot(223)
    plt.imshow(z, interpolation="Nearest", vmin=vmin, vmax=vmax)
    plt.title("Recovered data robust")
    plt.subplot(224)
    plt.imshow(y0, interpolation="Nearest", vmin=vmin, vmax=vmax)
    plt.title("... compared with original data")


def test4(i=10, step=0.2):
    [x, y, z] = np.mgrid[-2:2:step, -2:2:step, -2:2:step]
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    xslice = [-0.8, 1]
    yslice = 2
    zslice = [-2, 0]
    v0 = x * np.exp(-(x**2) - y**2 - z**2)
    vn = v0 + randn(x.size).reshape(x.shape) * 0.06
    v = smoothn(vn)
    plt.figure(4)
    plt.clf()
    vmin = np.min([np.min(v[:, :, i]), np.min(v0[:, :, i]), np.min(vn[:, :, i])])
    vmax = np.max([np.max(v[:, :, i]), np.max(v0[:, :, i]), np.max(vn[:, :, i])])
    plt.subplot(221)
    plt.imshow(v0[:, :, i], interpolation="Nearest", vmin=vmin, vmax=vmax)
    plt.title("clean z=%d" % i)
    plt.subplot(223)
    plt.imshow(vn[:, :, i], interpolation="Nearest", vmin=vmin, vmax=vmax)
    plt.title("noisy")
    plt.subplot(224)
    plt.imshow(v[:, :, i], interpolation="Nearest", vmin=vmin, vmax=vmax)
    plt.title("cleaned")


def test5():
    t = np.linspace(0, 2 * np.pi, 1000)
    x = 2 * np.cos(t) * (1 - np.cos(t)) + randn(np.size(t)) * 0.1
    y = 2 * np.sin(t) * (1 - np.cos(t)) + randn(np.size(t)) * 0.1
    zx = smoothn(x)
    zy = smoothn(y)
    plt.figure(5)
    plt.clf()
    plt.title("Cardioid")
    plt.plot(x, y, "r.")
    plt.plot(zx, zy, "k")


def test6(noise=0.05, nout=30):
    plt.figure(6)
    plt.clf()
    [x, y] = np.meshgrid(np.linspace(0, 1, 24), np.linspace(0, 1, 24))
    Vx0 = np.cos(2 * np.pi * x + np.pi / 2) * np.cos(2 * np.pi * y)
    Vy0 = np.sin(2 * np.pi * x + np.pi / 2) * np.sin(2 * np.pi * y)
    Vx = Vx0 + noise * randn(24, 24)  # adding Gaussian noise
    Vy = Vy0 + noise * randn(24, 24)  # adding Gaussian noise
    I = np.random.permutation(range(Vx.size))
    Vx = Vx.flatten()
    Vx[I[0:nout]] = (rand(nout) - 0.5) * 5  # adding outliers
    Vx = Vx.reshape(Vy.shape)
    Vy = Vy.flatten()
    Vy[I[0:nout]] = (rand(nout) - 0.5) * 5  # adding outliers
    Vy = Vy.reshape(Vx.shape)
    Vsx, Vsy = smoothn(Vx, Vy, isrobust=True)
    plt.subplot(131)
    plt.quiver(x, y, Vx, Vy, 2.5)
    plt.title("Noisy")
    plt.subplot(132)
    plt.quiver(x, y, Vsx, Vsy)
    plt.title("Recovered")
    plt.subplot(133)
    plt.quiver(x, y, Vx0, Vy0)
    plt.title("Original")
