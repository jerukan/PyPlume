"""
Collection of functions for performing DINEOF spatial gapfilling.

Written by Mika Siegelman 2023/02, last update 2023/02
"""
import random
import numpy as np


def EOF(A):
    """
    Uses SVD method to calculate EOFs.
    Input:
    -----
    A = matrix [nt x nx]

    Output:
    ------
    dictionary containing:
        u = u
        d = d
        v = v
        eigvals = eigenvalues
        eigvec  = eigenvectors (columns are eigenvalues) (v.T)
        tvecs   = temporal amplitude

    """
    nt, nx = np.shape(A)

    u, d, v = np.linalg.svd(A, full_matrices=False)  # returns V transposed (UDV')
    per_var = 100 * (d**2 / np.sum(d**2))
    eigvalues = (d**2) / nt  # variance contained in each mode
    eigvec = v.T  # eigenvectors or EOFs
    tvecs = np.dot(
        u, np.diag(d)
    )  # equivalent to projection of A on new basis, Principal Component
    eof = dict(u=u, d=d, v=v, eigvals=eigvalues, eigvec=eigvec, tvecs=tvecs)
    # print("The Eigenvalues:",eigvalues)
    # print("Variance of temporal amplitudes:",tvecs.var(axis=0))

    return eof


def reconstruction(eof, N):
    """
    Reconstructs EOF from SVD with N number of modes
    Input:
    -----
    eof = dictionary containing variables u,d,v from SVD
    N   = number of modes to use in reconstruction
    Output:
    ------
    rec = reconstruction using N modes, [nt x nx]
    """
    D = np.diag(eof["d"][:N])
    DV = np.dot(D, eof["v"][:N])
    rec = np.dot(eof["u"][:, :N], DV)
    return rec


def optimize_filled(Amaskin, maskT, nm, maxits, thresh):
    """
    Computes EOF then fills previous iteration of "bad" values with values from current EOF calculation.  Repeats
    for max iterations (maxits) until convergence (thresh) is reached.
    Input:
    -----
    Amaskin = Array filled w/ zeros in bad data
    nm  = number of modes to use for reconstruction
    maxits  = maximum iterations to test each EOF mode
    thresh  = threshold for convergence

    Output:
    ------
    Amask0 = The optimized Amaskin
    """
    Amask0 = Amaskin.copy()
    mse_i = mse(Amask0[maskT])
    for ni in range(maxits):
        Aeof = EOF(Amask0)
        Arec = reconstruction(Aeof, nm)
        Amask0[maskT] = Arec[maskT]  # fill your "bad" values with reconstructed values
        mse_rec = mse(Arec[maskT])
        errorcheck = np.abs(mse_rec - mse_i) / mse_i
        print(ni, errorcheck)
        if ni > 0 and errorcheck < thresh:
            print("Mode %s Reached convergence during iterative filling" % nm)
            break
        mse_i = mse_rec
    return Amask0


def optimize_N(Amaskin, modemax, valmask, maskT, vvalues, values_mse, thresh):
    """
    Finds the optimum number of EOFs to use to fill gappy data.
    Input:
    -----
    Amaskin = Array filled w/ zeros in bad data
    modemax = maximum mode numbers that will be allowed
    valmask = mask of validation dataset
    maskT   = mask of both "bad" data and validation dataset
    vvalues = the values of the validation dataset
    values_mse = mean square error of vaidation dataset
    thres   = threshold for convergence

    Output:
    ------
    nm = Ideal number of modes used in EOF
    Amask0 = The optimized Amaskin
    """
    Amask0 = Amaskin.copy()
    for nm in range(1, modemax + 1):
        Amask0 = optimize_filled(
            Amask0, maskT, nm, modemax, thresh
        )  # returns optimized reconstruction
        validdiff = (
            Amask0[valmask] - vvalues
        )  # difference between reconstructed points and validation dataset
        validmse = mse(validdiff)
        msediff = (
            values_mse - validmse
        )  # difference between previous iteration and latest reconstruction error
        values_mse = validmse  # change to become the standard for error for next step
        if (nm > 1) and (msediff < 0):
            # Stop because error is getting worse, but need to check past mode 1
            break

        if (nm == modemax) and (msediff > 0):
            print("Warning: Using %s did not reach convergence" % modemax)

    if nm < modemax:
        nm = nm - 1  # because the previous mode reconstruction is better
    print("Optimum number of modes: %s" % nm)
    return nm, Amask0


def fill_gappy_EOF(Amask, modemax, maxits, thresh=0.01):
    """
    Fills gappy data using method from: Beckers and Rixen et al. 2003
    Input:
    -----
    Amask = masked array [nt,nx]
    modemax = maximum number of EOFs allowed to be used for filling (number of EOFs used is optimized)
    maxits  = maximum iterations to test each EOF mode at
    thresh  = threshold for convergence

    Output:
    ------
    Afilled = filled data
    eoff    = EOF of Afilled
    """
    mask0 = Amask.mask
    Amask0 = Amask.copy()
    Amask_final = Amask.copy()
    valmask, vvalues = cross_validation(
        Amask, percentdata=0.05
    )  # validation mask values randomly selected from cross validation
    values_mse = mse(vvalues)  # mse of validation subset

    Amask0[mask0] = 0  # fill nan'd places w/ 0
    Amask0[valmask] = 0  # masks validation dataset
    maskT = mask0.copy()  # make mask with "bad" values AND validation dataset
    maskT[valmask == 1] = 1  # make mask with "bad" values AND validation dataset
    nm, AmaskOPT = optimize_N(
        Amask0, modemax, valmask, maskT, vvalues, values_mse, thresh
    )  # finds the optimized number of EOFs to use in reconstruction
    Amask_final[mask0] = AmaskOPT[mask0]
    Afilled = optimize_filled(Amask_final, mask0, nm, maxits, thresh)
    eoff = EOF(Afilled)
    return Afilled, eoff


def mse(dat):
    """
    Calculates mean square error
    Input:
    ------
    dat = array or matrix
    Output:
    ------
    Mean square error
    """
    return np.mean(dat**2)


def cross_validation(Amask, percentdata=0.05):
    """
    Makes mask that identifies the cross validation dataset.  This dataset is to used to determine optimum number of EOF modes to use
    Input:
    ------
    Amask       = masked array for EOF calculation [nt x nx]
    percentdata = Percentage of data used in cross validation dataset
    Output:
    ------
    valmask     = mask for validation dataset (True where data is in validation dataset)

    """
    mask0 = Amask.mask
    kkG = mask0 == 0  # identifier of valid data, 0 = GOOD DATA
    nt, nx = kkG.shape
    ix, it = np.meshgrid(
        np.arange(nx), np.arange(nt)
    )  # make indicies to identify each position in kkG
    Avalid = Amask[kkG]  # only good data
    xvalid = ix[kkG]  # x index of good positions
    tvalid = it[kkG]  # t index of good positions

    nvalid = int(
        len(Avalid) * percentdata
    )  # number of points to select randomly (set to 5% of data)
    inrand = random.sample(
        range(0, len(Avalid)), nvalid
    )  # identifies random sample of good data
    xinds, tinds = (
        xvalid[inrand],
        tvalid[inrand],
    )  # gets the x and t indicies of randomly selected data
    valmask = np.zeros_like(kkG, dtype=bool)  # initial mask
    valmask[tinds, xinds] = True  # True where data is part of validation dataset
    vvalues = Amask[
        valmask
    ]  # selects the randomly selected values to be used for cross validation
    return valmask, vvalues
