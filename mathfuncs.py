import numpy as n
from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy import stats
from scipy import fft
from scipy.special import gamma as gammafunct
from scipy.special import kv
from scipy.stats import gennorm
from scipy.optimize import curve_fit
from statsmodels.nonparametric import smoothers_lowess as slw

# import sympy as sp
from scipy.special import kv as _besselk, gamma as _gammafunct

from scipy.sparse.linalg import eigs, eigsh


def var_explained(true, pred):
    """
    Compute the fraction of variance explained by the model.

    Args:
        true (ndarray): True values. Can be 1D or 2D (n_cells x n_timepoints).
        pred (ndarray): Predicted values. Same shape as true.

    Returns:
        float or ndarray:
            - If input is 1D: single float with fraction of variance explained
            - If input is 2D: array of shape (n_cells,) with variance explained for each cell
    """
    true = n.asarray(true)
    pred = n.asarray(pred)

    if true.shape != pred.shape:
        raise ValueError("True and predicted arrays must have the same shape.")

    if true.ndim == 1:
        # Original behavior for 1D arrays
        ss_total = n.sum((true - n.mean(true)) ** 2)
        ss_residual = n.sum((true - pred) ** 2)
        return 1 - (ss_residual / ss_total)

    elif true.ndim == 2:
        # New behavior for 2D arrays (n_cells x n_timepoints)
        # Compute variance explained for each cell (row)
        ss_total = n.sum((true - n.mean(true, axis=1, keepdims=True)) ** 2, axis=1)
        ss_residual = n.sum((true - pred) ** 2, axis=1)

        # Handle case where ss_total is zero (constant true values)
        var_exp = n.zeros(true.shape[0])
        nonzero_mask = ss_total > 0
        var_exp[nonzero_mask] = 1 - (ss_residual[nonzero_mask] / ss_total[nonzero_mask])

        return var_exp

    else:
        raise ValueError("Input arrays must be 1D or 2D.")


def summarize_dist(vals, factor=None):
    vals = n.asarray(vals)
    vals = vals[~n.isnan(vals)]
    if vals.ndim != 1:
        raise ValueError("Input must be a 1D array.")
    if len(vals) == 0:
        print("#### EMPTY ARRAY ####")
        return {}

    # Scale the values
    max_abs = n.max(n.abs(vals))
    if factor is None:
        factor = 10 ** n.floor(n.log10(max_abs / 100)) if max_abs > 100 else 10 ** n.ceil(n.log10(10 / max_abs)) if max_abs < 10 else 1
    scaled = vals * factor
    pcts = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    # Compute stats
    stats = {
        "mean": n.mean(scaled),
        "median": n.median(scaled),
        "std": n.std(scaled),
        "min": n.min(scaled),
        "max": n.max(scaled),
        "range": n.max(scaled) - n.min(scaled),
        "percentiles": {p: n.percentile(scaled, p) for p in pcts},
        "scaling_factor": factor,
    }

    def fmt(x):
        return f"{x:6.2f}"

    # Prepare output lines
    lines = [
        "#### DISTRIBUTION SUMMARY ####",
        f"Scaling factor: {factor:.1f}",
        f"Mean:     {fmt(stats['mean'])}" + f"  Median:   {fmt(stats['median'])}" + f"  Std Dev:  {fmt(stats['std'])}",
        f"Min:      {fmt(stats['min'])}" + f"  Max:      {fmt(stats['max'])}" + f"  Range:    {fmt(stats['range'])}",
        "Percentiles:",
    ]
    header = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    values = [stats["percentiles"][float(h)] for h in header]

    header_str = "  ".join(f"{fmt(h):>5}" for h in header)
    value_str = "  ".join(f"{fmt(v):>5}" for v in values)

    lines.append(header_str)
    lines.append(value_str)

    print("\n".join(lines))

    # Flatten percentiles into the dict
    stats.update({f"percentile_{p}": v for p, v in stats["percentiles"].items()})
    del stats["percentiles"]
    return stats


def compute_lagcorrs(data, n_rows=50, tidxs=None):
    """
    compute the lagged covariance matrix using FFT method.
    lagcorrs[i,j,t] = E[data[i,tau] * data[j, tau + t]]

    Args:
        data (ndarray): nvar,nt
        n_rows (int, optional): clip and use only the first n_rows rows of data. Defaults to 50.
        tidxs (int, optional): clip the timepoints of the returned array. Defaults to None.

    Returns:
        lagcorrs: lagged covariance tensor
    """

    nvar, nt = data.shape
    # compute the fft of the cross-corr tensor in fourier space
    f_a_fft = fft.fft(data[:n_rows,], axis=-1, n=2 * nt)
    f_a_fft_conj = n.conj(f_a_fft)

    cross_spec_tensor = n.einsum("ik,jk->ijk", f_a_fft, f_a_fft_conj)
    lagged_corr_tensor = fft.ifft(cross_spec_tensor)[:, :].real

    maxs = lagged_corr_tensor[n.diag_indices(n_rows)][:, 0]
    norms = 0.5 * (maxs[n.newaxis] + maxs[:, n.newaxis])
    norm_lagged_corr_tensor = lagged_corr_tensor / norms[:, :, n.newaxis]
    if tidxs is None:
        tidxs = norm_lagged_corr_tensor.shape[-1] // 2
    lagcorrs = norm_lagged_corr_tensor[:, :, :tidxs]

    return lagcorrs


def autocorrelation_fft(
    data,
    dt=1.0,
    max_lag=None,
    t_window=None,
    demean=True,
    unbiased=True,
    normalize=True,
    estimate_tau=False,
    tau_exclude_zero_lag=True,
    tau_min_points=5,
):
    """
    Compute the autocorrelation function (ACF) for each timeseries using an FFT-based method.

    Args:
        data (ndarray): Array of shape (n_cells, n_time) or (n_time,). Each row is a timeseries.
        dt (float, optional): Sampling interval. Used to generate the time axis. Defaults to 1.0.
    max_lag (int, optional): Maximum lag (in samples) on either side of zero to return.
                 If None, returns the full range (nt-1) on each side.
    t_window (float, optional): If provided, trims the ACF to lags within +/- t_window seconds.
                    Overrides max_lag if both are given.
        demean (bool, optional): Subtract the per-series mean before computing ACF. Defaults to True.
        unbiased (bool, optional): If True, divide by (N - lag) to remove finite-sample bias.
                                   If False, divide by N (biased estimator). Defaults to True.
        normalize (bool, optional): If True, divide each ACF by its zero-lag value so that ACF[:,0] = 1.
                                    Defaults to True.

    Extra options (time constant fit):
        To also estimate a time constant per series by fitting an exponential decay to the
        non‑negative lags, call with estimate_tau=True. See return values below.

    Returns:
        acf (ndarray): Shape (n_cells, Lc) array of autocorrelations for symmetric lags
                       from negative to positive centered at 0.
        t (ndarray): Shape (Lc,) array of time lags (seconds) corresponding to acf columns, sorted ascending
                     from negative to positive and containing 0.
        tau (ndarray, optional): If estimate_tau=True, shape (n_cells,) array of fitted time constants (seconds).

    Notes:
        - Uses zero-padding and the Wiener–Khinchin theorem: IFFT(|FFT(x)|^2) to get the linear/aperiodic ACF.
        - Returns only non-negative lags (0..L-1). For symmetric lags, mirror as needed outside this function.
        - If a series is constant (zero variance), the normalized ACF will be zeros except ACF[:,0]=1 if normalize=True.
    """
    x = n.asarray(data)
    if x.ndim == 1:
        x = x[n.newaxis, :]
    if x.ndim != 2:
        raise ValueError("data must be 1D (nt,) or 2D (nc, nt)")

    nc, nt = x.shape

    # Optional de-meaning per series
    if demean:
        x = x - x.mean(axis=1, keepdims=True)

    # Zero-pad to avoid circular correlation: length >= 2*nt-1 gives linear ACF for |lag|<=nt-1
    nfft = 2 * nt
    X = fft.fft(x, n=nfft, axis=-1)
    S = X * n.conj(X)
    acf_full = fft.ifft(S, axis=-1).real  # shape (nc, 2*nt)
    # Assemble full linear ACF for lags k = -(nt-1)..(nt-1)
    # Non-negative lags are at indices [0:nt); negative lags are wrapped at the end
    neg = acf_full[:, nfft - (nt - 1) : nfft]
    pos = acf_full[:, :nt]
    acf_lin = n.concatenate([neg, pos], axis=1)  # shape (nc, 2*nt-1)

    # Build lag index and counts for unbiased normalization
    lags = n.arange(-(nt - 1), nt)

    # Normalization across lags
    if unbiased:
        counts = (nt - n.abs(lags))[n.newaxis, :]
        counts = n.maximum(counts, 1)
        acf = acf_lin / counts
    else:
        acf = acf_lin / nt

    if normalize:
        # Normalize so that acf at lag 0 equals 1 when variance > 0
        z = acf[:, lags == 0]
        safe = n.where(z == 0.0, 1.0, z)
        acf = acf / safe
        acf[:, lags == 0] = 1.0

    # Time axis in seconds, centered at 0
    t = lags.astype(float) * float(dt)

    # Optional trimming by time window or sample max_lag
    if t_window is not None:
        mask = n.abs(t) <= float(t_window) + 1e-12
        acf = acf[:, mask]
        t = t[mask]
    elif max_lag is not None:
        M = int(max_lag)
        M = max(0, min(M, nt - 1))
        mask = (lags >= -M) & (lags <= M)
        acf = acf[:, mask]
        t = t[mask]

    # Optionally estimate a time constant per series by fitting y = offset + amp * exp(-t/tau)
    if estimate_tau:
        # Work on non-negative lags only (decay side)
        pos_mask = t >= 0
        if tau_exclude_zero_lag:
            pos_mask = n.logical_and(pos_mask, t > 0)
        tt = t[pos_mask]
        taus = n.full((nc,), n.nan, dtype=float)

        if tt.size >= tau_min_points:
            # Bounds: tau>0; amp, offset unconstrained unless normalize
            if normalize:
                bounds = ((1e-12, -n.inf, -1.0), (n.inf, n.inf, 1.0))
            else:
                bounds = ((1e-12, -n.inf, -n.inf), (n.inf, n.inf, n.inf))

            def model(t_, tau, amp, offset):
                return offset + amp * n.exp(-t_ / tau)

            for i in range(nc):
                yy = acf[i, pos_mask]
                # Heuristic initial guesses
                # offset0: median of tail
                tail_n = max(3, int(n.ceil(0.1 * yy.size)))
                offset0 = n.median(yy[-tail_n:])
                amp0 = yy[0] - offset0 if yy.size > 0 else 1.0
                if not n.isfinite(amp0) or abs(amp0) < 1e-8:
                    amp0 = 1.0
                # tau0 via log-linear ignoring offset (where yy > offset0)
                msk = yy > offset0 + 1e-6
                tau0 = None
                if n.count_nonzero(msk) >= 2:
                    tsel = tt[msk]
                    ysel = yy[msk] - offset0
                    try:
                        slope, _ = n.polyfit(tsel, n.log(ysel), 1)
                        if slope < 0:
                            tau0 = -1.0 / slope
                    except Exception:
                        tau0 = None
                if tau0 is None or not n.isfinite(tau0) or tau0 <= 0:
                    tau0 = max(1e-3, 0.2 * (tt[-1] - tt[0]))

                p0 = (float(tau0), float(amp0), float(offset0))
                try:
                    popt, _ = curve_fit(model, tt, yy, p0=p0, bounds=bounds, maxfev=10000)
                    taus[i] = float(popt[0])
                except Exception:
                    taus[i] = n.nan

        # Return with taus
        return acf, t, taus

    return acf, t


def correlate(array, vector):
    """
    compute the correlation of each element in array with given vector

    Args:
        array (ndarray): n_cells, nt
        vector (ndarray): nt

    Returns:
        correlation of each row of the array with the vector, size n_cells
    """
    vector = n.squeeze(vector)
    squeeze_output = False
    if len(array.shape) == 1:
        array = array[n.newaxis, :]
        squeeze_output = True
    array = array - array.mean(axis=1, keepdims=True)
    vector = vector - vector.mean(axis=0)
    cov = (array * vector[n.newaxis]).sum(axis=1)
    var_arr = n.sqrt((array**2).sum(axis=1))
    var_vec = n.sqrt((vector**2).sum(axis=0))
    # print(vector.shape)
    # print(cov.shape, var_arr.shape, var_vec.shape)

    out = cov / (var_arr * var_vec)
    if squeeze_output:
        out = float(n.squeeze(out))

    return out


def correlate_vectors(arr1, arr2):
    """
    compute the correlation coefficient between two arrays of vectors
    each array contains n_item vectors of length n_dim, and the output is n_item long

    Args:
        arr1 (ndarray): n_items, n_dim
        arr2 (ndarray): n_items, n_dim
    """
    arr1 = arr1 - arr1.mean(axis=1, keepdims=True)
    arr2 = arr2 - arr2.mean(axis=1, keepdims=True)

    cov = (arr1 * arr2).sum(axis=1)
    var1 = n.sqrt((arr1**2).sum(axis=1))
    var2 = n.sqrt((arr2**2).sum(axis=1))

    corr = cov / (var1 * var2)
    return corr


def cov_mat(arr, nan_diag=False, dtype=n.float32, eps=1e-6,):
    """
    compute covariance matrix of a data matrix. Similar to n.cov

    Args:
        arr (ndarray): n_cells, n_timepoints

    Returns:
        cov: n_cells, n_cells
    """
    arr = arr.astype(dtype)
    arr = arr - arr.mean(axis=1, keepdims=True)

    cov = arr @ arr.T / (arr.shape[1] - 1)

    if nan_diag:
        n.fill_diagonal(cov, n.nan)
    return cov


def corr_mat(arr, nan_diag=False, dtype=n.float32, eps=1e-6):
    """
    compute correlation matrix of a data matrix. Similar to n.corrcoef

    Args:
        arr (ndarray): n_cells, n_timepoints

    Returns:
        corr: n_cells, n_cells
    """
    arr = arr.astype(dtype)
    arr = arr - arr.mean(axis=1, keepdims=True)

    cov = arr @ arr.T

    var = (arr**2).sum(axis=1)

    corr = cov / (n.sqrt(var[:, n.newaxis] @ var[n.newaxis]) + eps)

    if nan_diag:
        n.fill_diagonal(corr, n.nan)
    return corr


def cv_cov_mat(arr0, arr1, dtype=n.float32):
    """
    compute cross-validated covariance matrix to stimulus responses

    Args:
        arr0 (ndarray): n_cells_A, n_trials - first half
        arr1 (ndarray): n_cells_B, n_trials - second half

    Returns:
        cv_cov: cross-validated covariance
    """
    arr0 = arr0.astype(dtype)
    arr1 = arr1.astype(dtype)

    arr0 = arr0 - arr0.mean(axis=1, keepdims=True)
    arr1 = arr1 - arr1.mean(axis=1, keepdims=True)

    cov = arr0 @ arr1.T

    return cov


def cv_corr_mat(arr0, arr1, dtype=n.float32):
    """
    compute cross-validated corr matrix to stimulus responses

    Args:
        arr0 (ndarray): n_cells, n_trials - first half
        arr1 (ndarray): n_cells, n_trials - second half

    Returns:
        cv_corr: cross-validated corr
    """
    arr0 = arr0.astype(dtype)
    arr1 = arr1.astype(dtype)

    arr0 = arr0 - arr0.mean(axis=1, keepdims=True)
    arr1 = arr1 - arr1.mean(axis=1, keepdims=True)

    cov = arr0 @ arr1.T

    var0 = (arr0**2).sum(axis=1)
    var1 = (arr1**2).sum(axis=1)

    cv_corr = cov / n.sqrt(var0[:, n.newaxis] @ var1[n.newaxis])
    return cv_corr


def zscore(x, nax=0, m=None, std=None, return_params=False, auto_reshape=True, undo=False):
    """zscore a given axis of an n-dimensional array based on given or computed parameters.
       If you have an array of shape x,y,z and nax=1, the activity will be average over all
       x and z, so the mean and std will have shape 1,y,1.

    Args:
        x (ndarray): ndim array
        nax (list or int, optional): Axes to *not* average over, typically the neuron axis. Defaults to 0.
        m (ndarray, optional): mean. Defaults to computing from x.
        std (ndarray, optional): std. Defaults to computing from x.
        return_params (bool, optional): Return m and std in a tuple. Defaults to False.
        auto_reshape (bool, optional): Automatically fix the shapes of m and std. Defaults to True.
    """
    # x = n.squeeze(x)
    ndim = len(x.shape)
    if ndim == 1:
        nax = [-1]
    nax = n.array(nax).astype(int)
    axes_to_reduce = n.array([i if i not in nax else n.nan for i in range(ndim)])
    axes_to_reduce = tuple(n.array(axes_to_reduce)[~n.isnan(axes_to_reduce)].astype(int))
    if m is None:
        m = x.mean(axis=axes_to_reduce, keepdims=True)
    if std is None:
        std = x.std(axis=axes_to_reduce, keepdims=True)

    std += 1e-6

    if auto_reshape:
        param_shape = n.ones(ndim).astype(int)
        param_shape[nax] = n.array(x.shape)[nax]

        # if they are a scalar don't reshape
        if n.array(m).size > 1:
            m = m.reshape(*param_shape)
        if n.array(std).size > 1:
            std = std.reshape(*param_shape)

    if not undo:
        xz = (x - m) / std
    else:
        xz = (x * std) + m
    if return_params:
        return xz, (m, std)
    else:
        return xz


def fill_nans(signal, method="linear", axis=0):
    """
    Fill NaN values in a signal using specified method.

    Args:
        signal (ndarray): Input signal with NaN values.
        method (str, optional): Method to fill NaNs. Options are 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'.
                                Defaults to 'linear'.
        axis (int, optional): Axis along which to fill NaNs. Defaults to 0.

    Returns:
        ndarray: Signal with NaN values filled.
    """
    from scipy.interpolate import interp1d

    if n.isnan(signal).any():
        # Create a mask for non-NaN values
        mask = ~n.isnan(signal)
        # Create an interpolating function
        interp_func = interp1d(
            n.arange(signal.shape[axis])[mask],
            signal[mask],
            kind=method,
            axis=axis,
            bounds_error=False,
            fill_value="extrapolate",
        )
        # Apply the interpolation function
        filled_signal = interp_func(n.arange(signal.shape[axis]))
        return filled_signal
    else:
        return signal


def median_filter1d(signal, width=3, axis=0):
    """
    apply a simple median filter to a 1d signal

    Args:
        signal (ndarray): ndim ndarray
        width (int, optional): Width of filter. Defaults to 3.
        axis (int, optional): axis to apply filter on. Defaults to 0.

    Returns:
        signal: same shape as input, filtered
    """
    if width == 0:
        return signal
    from scipy.ndimage import median_filter

    out = median_filter(signal, size=width, axis=axis)
    return out


def area_within_points(xs, ys):
    # using the shoelace formula https://en.wikipedia.org/wiki/Shoelace_formula
    # xs, ys: (n_shapes, n_points)
    xs = n.asarray(xs)
    ys = n.asarray(ys)
    if xs.shape != ys.shape:
        raise ValueError("xs and ys must have the same shape")
    # roll by -1 along the last axis for the polygon formula
    area = 0.5 * n.abs(n.sum(xs * n.roll(ys, -1, axis=1) - ys * n.roll(xs, -1, axis=1), axis=1))
    return area


def filt(signal, width=3, axis=0, mode="gaussian"):
    """
    apply a simple filter to a 1d signal

    Args:
        signal (ndarray): ndim ndarray
        width (int, optional): Width of filter. Defaults to 3.
        axis (int, optional): axis to apply filter on. Defaults to 0.
        mode (str, optional): Type of filter. 'gaussian' or 'uniform'

    Returns:
        signal: same shape as input, filtered
    """
    if width == 0:
        return signal

    if mode == "gaussian":
        out = gaussian_filter1d(signal, sigma=width, axis=axis)
    elif mode == "uniform":
        # print(width)
        out = uniform_filter1d(signal, size=int(n.round(width)), axis=axis)
    else:
        assert False, "mode not implemented"
    return out


def compute_signal_related_variance_ragged(resp, mean_center=True):
    """
    Compute the fraction of signal-related variance (and SNR) for each cell
    given ragged repeats per stimulus.

    Args:
        resp (list of arrays): A list of length n_stimuli. Each element resp[i]
                               is an array of shape (R_i, n_cells), where R_i
                               is the number of repeats for stimulus i (can vary
                               across stimuli).
        mean_center (bool): Whether to subtract the grand mean of stimulus means
                            before computing signal variance.

    Returns:
        fraction_of_stimulus_variance (ndarray): shape (n_cells,), values in [0, 1].
        stim_to_noise_ratio (ndarray): shape (n_cells,).
    """
    n_stim = len(resp)
    # Ensure all stimuli have 2D array (R_i, n_cells) and consistent n_cells
    processed = []
    n_cells = None
    for i, arr in enumerate(resp):
        arr = n.asarray(arr)  # shape (R_i, n_cells)
        if arr.ndim != 2:
            raise ValueError(f"Stimulus {i} data must be 2D, got shape {arr.shape}")
        Ri, nc = arr.shape
        if n_cells is None:
            n_cells = nc
        elif nc != n_cells:
            raise ValueError(f"All stimuli must have the same number of cells; stimulus {i} has {nc}, expected {n_cells}")
        processed.append(arr)

    # Compute per-stimulus means (mu_i) and per-stimulus noise sums
    # mu_mat will be shape (n_stim, n_cells)
    mu_mat = n.zeros((n_stim, n_cells), dtype=float)
    # noise_accum[i] = (1/R_i) * sum_r (x_{i,r} - mu_i)^2 for each cell
    noise_accum = n.zeros((n_stim, n_cells), dtype=float)

    for i, arr in enumerate(processed):
        Ri = arr.shape[0]
        # mean across repeats for each cell
        mu_i = arr.mean(axis=0)  # shape (n_cells,)
        mu_mat[i, :] = mu_i

        # noise variance for this stimulus (per cell)
        residuals = arr - mu_i  # shape (R_i, n_cells)
        noise_accum[i, :] = (residuals**2).sum(axis=0) / Ri  # (1/R_i)*sum_r(...)

    # Grand mean across stimuli for each cell
    mu_bar = mu_mat.mean(axis=0)  # shape (n_cells,)

    # Compute signal variance: Var_i( mu_i )
    if mean_center:
        mu_centered = mu_mat - mu_bar  # shape (n_stim, n_cells)
        signal_var = (mu_centered**2).sum(axis=0) / n_stim  # (1/n_stim)*sum_i (mu_i - mu_bar)^2
    else:
        signal_var = (mu_mat**2).sum(axis=0) / n_stim  # (1/n_stim)*sum_i (mu_i)^2

    # Compute noise variance: average of noise_accum across stimuli
    noise_var = noise_accum.mean(axis=0)  # (1/n_stim) * sum_i [ (1/R_i)*sum_r (x - mu_i)^2 ]

    total_var = signal_var + noise_var

    # Compute fraction and SNR
    fraction_of_stimulus_variance = n.zeros_like(signal_var)
    stim_to_noise_ratio = n.zeros_like(signal_var)

    # Avoid division by zero
    nonzero_mask = total_var > 0
    fraction_of_stimulus_variance[nonzero_mask] = signal_var[nonzero_mask] / total_var[nonzero_mask]
    # For cells with zero total variance, fraction remains zero

    nonzero_noise = noise_var > 0
    stim_to_noise_ratio[nonzero_noise] = signal_var[nonzero_noise] / noise_var[nonzero_noise]
    # For cells with zero noise variance, SNR is set to zero by default

    return fraction_of_stimulus_variance, stim_to_noise_ratio


def compute_signal_related_variance_ragged_per_stim(resp, mean_center=True):
    """
    Compute the fraction of signal-related variance (and SNR) for each cell
    given ragged repeats per stimulus, and also per-stimulus fraction of
    stimulus-related variance for each neuron.

    Args:
        resp (list of arrays): A list of length n_stimuli. Each element resp[i]
                               is an array of shape (R_i, n_cells), where R_i
                               is the number of repeats for stimulus i (can vary
                               across stimuli).
        mean_center (bool): Whether to subtract the grand mean of stimulus means
                            before computing signal variance.

    Returns:
        fraction_of_stimulus_variance (ndarray): shape (n_cells,), values in [0, 1].
        stim_to_noise_ratio (ndarray): shape (n_cells,).
        fraction_per_stim (ndarray): shape (n_stimuli, n_cells), where each element
                                     is the fraction of variance for that stimulus
                                     and cell attributed to the “signal” (the squared
                                     deviation of the stimulus mean from the grand mean)
                                     vs the total (signal + noise for that stimulus).
    """
    n_stim = len(resp)
    # Ensure all stimuli have 2D array (R_i, n_cells) and consistent n_cells
    processed = []
    n_cells = None
    for i, arr in enumerate(resp):
        arr = n.asarray(arr)  # shape (R_i, n_cells)
        if arr.ndim != 2:
            raise ValueError(f"Stimulus {i} data must be 2D, got shape {arr.shape}")
        Ri, nc = arr.shape
        if n_cells is None:
            n_cells = nc
        elif nc != n_cells:
            raise ValueError(f"All stimuli must have the same number of cells; stimulus {i} has {nc}, expected {n_cells}")
        processed.append(arr)

    # Compute per-stimulus means (mu_i) and per-stimulus noise variances
    mu_mat = n.zeros((n_stim, n_cells), dtype=float)
    noise_accum = n.zeros((n_stim, n_cells), dtype=float)

    for i, arr in enumerate(processed):
        Ri = arr.shape[0]
        # mean across repeats for each cell
        mu_i = arr.mean(axis=0)  # shape (n_cells,)
        mu_mat[i, :] = mu_i

        # noise variance for this stimulus (per cell)
        residuals = arr - mu_i  # shape (R_i, n_cells)
        noise_accum[i, :] = (residuals**2).sum(axis=0) / Ri  # (1/R_i)*sum_r(...)

    # Grand mean across stimuli for each cell
    mu_bar = mu_mat.mean(axis=0)  # shape (n_cells,)

    # Compute per-stimulus "signal" numerator: (mu_i - mu_bar)^2 (if mean_center),
    # else mu_i^2
    if mean_center:
        signal_per_stim = (mu_mat - mu_bar) ** 2  # shape: (n_stim, n_cells)
    else:
        signal_per_stim = mu_mat**2

    # Fraction per stimulus per cell: signal_per_stim / (signal_per_stim + noise_accum)
    fraction_per_stim = n.zeros_like(signal_per_stim)
    total_per_stim = signal_per_stim + noise_accum
    nonzero_mask_stim = total_per_stim > 0
    fraction_per_stim[nonzero_mask_stim] = signal_per_stim[nonzero_mask_stim] / total_per_stim[nonzero_mask_stim]
    # If total_per_stim == 0, fraction remains zero

    # Now aggregate across stimuli to get overall signal_var and noise_var per cell
    # signal_var = (1/n_stim) * sum_i signal_per_stim[i]
    signal_var = signal_per_stim.mean(axis=0)  # shape: (n_cells,)

    # noise_var = (1/n_stim) * sum_i noise_accum[i]
    noise_var = noise_accum.mean(axis=0)  # shape: (n_cells,)

    total_var = signal_var + noise_var

    fraction_of_stimulus_variance = n.zeros_like(signal_var)
    stim_to_noise_ratio = n.zeros_like(signal_var)

    nonzero_mask = total_var > 0
    fraction_of_stimulus_variance[nonzero_mask] = signal_var[nonzero_mask] / total_var[nonzero_mask]

    nonzero_noise = noise_var > 0
    stim_to_noise_ratio[nonzero_noise] = signal_var[nonzero_noise] / noise_var[nonzero_noise]

    # fraction_per_stim[i, c] represents the fraction of variance for stimulus i
    # and cell c that is attributed to the signal (i.e., the mean response of that
    # cell to the stimulus) versus the total variance (signal + noise).

    # Let:
    #   x_{i,r,c} = response of cell c to stimulus i on repeat r
    #   R_i       = number of repeats for stimulus i
    #   μ_{i,c}   = (1 / R_i) * sum_r x_{i,r,c}           # mean response to stim i
    #   μ̄_c      = (1 / n_stim) * sum_i μ_{i,c}          # grand mean across stimuli
    #   σ²_noise_{i,c} = (1 / R_i) * sum_r (x_{i,r,c} - μ_{i,c})²   # within-stimulus variance

    # Then:
    # If mean_center is True:
    #   signal_var_{i,c} = (μ_{i,c} - μ̄_c)²
    #   fraction_per_stim[i, c] = signal_var_{i,c} / (signal_var_{i,c} + σ²_noise_{i,c})
    #
    # If mean_center is False:
    #   signal_var_{i,c} = μ_{i,c}²
    #   fraction_per_stim[i, c] = μ_{i,c}² / (μ_{i,c}² + σ²_noise_{i,c})

    return fraction_of_stimulus_variance, stim_to_noise_ratio, fraction_per_stim


def compute_signal_related_variance(resp_a, resp_b, mean_center=True, exclude_nan=True):
    """
    compute the fraction of signal-related variance for each neuron,
    as per Stringer et al Nature 2019. Cross-validated by splitting
    responses into two halves. Note, this only is "correct" if resp_a
    and resp_b are *not* averages of many trials.

    Args:
        resp_a (ndarray): n_stimuli, n_cells
        resp_b (ndarray): n_stimuli, n_cells

    Returns:
        fraction_of_stimulus_variance: 0-1, 0 is non-stimulus-caring, 1 is only-stimulus-caring neurons
        stim_to_noise_ratio: ratio of the stim-related variance to all other variance
    """
    if len(resp_a.shape) > 2:
        # if the stimulus is multi-dimensional, flatten across all stimuli
        resp_a = resp_a.reshape(-1, resp_a.shape[-1])
        resp_b = resp_b.reshape(-1, resp_b.shape[-1])
    ns, nc = resp_a.shape
    if exclude_nan:
        # if any stimulus has nan responses, exclude it from both resp_a and b
        nan_mask = n.any(n.isnan(resp_a), axis=1) | n.any(n.isnan(resp_b), axis=1)
        resp_a = resp_a[~nan_mask]
        resp_b = resp_b[~nan_mask]
        if resp_a.shape[0] < 2:
            print("Not enough valid stimuli to compute signal-related variance.")
            return n.zeros(nc), n.zeros(nc)

    if mean_center:
        # mean-center the activity of each cell
        resp_a = resp_a - resp_a.mean(axis=0)
        resp_b = resp_b - resp_b.mean(axis=0)

    # compute the cross-trial stimulus covariance of each cell
    # dot-product each cell's (n_stim, ) vector from one half
    # with its own (n_stim, ) vector on the other half

    covariance = (resp_a * resp_b).sum(axis=0) / ns

    # compute the variance of each cell across both halves
    resp_a_variance = (resp_a**2).sum(axis=0) / ns
    resp_b_variance = (resp_b**2).sum(axis=0) / ns
    total_variance = (resp_a_variance + resp_b_variance) / 2

    # compute the fraction of the total variance that is
    # captured in the covariance
    fraction_of_stimulus_variance = covariance / total_variance

    # if you want, you can compute SNR as well:
    stim_to_noise_ratio = fraction_of_stimulus_variance / (1 - fraction_of_stimulus_variance)

    return fraction_of_stimulus_variance, stim_to_noise_ratio


def covariance_matrix(resp1, resp2, mean_subtract=False):
    # resp1, resp2: n_samples x n_features
    # returns: covmat: n_features x n_features
    ns, nf = resp1.shape
    if mean_subtract:
        resp1 -= resp1.mean(axis=0)
        resp2 -= resp2.mean(axis=0)
    covmat = resp1.T @ resp2 / (ns - 1)
    return covmat


def response_cov(x, y):
    # x: n_stimuli, n_features
    # y: n_stimuli, n_features
    return ((x - x.mean(axis=0)) * (y - y.mean(axis=0))).mean(axis=0)


def proj(resp, vecs, subtract_mean=False):
    # resp: n_stimuli x n_neurons
    # vecs: n_vecs x n_neurons
    # returns proj: n_stimuli x n_vecs
    if subtract_mean:
        respx = resp - resp.mean(axis=0)
    else:
        respx = resp
    return respx @ vecs.T


# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6758222/
# Ringach et al
def circular_variance(angles, responses):
    """
    Compute the preferred orientation and circular variance of cells
    as per Ringach et al. 2002

    Args:
        angles (ndarray): Angles used in experiment, in degrees
        responses (ndarray): (n_angles, n_cells) responses of all cells to all angles

    Returns:
        preferred_angles: 0-180 degrees, favorite angle of each cell
        circular_variance: 0 is very selective, 1 is not selective at all

    """
    # responses should be of shape n_angles, n_cells
    # angles is of shape n_angles IN DEGREES
    angles_radians = n.deg2rad(angles)[:, n.newaxis]

    numerator = (responses * n.exp(angles_radians * 2j)).sum(axis=0)
    denominator = responses.sum(axis=0)
    resultant = numerator / denominator

    circular_variance = 1 - n.abs(resultant)

    preferred_angles = n.rad2deg(n.angle(resultant))
    preferred_angles = preferred_angles / 2 + 90
    preferred_angles = n.mod(preferred_angles + 90, 180)

    return preferred_angles, circular_variance


def fix_overflow(data, n_bytes=4):
    exp = n_bytes * 8
    max = 2**exp
    thresh = 2 ** (exp - 1)

    data[data > thresh] -= max

    return data


def project(activity, vector):
    vector_norm = vector / n.linalg.norm(vector)
    activity_norm = n.linalg.norm(activity, axis=0)

    proj = (activity.T @ vector_norm) / activity_norm

    return proj


def dog_filt(vector, fwhm_1, fwhm_2, axis=-1):
    """
    difference of gaussians filter, as implemented by Nguyen et al 2023

    Args:
        vector (ndarray): timeseries to filter, ndim
        fwhm_1 (float): fwhm width of the first gaussian
        fwhm_2 (float): fwhm width of the second gaussian
        axis (int, optional): axis to filter over. Defaults to -1.

    Returns:
        filtered_vec: same size as input vector
    """
    sigma_1 = fwhm_1 / (2 * n.sqrt(2 * n.log(2)))
    sigma_2 = fwhm_2 / (2 * n.sqrt(2 * n.log(2)))

    vec1 = gaussian_filter1d(vector, sigma_1, axis=axis)
    vec2 = gaussian_filter1d(vector, sigma_2, axis=axis)

    return vec1 - vec2


def bin2d(im, bin_size):
    if type(bin_size) is tuple:
        im = bin1d(bin1d(im, bin_size[0], axis=0), bin_size[1], axis=1)
    elif bin_size is not None:
        im = bin1d(bin1d(im, bin_size, axis=0), bin_size, axis=1)
    return im


def bin1d(X, bin_size, axis=0):
    # From rastermap! https://github.com/MouseLand/rastermap/blob/main/rastermap/utils.py
    """mean bin over axis of data with bin bin_size"""
    if bin_size > 0:
        size = list(X.shape)
        Xb = X.swapaxes(0, axis)
        Xb = n.nanmean(
            Xb[: size[axis] // bin_size * bin_size].reshape((size[axis] // bin_size, bin_size, -1)),
            axis=1,
        )
        Xb = Xb.swapaxes(axis, 0)
        size[axis] = Xb.shape[axis]
        Xb = Xb.reshape(size)
        return Xb
    else:
        return X


def gaussian_rbf(scale, distance):
    return n.exp(-((distance / scale) ** 2))


def exp_kernel(scale, distance):
    return n.exp(-n.abs(distance) / scale)


def sample_generalized_normal(beta, size, loc=0, scale=1, seed=None):
    """
    Sample from a generalized normal distribution.
    beta = 1 is laplace
    beta = 2 is normal
    """
    rng = n.random.default_rng(seed)
    return gennorm.rvs(beta, loc=loc, scale=scale, size=size, random_state=rng)


def matern_kernel_nd(x, scale, nu=0.5, D=1, n=n):
    """
    Isotropic Matérn kernel k(r) where r = ||x|| in R^D.

    Parameters
    ----------
    x     : array_like, shape (..., D) or (...,)
            Spatial differences.  If D>1, pass vectors in the last axis.
    scale : positive float, length‐scale ℓ
    nu    : smoothness ν; ν=0.5 → exponential, ν→∞ → Gaussian
    D     : ambient spatial dimension (not used in k(r) itself, but kept for API)
    n     : numpy or torch module (must provide exp, sqrt, abs, exp, linalg.norm,
            and, for the general ν, either scipy.special.kv/γ or torch.special.kv/lgamma)
    """
    assert n is not None, "pass n=np or n=torch"
    x = n.asarray(x)
    # radial distance
    if x.ndim > 0 and x.shape[-1] == D:
        r = n.linalg.norm(x, axis=-1)
    else:
        r = n.abs(x)

    # ν = 0.5 → exponential(ℓ): exp(−r/ℓ)
    if nu == 0.5:
        return n.exp(-r / scale)
    # ν = ∞ → Gaussian(ℓ): exp(−½ (r/ℓ)²)
    if nu == n.inf:
        return n.exp(-0.5 * (r / scale) ** 2)

    # general ν:
    arg = n.sqrt(2 * nu) * r / scale

    # pick a γ(ν) and K_ν(arg) that work in both frameworks
    if _besselk is not None and _gammafunct is not None and n.__name__ == "numpy":
        gamma_nu = _gammafunct(nu)
        K_nu = _besselk(nu, arg)
    else:
        # torch: use torch.special.kv and torch.lgamma
        gamma_nu = n.exp(n.lgamma(nu))
        K_nu = n.special.kv(nu, arg)

    return (2 ** (1 - nu) / gamma_nu) * (arg**nu) * K_nu


def matern_spectral_nd(w, scale, nu=0.5, D=1, n=n):
    """
    Isotropic Matérn spectral density S(k) in D dims,
    assuming `w` is the angular frequency (2π·k).

    S(k) = coeff * (2ν/ℓ² + ‖w‖²)^−(ν + D/2),
    where coeff = 2·π^(D/2)·(2ν)^ν·Γ(ν + D/2) / [Γ(ν)·ℓ^(2ν)].
    """
    # print(scale)
    assert n is not None, "pass n=np or n=torch"
    w = n.asarray(w)
    if w.ndim > 0 and w.shape[-1] == D:
        knorm = n.linalg.norm(w, axis=-1)
    else:
        knorm = w

    # ν = 0.5: matches your 1D exp‐kernel case
    if nu == 0.5:
        return 2 * scale / (1 + (scale * knorm) ** 2)

    # ν = ∞: matches your 1D Gaussian case
    if nu == n.inf:
        return n.sqrt(n.pi) * scale * n.exp(-0.25 * (scale * knorm) ** 2)

    # general ν: build the coefficient so that when D=1 it becomes
    #    2 √π (2ν)^ν Γ(ν+½) / [Γ(ν) ℓ^(2ν)]
    # i.e. exactly your 1D version.
    if _gammafunct is not None and n.__name__ == "numpy":
        gamma_nu = _gammafunct(nu)
        gamma_nu_D2 = _gammafunct(nu + D / 2)
    else:
        gamma_nu = n.exp(n.lgamma(nu))
        gamma_nu_D2 = n.exp(n.lgamma(nu + D / 2))

    coeff = 2 * (n.pi) ** (D / 2) * (2 * nu) ** nu * gamma_nu_D2 / (gamma_nu * scale ** (2 * nu))
    return coeff * (2 * nu / scale**2 + knorm**2) ** (-(nu + D / 2))


def matern_kernel_f(omega, scale, ord=0.5, n=n):
    """
    Compute the Fourier transform (spectral density) of a 1D Matérn kernel.

    Parameters
    ----------
    omega : array_like
        Angular frequency (ω).
    scale : float
        Length-scale parameter (l) of the Matérn kernel.
    ord : float or 'inf', optional
        Smoothness parameter ν.
        - ν = 0.5 yields the exponential kernel.
        - ν = n.inf yields the Gaussian (RBF) kernel.
        - Other ν > 0 use the general Matérn form.

    Returns
    -------
    S : array_like
        Spectral density S(ω) of the Matérn kernel.
    """
    nu = ord
    omega = n.asarray(omega)

    # ν = 0.5: exponential kernel ↔ S(ω) = 2l / (1 + (l ω)^2)
    if nu == 0.5:
        return 2 * scale / (1 + (scale * omega) ** 2)

    # ν = ∞: Gaussian kernel K(x)=exp(-x^2/scale^2) ↔ S(ω)=√π·l·exp(-ω^2·scale^2/4)
    if nu == n.inf:
        return n.sqrt(n.pi) * scale * n.exp(-((scale * omega) ** 2) / 4)

    # General ν: S(ω) = [2 π^{1/2} Γ(ν+1/2) (2ν)^ν / (Γ(ν) l^{2ν})] · (2ν/l^2 + ω^2)^{-(ν+1/2)}
    coeff = (2 * n.sqrt(n.pi) * gammafunct(nu + 0.5) * (2 * nu) ** nu) / (gammafunct(nu) * scale ** (2 * nu))
    return coeff * (2 * nu / scale**2 + omega**2) ** (-(nu + 0.5))


def matern_kernel(x, scale, ord=0.5, n=n):
    if ord == 0.5:
        return exp_kernel(scale, x)
    elif ord == n.inf:
        return gaussian_rbf(n.sqrt(2) * scale, x)
    else:
        # General case using Bessel function
        nu = ord
        arg = n.sqrt(2 * nu) * n.abs(x) / scale
        return (2 ** (1 - nu) / gammafunct(nu)) * (arg**nu) * kv(nu, arg)


def d_matern_kernel_d_scale(x, scale, ord=0.5, n=n):
    """
    Derivative of matern_kernel(x, scale, ord) w.r.t. scale.

    Matches matern_kernel implementation above (note the sqrt(2)*scale
    passed to gaussian_rbf for ord == inf). The previous version
    over-shot the ν=inf (RBF) case by a factor sqrt(2); this fixes it.

    Returns array shaped like x.
    """
    x = n.asarray(x)
    ax = n.abs(x)

    if ord == 0.5:
        # Exponential kernel: k = exp(-|x|/scale)
        k = exp_kernel(scale, x)
        return k * (ax / (scale**2))

    if ord == n.inf:
        # matern_kernel calls gaussian_rbf(n.sqrt(2)*scale, x)
        # If gaussian_rbf(L, r) = exp(-(r**2)/(L**2)),
        # k = exp(-r^2 / (2 scale^2))
        # dk/dscale = k * (r^2 / scale^3)
        r2 = ax * ax
        k = gaussian_rbf(n.sqrt(2) * scale, x)
        return k * (r2 / (scale**3))

    # General ν > 0
    nu = float(ord)
    r = n.sqrt(2.0 * nu) * ax / scale  # dimensionless
    out = n.zeros_like(ax, dtype=float)
    nz = r > 0
    if nz.any():
        # k = 2^{1-ν}/Γ(ν) * r^ν K_ν(r)
        # Using identity: d/dr (r^ν K_ν(r)) = - r^ν K_{ν-1}(r)
        # dk/dℓ = (C/ℓ) * r^{ν+1} K_{ν-1}(r), where C = 2^{1-ν}/Γ(ν)
        from scipy.special import kv, gamma

        C = 2.0 ** (1.0 - nu) / gamma(nu)
        r_nz = r[nz]
        out[nz] = (C / scale) * (r_nz ** (nu + 1.0)) * kv(nu - 1.0, r_nz)
    # At r=0 derivative is 0 (k is flat in r there), so zeros stay.
    return out


def gaussian(x, sigma, amp=1, offset=0):
    return amp * n.exp(-((x / sigma) ** 2)) + offset


def sum_of_gaussians(x, sigma1, sigma2, amp1, amp2):
    """
    return f(x) = amp1 * e^(-(x/sigma1)^2) + amp2 * e^(-(x/sigma2)^2)
    """
    return amp1 * n.exp(-((x / sigma1) ** 2)) + amp2 * n.exp(-((x / sigma2) ** 2))


def exp_decay(x, sigma, amp=1, offset=0):
    return amp * n.exp(-x / sigma) + offset


def d_exp_decay_by_d_sigma(x, sigma, amp=1, offset=0):
    return amp * x * n.exp(-x / sigma) / (sigma**2)


# def d_exp_decay_by_d_sigma(x, sigma, amp=1, offset=0):
#     return exp_decay(x, sigma, amp=-amp / sigma, offset=offset)


def get_powerlaw(ss, trange):
    # code from github for stringer 2019 paper
    """fit exponent to variance curve"""
    logss = n.log(n.abs(ss))
    y = logss[trange][:, n.newaxis]
    trange += 1
    nt = trange.size
    x = n.concatenate((-n.log(trange)[:, n.newaxis], n.ones((nt, 1))), axis=1)
    w = 1.0 / trange.astype(n.float32)[:, n.newaxis]
    b = n.linalg.solve(x.T @ (x * w), (w * x).T @ y).flatten()

    allrange = n.arange(0, ss.size).astype(int) + 1
    x = n.concatenate((-n.log(allrange)[:, n.newaxis], n.ones((ss.size, 1))), axis=1)
    ypred = n.exp((x * b).sum(axis=1))
    alpha = b[0]
    return alpha, ypred, b[1]


def gauss(x, H, A, x0, sigma):
    """
    gaussian with mean x0, std sigma, amplitude A

    Args:
        x (ndarray): input
        A (float): amplitude
        x0 (float): mean
        sigma (float): stdev

    Returns:
        y: gaussian(x)
    """
    return H + A * n.exp(-((x - x0) ** 2) / (2 * sigma**2))


def gauss_fit(x, y, maxfev=1000, bounds=((-n.inf, -n.inf, -n.inf, 0), n.inf)):
    """
    fit gaussian adapted from https://gist.github.com/cpascual/a03d0d49ddd2c87d7e84b9f4ad2df466

    Args:
        x (ndarray): xvals
        y (ndarray): yvals

    Returns:
        popt: curve_fit optimal parameters
    """
    mean = sum(x * y) / sum(y)
    sigma = n.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gauss, x, y, p0=[min(y), max(y), mean, sigma], maxfev=maxfev, bounds=bounds)
    return popt


def fit_exp_decay(x, y, maxfev=1000, p0=None, bounds=((0, -n.inf, -n.inf), (n.inf, n.inf, n.inf))):
    """
    fit exponential decay to data

    Args:
        x (ndarray): xvals
        y (ndarray): yvals

    Returns:
        popt: curve_fit optimal parameters
    """
    # initial guess: amplitude = max(y), decay = 1, offset = min(y)
    if p0 is None:
        p0 = [n.ptp(x) * 0.2, max(y) - min(y), min(y)]
    # print(p0)
    popt, pcov = curve_fit(exp_decay, x, y, p0=p0, maxfev=maxfev, bounds=bounds)
    return popt


def compute_noise_level(F, frame_rate):
    # using method from Rupprecth et al 2021 Nature Neuroscience
    dF_F = n.diff(F, axis=1) / F.mean(axis=1, keepdims=True)
    noise_levels = n.nanmedian(n.abs(n.diff(dF_F, axis=-1)), axis=-1) / n.sqrt(frame_rate)
    return noise_levels


def gamma_pdf(x, shape, scale=1, loc=0):
    y = (x - loc) / scale
    num = (y ** (shape - 1)) * n.exp(-y)
    n.clip(num, a_min=0, a_max=None)

    denom = gammafunct(shape) * scale
    return num / denom


def zig_pdf_sing(x, zinf, shape, scale):
    if x == 0:
        return zinf
    elif x > 0:
        return gamma_pdf(x, shape, scale) * (1 - zinf)
    else:
        return 0


# def zig_cdf(x, zinf,shape,scale)


def zig_pdf(x, zinf, shape, scale, loc=0):
    # this is not really a pdf because density at 0 is infinite!
    x = n.array(x)
    gpdf = gamma_pdf(x, shape, scale, loc)
    gpdf[x != 0] = gpdf[x != 0] * (1 - zinf)
    gpdf[x == 0] = zinf
    return gpdf


def full_zig_pdf(zinf, shape, scale, loc=0):
    xs = n.linspace(0, scale * 10, 101)
    return xs, zig_pdf(xs, zinf, shape, scale)


def sample_from_cdfs(cdfs, xs, n_samples=10, seed=10):
    if seed is not None:
        n.random.seed(seed)
    samples = []
    for i in range(n_samples):
        # print("\nSAMPLE")
        rand_unif = n.random.uniform(0, 1, size=cdfs.shape[1:])
        # print(rand_unif)
        diffs = n.abs(cdfs - rand_unif)
        argmins = n.argmin(diffs, axis=0)
        # print(argmins)
        sample = xs[argmins]
        # print(sample)
        samples.append(sample)
    return n.array(samples)


def zig_cdfs(zinfs, gshapes, gscales, npts=101, vmax=10):
    xs = n.linspace(0, vmax, npts)
    zig_pdfs_m = n.zeros((npts, *zinfs.shape))
    for idx in range(npts):
        zig_pdfs_m[idx] = zig_pdf(xs[idx], zinfs, gshapes, gscales)

    zig_cdfs_m = n.zeros_like(zig_pdfs_m)
    zig_cdfs_m[0] = zinfs
    zig_cdfs_m[1:] = zig_cdfs_m[0] + n.cumsum(zig_pdfs_m[1:] / zig_pdfs_m[1:].sum(axis=0), axis=0) * (1 - zig_cdfs_m[0])
    return xs, zig_cdfs_m


def lowess_fit(xs, ys, frac=0.1, npts=10000, nxs=30, xpct=99, seed=2358, xvals=None, it=3):
    if npts is None or len(xs) < npts:
        npts = len(xs)
    else:
        n.random.seed(seed)
        rand_idxs = n.random.choice(n.arange(len(xs)), npts, replace=False)
        xs = xs[rand_idxs]
        ys = ys[rand_idxs]

    xpcts = (100 - xpct) / 2, (100 - xpct) / 2 + xpct
    if xvals is None:
        xvals = n.linspace(n.percentile(xs, xpcts[0]), n.percentile(xs, xpcts[1]), nxs)

    yvals = slw.lowess(ys, xs, frac=frac, xvals=xvals, it=it)
    return xvals, yvals


def lowess_fit_equal_sampling(xs, ys, frac=0.1, nxs=30, npts=10000, xpct=99.9, seed=2358):
    xpcts = (100 - xpct) / 2, (100 - xpct) / 2 + xpct
    xvals = n.linspace(
        n.percentile(xs, xpcts[0], method="closest_observation"),
        n.percentile(xs, xpcts[1], method="closest_observation"),
        nxs,
    )
    # print(xvals)
    rand_idxs = []
    n_samples_per_bin = []
    npts_per_bin = npts // (nxs - 1)
    if seed is not None:
        n.random.seed(seed)
    for i in range(len(xvals) - 1):
        b0, b1 = xvals[i], xvals[i + 1]
        # print(b0, b1, xs.max())
        bin_idxs = n.where(n.logical_and(xs > b0, xs <= b1))[0]
        # print(xs[bin_idxs])
        n_samples_bin = min(len(bin_idxs), npts_per_bin)
        if n_samples_bin > 0:
            rand_idxs += list(n.random.choice(bin_idxs, n_samples_bin, replace=False))
        n_samples_per_bin.append(n_samples_bin)

    yvals = slw.lowess(ys[rand_idxs], xs[rand_idxs], frac=frac, xvals=xvals)

    return xvals, yvals, n_samples_per_bin, (xs[rand_idxs], ys[rand_idxs])


def squared_exponential(dt, tau, sigma_n=1e-3):
    # squared exponential covariance function from GPFA
    sigma_f = n.sqrt(1 - sigma_n**2)
    dt = n.abs(dt)
    return sigma_f**2 * n.exp(-(dt**2) / (2 * tau**2)) + sigma_n**2 * (dt == 0)


def squared_exponential_f(w, tau, sigma_n=1e-3):
    # Fourier transform of squared_exponential
    sigma_f = n.sqrt(1 - sigma_n**2)
    return n.sqrt(2 * n.pi) * tau * squared_exponential(dt=w, tau=(1 / tau)) + sigma_n**2


def temporal_covariance_kernel(data, win_size=101, dt=1):
    assert len(data.shape) == 1
    assert win_size % 2 == 1
    data_len = len(data)
    data = data - data.mean()
    half_win = win_size // 2
    cov_kernel = n.zeros(win_size)
    for i in range(win_size):
        lag = i - half_win
        if lag >= 0:
            cov_kernel[i] = n.mean(data[lag:] * data[: data_len - lag])
        else:
            cov_kernel[i] = n.mean(data[: data_len + lag] * data[-lag:])

    # also return x-coordinates
    xs = n.arange(-half_win, half_win + 1) * dt

    return xs, cov_kernel


def spatial_covariance_kernel(data, coords, npts=51):
    # data is of shape n_points, n_samples
    # coords is of shape n_points, n_dim
    # compute the spatial covariance kernel
    # interpolate it to fill empty distances and return on an even grid
    assert len(data.shape) == 2
    assert len(coords.shape) == 2
    data = data - data.mean(axis=1, keepdims=True)
    n_points, n_samples = data.shape
    n_dim = coords.shape[1]
    assert n_points == coords.shape[0]
    distmat = n.sqrt(((coords[n.newaxis] - coords[:, n.newaxis]) ** 2).sum(axis=2))
    covmat = n.cov(data)
    flat_dist = distmat[n.tril_indices(n_points, k=-1)]
    flat_cov = covmat[n.tril_indices(n_points, k=-1)]

    # remove nans
    valid_idxs = ~n.isnan(flat_cov)
    flat_dist = flat_dist[valid_idxs]
    flat_cov = flat_cov[valid_idxs]
    # sort by distance
    sort_idxs = n.argsort(flat_dist)
    flat_dist = flat_dist[sort_idxs]
    flat_cov = flat_cov[sort_idxs]

    # first, get the unique distances
    unique_dists = n.unique(flat_dist)
    # then, get the mean covariance at each distance
    mean_covs = n.zeros_like(unique_dists)
    for i, dist in enumerate(unique_dists):
        mean_covs[i] = n.mean(flat_cov[flat_dist == dist])
    return unique_dists, mean_covs
    # # interpolate
    # interp_covs = n.interp(n.linspace(0, flat_dist.max(), npts), unique_dists, mean_covs)
    # return n.linspace(0, flat_dist.max(), npts), interp_covs


def eig_decomp_slow(W):
    # compute (and sort) positive eigenvalues and eigenvectors of W
    eigvals, eigvecs = n.linalg.eig(W)
    signs = n.sign(eigvals)
    eigvals *= signs
    eigvecs *= signs
    eigvecs = eigvecs.T[n.argsort(-eigvals)]
    eigvals = eigvals[n.argsort(-eigvals)]
    return eigvals, eigvecs


def eig_decomp(W, n_top=None, which="LM"):
    """
    Compute (and sort) eigenvalues/eigenvectors of W.
    If n_top is None (or >= size of W): full dense decomposition.
    Else: use sparse iterative methods to approximate the largest n_top eigenpairs.

    Parameters
    ----------
    W : (N, N) array
    n_top : int or None
        Number of leading eigenvalues/eigenvectors to return. If None, return all.
    which : str
        Passed to scipy.sparse.linalg.eigs when using the partial method (default 'LM').

    Returns
    -------
    eigvals : (M,) array (M = n_top or N), sorted descending (made positive by sign flip like original code)
    eigvecs : (M, N) array, rows correspond to eigenvectors aligned with eigvals.
    """
    N = W.shape[0]
    use_partial = n_top is not None and 0 < n_top < N
    if use_partial:
        try:
            symmetric = n.allclose(W, W.T.conj())
            if symmetric:
                # largest algebraic (good if expecting real symmetric with possibly negative values)
                eigvals, eigvecs = eigsh(W, k=n_top, which="LA")
            else:
                eigvals, eigvecs = eigs(W, k=n_top, which=which)
        except Exception:
            # fallback to full if sparse method fails
            eigvals, eigvecs = n.linalg.eig(W)
    else:
        eigvals, eigvecs = n.linalg.eig(W)

    # Ensure arrays are real if imaginary parts are negligible
    if n.iscomplexobj(eigvals) and n.allclose(eigvals.imag, 0, atol=1e-12):
        eigvals = eigvals.real
    if n.iscomplexobj(eigvecs):
        imag_ok = n.allclose(eigvecs.imag, 0, atol=1e-12)
        if imag_ok:
            eigvecs = eigvecs.real

    # Match original behavior: flip signs so eigenvalues become positive (magnitude) and apply to vectors
    signs = n.sign(eigvals.real)
    # Avoid zeros producing 0 sign (keep them 1 to not zero out eigenvectors)
    signs[signs == 0] = 1
    eigvals = eigvals * signs
    eigvecs = eigvecs * signs

    # Sort descending
    order = n.argsort(-eigvals.real)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Return eigenvectors as (M, N) like original (they transposed after sorting)
    eigvecs = eigvecs.T
    return eigvals, eigvecs


def power_iteration(W, n_iter=100, tol=1e-6, random_state=None, return_vector=False, normalize=True, fallback_eig=False):
    """Estimate the largest eigenvalue (spectral radius) of a square matrix using power iteration.

    Parameters
    ----------
    W : array_like (N, N)
        Square matrix. Can be dense ``numpy.ndarray`` or anything supporting ``@`` and ``.T``.
    n_iter : int
        Maximum number of iterations (default 100).
    tol : float
        Relative tolerance on successive Rayleigh quotient changes for early stopping.
    random_state : int | None
        Seed for reproducible initialization.
    return_vector : bool
        If True, also return the approximate leading eigenvector (unit norm).
    normalize : bool
        If True (default), return eigenvalue scaled to be non‑negative (flip sign with vector if needed).
    fallback_eig : bool
        If True, and the iteration fails to converge (NaNs or stagnation with zero norm), fall back
        to ``eig_decomp(W, n_top=1)``.

    Returns
    -------
    lam : float
        Estimated largest eigenvalue (by magnitude; sign adjusted if ``normalize``).
    vec : ndarray (N,) (optional)
        Approximate leading eigenvector if ``return_vector`` is True.

    Notes
    -----
    * For symmetric (Hermitian) matrices this converges linearly with rate |λ2/λ1|.
    * If the dominant eigenvalue is not unique in magnitude or the matrix is (near) defective,
      convergence may stall; you can increase ``n_iter`` or use ``eigsh`` / ``eigs`` instead.
    * This implementation uses the Rayleigh quotient for a refined eigenvalue estimate each step.
    """
    W = n.asarray(W)
    N = W.shape[0]
    if W.shape[0] != W.shape[1]:
        raise ValueError("power_iteration expects a square matrix")

    rng = n.random.default_rng(random_state)
    v = rng.normal(size=N)
    v /= n.linalg.norm(v)

    prev_lam = 0.0
    for _ in range(int(n_iter)):
        w = W @ v
        norm_w = n.linalg.norm(w)
        if norm_w == 0 or not n.isfinite(norm_w):
            if fallback_eig:
                lam_full, vec_full = eig_decomp(W, n_top=1)
                lam = float(lam_full[0])
                if return_vector:
                    return lam, vec_full[0]
                return lam
            return (0.0, v) if return_vector else 0.0
        v = w / norm_w
        lam = float(v @ (W @ v))  # Rayleigh quotient
        # Relative convergence check
        if prev_lam != 0 and abs(lam - prev_lam) <= tol * max(1.0, abs(lam)):
            break
        prev_lam = lam

    if normalize and lam < 0:
        lam = -lam
        v = -v

    if return_vector:
        return lam, v
    return lam


def sample_student_t(shape, df=n.inf, scale=1.0, seed=None):
    """Sample a matrix of Student-t random variables with (approx) unit variance.

    Parameters
    ----------
    shape : tuple
        Output shape (e.g. (N_CELL, ND_LATENT)).
    df : float
        Degrees of freedom. np.inf (or very large) gives Gaussian.
    scale : float
        Overall scale multiplier applied after variance normalization.
    seed : int | None
        Optional RNG seed for reproducibility.

    Returns
    -------
    ndarray
        Samples with (approx) unit variance when df > 2.
    """
    rng = n.random.default_rng(seed)
    if n.isinf(df) or df > 1e6:  # treat very large as Gaussian
        return rng.normal(0.0, scale, size=shape)
    # Standard normal numerator
    z = rng.normal(0.0, 1.0, size=shape)
    # Chi-square denominator: draw one per element (independent scaling)
    # Previously size=(shape[0],1) which introduced shared row-wise scaling.
    v = rng.chisquare(df, size=shape)
    t = z / n.sqrt(v / df)
    if df > 2:
        # Normalize to unit variance: Var[t] = df/(df-2)
        t *= n.sqrt((df - 2) / df)

    t *= scale / t.std()

    return t


def compare_groups_to_reference(binned_shfracs, self_shfracs, alpha=0.05, correction="bonferroni"):
    """
    Compare binned shared fractions to self shared fractions using statistical tests.

    Parameters:
    -----------
    binned_shfracs : list of arrays
        List where each element contains shared fractions for a distance bin
    self_shfracs : array
        Self-prediction shared fractions (baseline)
    alpha : float
        Significance level (default 0.05)
    correction : str
        Multiple comparisons correction ('bonferroni', 'holm', or None)

    Returns:
    --------
    dict : Results containing p-values, effect sizes, and corrected p-values
    """
    import numpy as np
    from scipy import stats

    n_bins = len(binned_shfracs)
    results = {"n_bins": n_bins, "welch_pvals": [], "mannwhitney_pvals": [], "cohen_d": [], "bin_sizes": [], "bin_means": [], "self_mean": n.mean(self_shfracs), "self_std": n.std(self_shfracs)}

    # Perform tests for each distance bin
    for i, bin_data in enumerate(binned_shfracs):
        if len(bin_data) == 0:
            results["welch_pvals"].append(n.nan)
            results["mannwhitney_pvals"].append(n.nan)
            results["cohen_d"].append(n.nan)
            results["bin_sizes"].append(0)
            results["bin_means"].append(n.nan)
            continue

        # Remove NaN values
        bin_clean = bin_data[~n.isnan(bin_data)]
        self_clean = self_shfracs[~n.isnan(self_shfracs)]

        if len(bin_clean) == 0:
            results["welch_pvals"].append(n.nan)
            results["mannwhitney_pvals"].append(n.nan)
            results["cohen_d"].append(n.nan)
            results["bin_sizes"].append(0)
            results["bin_means"].append(n.nan)
            continue

        # Welch's t-test (unequal variances)
        welch_stat, welch_p = stats.ttest_ind(bin_clean, self_clean, equal_var=False)

        # Mann-Whitney U test
        mw_stat, mw_p = stats.mannwhitneyu(bin_clean, self_clean, alternative="two-sided")

        # Cohen's d effect size
        pooled_std = n.sqrt(((len(bin_clean) - 1) * n.var(bin_clean, ddof=1) + (len(self_clean) - 1) * n.var(self_clean, ddof=1)) / (len(bin_clean) + len(self_clean) - 2))
        cohen_d = (n.mean(bin_clean) - n.mean(self_clean)) / pooled_std

        results["welch_pvals"].append(welch_p)
        results["mannwhitney_pvals"].append(mw_p)
        results["cohen_d"].append(cohen_d)
        results["bin_sizes"].append(len(bin_clean))
        results["bin_means"].append(n.mean(bin_clean))

    # Convert to arrays
    results["welch_pvals"] = n.array(results["welch_pvals"])
    results["mannwhitney_pvals"] = n.array(results["mannwhitney_pvals"])
    results["cohen_d"] = n.array(results["cohen_d"])

    # Apply multiple comparisons correction
    if correction == "bonferroni":
        results["welch_pvals_corrected"] = results["welch_pvals"] * n_bins
        results["mannwhitney_pvals_corrected"] = results["mannwhitney_pvals"] * n_bins
    elif correction == "holm":
        # Holm-Bonferroni correction
        welch_order = n.argsort(results["welch_pvals"])
        mw_order = n.argsort(results["mannwhitney_pvals"])

        welch_corrected = n.full_like(results["welch_pvals"], n.nan)
        mw_corrected = n.full_like(results["mannwhitney_pvals"], n.nan)

        for i, idx in enumerate(welch_order):
            if not n.isnan(results["welch_pvals"][idx]):
                welch_corrected[idx] = results["welch_pvals"][idx] * (n_bins - i)

        for i, idx in enumerate(mw_order):
            if not n.isnan(results["mannwhitney_pvals"][idx]):
                mw_corrected[idx] = results["mannwhitney_pvals"][idx] * (n_bins - i)

        results["welch_pvals_corrected"] = welch_corrected
        results["mannwhitney_pvals_corrected"] = mw_corrected
    else:
        results["welch_pvals_corrected"] = results["welch_pvals"]
        results["mannwhitney_pvals_corrected"] = results["mannwhitney_pvals"]

    # Clip corrected p-values to max of 1.0
    results["welch_pvals_corrected"] = n.clip(results["welch_pvals_corrected"], 0, 1)
    results["mannwhitney_pvals_corrected"] = n.clip(results["mannwhitney_pvals_corrected"], 0, 1)

    # Summary statistics
    results["significant_welch"] = results["welch_pvals_corrected"] < alpha
    results["significant_mw"] = results["mannwhitney_pvals_corrected"] < alpha

    return results


def sample_poisson(means, seed=None):
    """
    Sample from a Poisson distribution with given means.

    Args:
        means (ndarray): Array of means for the Poisson distribution.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        ndarray: Samples drawn from the Poisson distribution.
    """
    if seed is not None:
        n.random.seed(seed)
    shape = means.shape
    return n.random.poisson(means.flatten()).reshape(shape)


def generate_interpolation_matrix(sample_positions, grid_coords):
    # sample_positions is of shape (n_samples, n_dim)
    # grid_coords is of shape (n_grid, n_dim)
    # assume we have a data_grid sample (not provided here) of shape (n_grid, n_time)
    # this function outputs a matrix M of shape (n_samples, n_grid)
    # such that data_sampled = M @ data_grid
    # Implementation: bilinear interpolation on an axis-aligned regular grid.
    # We infer the grid layout (ny, nx) from unique x and y values in grid_coords,
    # then for each sample position (x,y) compute the four surrounding corners and weights.
    sp = n.asarray(sample_positions, dtype=float)
    gc = n.asarray(grid_coords, dtype=float)
    if sp.ndim != 2 or sp.shape[1] != 2:
        raise ValueError("sample_positions must be (n_samples, 2) [x, y]")
    if gc.ndim != 2 or gc.shape[1] != 2:
        raise ValueError("grid_coords must be (n_grid, 2) [x, y]")

    n_samples = sp.shape[0]
    n_grid = gc.shape[0]

    # Deduce axes
    xs = n.unique(gc[:, 0])
    ys = n.unique(gc[:, 1])
    nx = xs.size
    ny = ys.size
    if nx * ny != n_grid:
        raise ValueError("grid_coords do not form a full rectangular grid (nx*ny != n_grid)")

    # Spacing (assumed uniform)
    dx = xs[1] - xs[0] if nx > 1 else 1.0
    dy = ys[1] - ys[0] if ny > 1 else 1.0
    x0 = xs[0]
    y0 = ys[0]

    # Map (iy, ix) -> index into the provided flattened grid_coords ordering
    # Build an index map by locating each coordinate in the axes.
    idx_map = n.empty((ny, nx), dtype=n.int64)
    # Tolerance for matching; handles tiny FP differences
    atol = max(abs(dx), abs(dy)) * 1e-12 if (nx > 1 or ny > 1) else 1e-12
    for k in range(n_grid):
        xk, yk = gc[k, 0], gc[k, 1]
        ix = int(n.clip(n.rint((xk - x0) / dx), 0, nx - 1)) if nx > 1 else 0
        iy = int(n.clip(n.rint((yk - y0) / dy), 0, ny - 1)) if ny > 1 else 0
        # Verify closeness; if not, fall back to nearest by absolute difference
        if nx > 1 and not n.isclose(xk, xs[ix], atol=atol, rtol=0):
            ix = int(n.argmin(n.abs(xs - xk)))
        if ny > 1 and not n.isclose(yk, ys[iy], atol=atol, rtol=0):
            iy = int(n.argmin(n.abs(ys - yk)))
        idx_map[iy, ix] = k

    # Allocate dense interpolation matrix
    M = n.zeros((n_samples, n_grid), dtype=float)

    # Helper for degenerate axes (nx==1 or ny==1) -> linear/nearest interpolation
    for i in range(n_samples):
        x, y = sp[i, 0], sp[i, 1]

        if nx == 1 and ny == 1:
            # Single-point grid; all weight on that point
            M[i, idx_map[0, 0]] = 1.0
            continue

        if nx == 1:
            # 1D along y
            iyf = (y - y0) / dy if ny > 1 else 0.0
            iyf = float(n.clip(iyf, 0.0, ny - 1))
            iy0 = int(n.clip(n.floor(iyf), 0, max(ny - 2, 0)))
            iy1 = int(n.clip(iy0 + 1, 0, ny - 1))
            wy = iyf - iy0
            w0 = 1.0 - wy
            w1 = wy
            M[i, idx_map[iy0, 0]] += w0
            M[i, idx_map[iy1, 0]] += w1
            continue

        if ny == 1:
            # 1D along x
            ixf = (x - x0) / dx if nx > 1 else 0.0
            ixf = float(n.clip(ixf, 0.0, nx - 1))
            ix0 = int(n.clip(n.floor(ixf), 0, max(nx - 2, 0)))
            ix1 = int(n.clip(ix0 + 1, 0, nx - 1))
            wx = ixf - ix0
            w0 = 1.0 - wx
            w1 = wx
            M[i, idx_map[0, ix0]] += w0
            M[i, idx_map[0, ix1]] += w1
            continue

        # 2D bilinear case
        ixf = (x - x0) / dx
        iyf = (y - y0) / dy
        ixf = float(n.clip(ixf, 0.0, nx - 1))
        iyf = float(n.clip(iyf, 0.0, ny - 1))
        ix0 = int(n.clip(n.floor(ixf), 0, nx - 2))
        iy0 = int(n.clip(n.floor(iyf), 0, ny - 2))
        ix1 = ix0 + 1
        iy1 = iy0 + 1

        wx = ixf - ix0
        wy = iyf - iy0
        w00 = (1.0 - wy) * (1.0 - wx)
        w01 = (1.0 - wy) * wx
        w10 = wy * (1.0 - wx)
        w11 = wy * wx

        M[i, idx_map[iy0, ix0]] += w00
        M[i, idx_map[iy0, ix1]] += w01
        M[i, idx_map[iy1, ix0]] += w10
        M[i, idx_map[iy1, ix1]] += w11

    return M


def generate_sparse_interpolation_matrix(sample_positions, grid_coords):
    """Build a sparse CSR interpolation matrix using bilinear interpolation.

    Parameters
    ----------
    sample_positions : ndarray, shape (n_samples, 2)
        Target positions (x, y) in the same physical units as grid_coords.
    grid_coords : ndarray, shape (n_grid, 2)
        Grid coordinates (x, y) for a rectangular regular grid. The ordering
        must correspond to how grid data will be flattened (e.g., from
        FFTFieldModel.grid_coords(flat=True, order='C')).

    Returns
    -------
    scipy.sparse.csr_matrix
        An (n_samples, n_grid) matrix M such that data_sampled = M @ data_grid,
        where data_grid is a flattened grid vector (n_grid,).

    Notes
    -----
    - Assumes a regular, axis-aligned grid with uniform spacing along x and y.
    - Handles edge clamping and degenerate 1D cases (nx==1 or ny==1).
    - Requires SciPy; raises ImportError if not available.
    """
    try:
        from scipy import sparse as sp
    except Exception as e:  # pragma: no cover - optional dependency environment
        raise ImportError("generate_sparse_interpolation_matrix requires SciPy. Install with `pip install scipy`.") from e

    sp_in = n.asarray(sample_positions, dtype=float)
    gc = n.asarray(grid_coords, dtype=float)
    if sp_in.ndim != 2 or sp_in.shape[1] != 2:
        raise ValueError("sample_positions must be (n_samples, 2) [x, y]")
    if gc.ndim != 2 or gc.shape[1] != 2:
        raise ValueError("grid_coords must be (n_grid, 2) [x, y]")

    n_samples = sp_in.shape[0]
    n_grid = gc.shape[0]

    # Infer axes from unique coordinates
    xs = n.unique(gc[:, 0])
    ys = n.unique(gc[:, 1])
    nx = xs.size
    ny = ys.size
    if nx * ny != n_grid:
        raise ValueError("grid_coords do not form a full rectangular grid (nx*ny != n_grid)")

    dx = xs[1] - xs[0] if nx > 1 else 1.0
    dy = ys[1] - ys[0] if ny > 1 else 1.0
    x0 = xs[0]
    y0 = ys[0]

    # Map (iy, ix) -> provided flat index
    idx_map = n.empty((ny, nx), dtype=n.int64)
    atol = max(abs(dx), abs(dy)) * 1e-12 if (nx > 1 or ny > 1) else 1e-12
    for k in range(n_grid):
        xk, yk = gc[k, 0], gc[k, 1]
        ix = int(n.clip(n.rint((xk - x0) / dx), 0, nx - 1)) if nx > 1 else 0
        iy = int(n.clip(n.rint((yk - y0) / dy), 0, ny - 1)) if ny > 1 else 0
        if nx > 1 and not n.isclose(xk, xs[ix], atol=atol, rtol=0):
            ix = int(n.argmin(n.abs(xs - xk)))
        if ny > 1 and not n.isclose(yk, ys[iy], atol=atol, rtol=0):
            iy = int(n.argmin(n.abs(ys - yk)))
        idx_map[iy, ix] = k

    data = []
    rows = []
    cols = []

    # Build contributions per sample
    for i in range(n_samples):
        x, y = float(sp_in[i, 0]), float(sp_in[i, 1])

        if nx == 1 and ny == 1:
            rows.append(i)
            cols.append(int(idx_map[0, 0]))
            data.append(1.0)
            continue

        if nx == 1:
            iyf = (y - y0) / dy if ny > 1 else 0.0
            iyf = float(n.clip(iyf, 0.0, ny - 1))
            iy0 = int(n.clip(n.floor(iyf), 0, max(ny - 2, 0)))
            iy1 = int(n.clip(iy0 + 1, 0, ny - 1))
            wy = iyf - iy0
            w0 = 1.0 - wy
            w1 = wy
            if w0 != 0.0:
                rows.append(i)
                cols.append(int(idx_map[iy0, 0]))
                data.append(w0)
            if w1 != 0.0:
                rows.append(i)
                cols.append(int(idx_map[iy1, 0]))
                data.append(w1)
            continue

        if ny == 1:
            ixf = (x - x0) / dx if nx > 1 else 0.0
            ixf = float(n.clip(ixf, 0.0, nx - 1))
            ix0 = int(n.clip(n.floor(ixf), 0, max(nx - 2, 0)))
            ix1 = int(n.clip(ix0 + 1, 0, nx - 1))
            wx = ixf - ix0
            w0 = 1.0 - wx
            w1 = wx
            if w0 != 0.0:
                rows.append(i)
                cols.append(int(idx_map[0, ix0]))
                data.append(w0)
            if w1 != 0.0:
                rows.append(i)
                cols.append(int(idx_map[0, ix1]))
                data.append(w1)
            continue

        # 2D bilinear
        ixf = float(n.clip((x - x0) / dx, 0.0, nx - 1))
        iyf = float(n.clip((y - y0) / dy, 0.0, ny - 1))
        ix0 = int(n.clip(n.floor(ixf), 0, nx - 2))
        ix1 = ix0 + 1
        iy0 = int(n.clip(n.floor(iyf), 0, ny - 2))
        iy1 = iy0 + 1
        wx = ixf - ix0
        wy = iyf - iy0
        w00 = (1.0 - wy) * (1.0 - wx)
        w01 = (1.0 - wy) * wx
        w10 = wy * (1.0 - wx)
        w11 = wy * wx

        if w00 != 0.0:
            rows.append(i)
            cols.append(int(idx_map[iy0, ix0]))
            data.append(w00)
        if w01 != 0.0:
            rows.append(i)
            cols.append(int(idx_map[iy0, ix1]))
            data.append(w01)
        if w10 != 0.0:
            rows.append(i)
            cols.append(int(idx_map[iy1, ix0]))
            data.append(w10)
        if w11 != 0.0:
            rows.append(i)
            cols.append(int(idx_map[iy1, ix1]))
            data.append(w11)

    M = sp.csr_matrix((data, (rows, cols)), shape=(n_samples, n_grid))
    return M


def generate_powerlaw_noise(nd, nt, beta=1.0, dt=1.0, seed=None, normalize=True, remove_dc=True):
    """Generate independent 1/f^beta (power-law) colored noise processes.

    Parameters
    ----------
    nd : int
        Number of independent dimensions (rows) to generate.
    nt : int
        Number of time points per process.
    beta : float
        Exponent for the power spectral density (PSD ∝ 1/f^beta). Common:
          0 white, 1 pink, 2 brown (integrated white), etc.
    dt : float
        Sampling interval (only affects frequency axis scaling; set 1 if irrelevant).
    seed : int | None
        RNG seed for reproducibility.
    normalize : bool
        If True, each process is z-scored (zero mean, unit std) after synthesis.
    remove_dc : bool
        If True (default), zero out the DC component to avoid huge low‑freq energy for beta>0.

    Returns
    -------
    noise : (nd, nt) ndarray
        Each row an independent colored noise realization.

    Notes
    -----
    * Implements spectral shaping: FFT(white) * (1/f^{beta/2}) in amplitude → PSD 1/f^beta.
    * Uses Hermitian symmetry for real output (via real IFFT of explicit complex spectrum).
    * For nt even, handles Nyquist bin separately (kept real).
    """
    if nd <= 0 or nt <= 0:
        raise ValueError("nd and nt must be positive")
    rng = n.random.default_rng(seed)

    # Frequencies (cycles per unit time) from FFT convention
    freqs = n.fft.fftfreq(nt, d=dt)
    # Amplitude scaling so that |A(f)|^2 ∝ 1/f^beta → A(f) ∝ 1/f^{beta/2}
    amp = n.ones(nt)
    nonzero = freqs != 0
    if beta != 0:
        amp[nonzero] = 1.0 / (n.abs(freqs[nonzero]) ** (beta / 2.0))

    # Optionally remove DC (otherwise very large variance for beta>0)
    if remove_dc:
        amp[freqs == 0] = 0.0

    # Prepare complex spectrum: random phases, Gaussian amplitudes (white noise in time)
    # We'll build half + mirror to enforce real signal.
    noise = n.empty((nd, nt), dtype=float)

    # Indices for positive freqs (including zero & Nyquist if even)
    if nt % 2 == 0:
        pos_idxs = n.arange(0, nt // 2 + 1)
        nyquist = nt // 2
    else:
        pos_idxs = n.arange(0, (nt + 1) // 2)
        nyquist = None

    for i in range(nd):
        # Random complex spectrum
        spec = n.zeros(nt, dtype=complex)
        # Random phases for positive freqs excluding DC and Nyquist
        phases = rng.uniform(0, 2 * n.pi, size=pos_idxs.shape[0])
        # Standard normal magnitude (white) then scale by amp
        magnitudes = rng.normal(0.0, 1.0, size=pos_idxs.shape[0])
        real_part = magnitudes * n.cos(phases)
        imag_part = magnitudes * n.sin(phases)
        spec[pos_idxs] = (real_part + 1j * imag_part) * amp[pos_idxs]

        # Enforce real signal: mirror
        # Skip first (DC) and last (Nyquist) to avoid duplicating
        if nyquist is not None:
            neg_slice = spec[1:nyquist]
            spec[nyquist + 1 :] = n.conj(neg_slice[::-1])
            # DC and Nyquist must be purely real
            spec[0] = spec[0].real + 0j
            spec[nyquist] = spec[nyquist].real + 0j
        else:
            neg_slice = spec[1 : pos_idxs[-1]]
            spec[pos_idxs[-1] + 1 :] = n.conj(neg_slice[::-1])
            spec[0] = spec[0].real + 0j

        # IFFT to time domain
        series = n.fft.ifft(spec).real
        if normalize:
            m = series.mean()
            s = series.std()
            if s > 0:
                series = (series - m) / s
            else:
                series = series - m
        noise[i] = series

    return noise


def generate_ar1(
    nd,
    nt,
    rho=0.98,
    noise="gaussian",
    scale=1.0,
    df=3.0,
    seed=None,
    normalize=True,
):
    """Generate AR(1) processes x_t = rho * x_{t-1} + eps_t for nd dimensions.

    Parameters
    ----------
    nd : int
        Number of independent processes.
    nt : int
        Number of time points.
    rho : float or (nd,) array
        AR(1) coefficient. Values close to 1 produce smoother trajectories.
    noise : {"gaussian","laplace","student","t"}
        Innovation distribution type.
    scale : float
        Scale parameter of innovations.
    df : float
        Degrees of freedom for Student-t innovations (if used).
    seed : int | None
        RNG seed.
    normalize : bool
        If True, z-score each process at the end.

    Returns
    -------
    x : (nd, nt) ndarray
        AR(1) realizations.
    """
    if nd <= 0 or nt <= 0:
        raise ValueError("nd and nt must be positive")
    rng = n.random.default_rng(seed)

    # Broadcast rho per-dimension if a scalar
    rho_vec = n.asarray(rho) if hasattr(rho, "__len__") else n.full((nd,), float(rho))
    if rho_vec.shape != (nd,):
        rho_vec = n.broadcast_to(rho_vec, (nd,))

    # Draw innovations
    noise = (noise or "gaussian").lower()
    if noise in ("gaussian", "normal"):
        eps = rng.normal(0.0, scale, size=(nd, nt))
    elif noise in ("laplace", "double-exponential"):
        eps = rng.laplace(0.0, scale, size=(nd, nt))
    elif noise in ("student", "t"):
        eps = rng.standard_t(df, size=(nd, nt)) * scale
    else:
        raise ValueError(f"Unknown innovation noise type: {noise}")

    x = n.empty((nd, nt), dtype=float)
    # Start from zero; subsequent normalization removes transients
    x[:, 0] = eps[:, 0]
    for t in range(1, nt):
        x[:, t] = rho_vec * x[:, t - 1] + eps[:, t]

    if normalize:
        x = x - x.mean(axis=1, keepdims=True)
        s = x.std(axis=1, keepdims=True)
        s[s == 0] = 1.0
        x = x / s
    return x


def sample_symmetric_pareto(alpha, scale=1.0, loc=0.0, size=(), seed=None, method="fold", clip=None):
    """Sample from a symmetric Pareto (heavy-tailed) distribution.

    Construction
    ------------
    Let X >= scale follow a (shifted) standard Pareto with tail index ``alpha``:
        P(X > x) = (scale / x)^alpha for x >= scale.
    We create a *symmetric* variable Y about ``loc`` by taking a base Pareto
    magnitude M = X - scale >= 0 (so that M=0 at the mode) and assigning
    a random sign (50/50) then translating by ``loc``:
        Y = loc + S * M,  S ∈ {+1, -1}.

    Thus the PDF is proportional to (scale + |y-loc|)^(-(alpha+1)) for y ≠ loc.
    When loc=0 this matches two mirrored shifted Pareto tails joined at 0.

    Parameters
    ----------
    alpha : float
        Shape (tail index) > 0. Larger alpha → lighter tails.
    scale : float
        Scale >= 0 determining where the power-law tail begins (acts as shift before symmetry).
    loc : float
        Center of symmetry. loc=0 gives symmetry about zero.
    size : int | tuple | None
        Output shape. If empty tuple, return a single sample (scalar).
    seed : int | None
        RNG seed for reproducibility.
    method : str
        Reserved for future extensions (currently only 'fold').

    Returns
    -------
    samples : ndarray or float
        Samples with specified shape.

    Notes
    -----
    * The variance exists only if alpha > 2; the mean of |Y-loc| exists if alpha > 1.
    * The returned distribution has mode at ``loc`` and power-law tails.
    * If scale=0 this reduces to a pure symmetric Pareto tail: |Y-loc| ~ Pareto(alpha, 0).
    """
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    if scale < 0:
        raise ValueError("scale must be >= 0")

    rng = n.random.default_rng(seed)
    # Determine total number of samples
    if size == ():
        n_samples = 1
    else:
        n_samples = int(n.prod(size))

    # Sample standard uniform for inverse CDF of shifted Pareto (X >= scale)
    # Standard Pareto with xm = scale: X = scale * U^{-1/alpha}
    # We then shift by subtracting scale to make magnitude start at 0 (mode at 0)
    U = rng.uniform(0.0, 1.0, size=n_samples)
    X = scale * (U ** (-1.0 / alpha))  # >= scale
    M = X - scale  # magnitude >= 0, with power-law tail

    if clip is not None:
        M = n.clip(M, -float(clip), float(clip))
    # Random signs for symmetry
    signs = rng.choice((-1.0, 1.0), size=n_samples)
    Y = loc + signs * M

    if size == ():
        return float(Y[0])
    else:
        return Y.reshape(size)
