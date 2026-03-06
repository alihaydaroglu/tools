import numpy as n

# import autograd.numpy as n
from sklearn.decomposition import PCA
from scipy import linalg


from . import utils


def fit_pca(x, window=None, times=None, n_components=20, return_pca=False):
    """fit PCA to the given data. Optionally window the dataset based on array of times.

    Args:
        x (ndarray): n_samples x n_features
        window (tuple, optional): Tuple of two floats corresponding to the left and right bounds of the time window to fit PCA in. Defaults to None.
        times (ndarray, optional): Sample times, used to determine the edges of the window. Defaults to None.
        n_components (int, optional): Number of components to fit. Defaults to 20.
    Returns:
        pca.components_
        pca.mean_
    """
    start_idx, end_idx = get_start_end_idxs(window, times)
    data = x[start_idx:end_idx]
    if n_components is None:
        n_components = min(data.shape)
    elif min(data.shape) < n_components:
        n_components = min(data.shape)
    pca = PCA(n_components=n_components)
    pca.fit(data)
    if return_pca:
        return pca.components_, pca.mean_, pca
    return pca.components_, pca.mean_


def fit_and_expvars(x, window=None, times=None, n_components=20):
    """fit PCA and return the explained variance by each component

    Args:
        see fit_pca
    """
    pc = fit_pca(x, window, times, n_components)
    return pc, explained_vars(x, *pc)


def transform(x, components, mean=None):
    """project x onto the given components/dimension

    Args:
        x (ndarray): original data, n_samples x n_features
        components (ndarray): dimensions, n_dims x n_features
        mean (ndarray, optional): n_features x 1. Center x by these means if given. If none given, extracts the mean of each feature from x.

    Returns:
        ndarray: n_samples x n_dims
    """
    if mean is None:
        mean = n.nanmean(x, axis=0).reshape(1, -1)
    x = x - mean
    # print(x.shape)
    return x @ components.T


def untransform(x_p, components, mean=0):
    if mean is None:
        mean = 0
    return x_p @ components + mean


def reconstruct(x, components, mean=None):
    return untransform(transform(x, components, mean), components, mean)


def transform_and_expvars(
    x,
    components,
    mean=None,
    expvar_window=None,
    times=None,
    return_frac=True,
    return_sum=False,
):
    xt = transform(x, components, mean)
    var = explained_vars(x, components, mean, expvar_window, times, return_frac, return_sum)
    return xt, var


def explained_vars(x, components, mean=None, window=None, times=None, return_frac=True, return_sum=False):
    """calculate the fraction of variance of x explained by each component

    Args:
        x (ndarray): n_samples x n_features
        components (ndarray): n_dims x n_features
        mean (ndarray, optional):  Defaults to None.

    Returns:
        ndarray: n_dims, each float is the fraction of variance explained
    """
    if window is not None and times is not None:
        start_idx, end_idx = get_start_end_idxs(window, times)
        x = x[start_idx:end_idx]
    xvar = n.nanvar(x, axis=0)
    var_sum = xvar.sum()
    xt = transform(x, components, mean)
    xtvar = n.var(xt, axis=0)
    if return_frac:
        if return_sum:
            return (xtvar / var_sum).sum()
        else:
            return xtvar / var_sum

    else:
        return xtvar


def get_start_end_idxs(window, times):
    if window is not None and times is not None:
        start_idx = n.where(times > window[0])[0][0]
        end_idx = n.where(times > window[1])[0]
        if len(end_idx) > 0:
            end_idx = end_idx[0]
        else:
            end_idx = len(times) - 1
    else:
        if window is not None or times is not None:
            assert False, "Error"
        end_idx = None
        start_idx = 0
    return start_idx, end_idx


def shuff_cvpca(respmat, n_shuff=5, seed=10, reduction=n.mean, nhalf=None):
    if seed is not None:
        n.random.seed(seed)
    cvpcas = []
    nstim, nrep, nc = respmat.shape
    shuff_mat = n.swapaxes(respmat.copy(), 0, 1)
    for shuff in range(n_shuff):
        n.random.shuffle(shuff_mat)

        if nhalf is None:
            nhalf = nrep // 2
        cvpca = cv_PCA(
            reduction(shuff_mat[:nhalf], axis=0),
            reduction(shuff_mat[nhalf : 2 * nhalf], axis=0),
        )
        cvpcas.append(cvpca)
    return n.array(cvpcas)


def cv_PCA(r0, r1, n_comp=1024, return_comps=False, comps=None):
    """cvPCA, training on r0 and testing on r1
    as described in stringer 2019 Nature

    Args:
        r0 (ndarray): n_stimuli x n_neurons
        r1 (ndarray): n_stimuli x n_neurons

    Returns:
        ndarray: array of signal variance by each dimension
    """
    # r0,r1: n_stimuli x n_neurons
    n_components = min(n_comp, min(r0.shape))
    if comps is None:
        pc_comps, pc_mean, pca = fit_pca(r0, n_components=n_components, return_pca=True)
    else:
        pc_comps, pc_mean = comps
    p0 = transform(r0, pc_comps, pc_mean)
    p1 = transform(r1, pc_comps, pc_mean)
    sig_vars = (p0 * p1).sum(axis=0) / p0.shape[0]
    if return_comps:
        return sig_vars, (pc_comps, pc_mean)

    return sig_vars


def split_and_svca(spks, cell_split=None, t_split=None, n_comp=None, verbose=True, seed = None, return_splits=False):
    nc, nt = spks.shape
    if seed is not None:
        n.random.seed(seed)
    if cell_split is None:
        crand = n.random.permutation(n.arange(nc))
        cell_split = crand[: nc // 2], crand[nc // 2 :]

    if t_split is None:
        t_split = utils.chunk_indices(nt, 20, 10, (0.5, 0.5), sort=True, seed=seed)

    if n_comp is None:
        n_comp = n.min([len(cs) for cs in cell_split] + [len(ts) for ts in t_split])
        if verbose:
            print("Using %d components" % n_comp)

    # first half of cells
    a0 = spks[cell_split[0]][:, t_split[0]]
    a1 = spks[cell_split[0]][:, t_split[1]]
    # second half of cells
    b0 = spks[cell_split[1]][:, t_split[0]]
    b1 = spks[cell_split[1]][:, t_split[1]]

    if return_splits:
        return (
            svca(a0, a1, b0, b1, n_comp=n_comp, verbose=verbose)[0],
            t_split,
            cell_split,
        )

    return svca(a0, a1, b0, b1, n_comp=n_comp, verbose=verbose)


def svca(a0, a1, b0, b1, n_comp=None, verbose=True):
    # a,b are two sets of non-overlapping cells
    # 1,2 are two sets of non-overlapping timepoints

    # covariance matrix of two halves of cells at train time
    cov = a0 @ b0.T
    # svd
    uf, sf, vf = linalg.svd(cov)
    u = uf[:, :n_comp]
    s = sf[:n_comp]
    v = vf[:n_comp]
    svd = (u, s, v)
    # project test time of each set of cells to cov-space
    a1p = u.T @ a1
    b1p = v @ b1

    # shared variance of the test set timepoints in the cov-space defined by the training set timepoints
    shared_var = (a1p * b1p).sum(axis=1)
    # total variance of test set timepoints projected into cov-space
    # if two populations are identical, cov-space will capture all of the variance,
    # and shared_var == tot_var
    a1pvar = (a1p**2).sum(axis=1)
    b1pvar = (b1p**2).sum(axis=1)
    proj_vars = (a1pvar, b1pvar)
    tot_cov_space_var = (a1pvar + b1pvar) / 2

    # total variance in neural space of the test timepoints
    a1var = (a1**2).sum(axis=1)
    b1var = (b1**2).sum(axis=1)
    full_vars = (a1var, b1var)

    if verbose:
        frac_shared_variance = shared_var.sum() / tot_cov_space_var.sum()
        frac_cov_space_a1 = a1pvar.sum() / a1var.sum()
        frac_cov_space_b1 = b1pvar.sum() / b1var.sum()

        frac_shared_of_cov_space_a1 = shared_var.sum() / a1pvar.sum()
        frac_shared_of_cov_space_b1 = shared_var.sum() / b1pvar.sum()

        print("%%%.2f of the variance in the cov-space is shared" % (100 * frac_shared_variance))
        print("Cov space captures %%%.2f of variance in a1, %%%.2f of this is shared" % (100 * frac_cov_space_a1, 100 * frac_shared_of_cov_space_a1))
        print("Cov space captures %%%.2f of variance in b1, %%%.2f of this is shared" % (100 * frac_cov_space_b1, 100 * frac_shared_of_cov_space_b1))

    return shared_var, tot_cov_space_var, proj_vars, full_vars, svd


def cv_fourier_m(F1, F2, theta1, theta2, m):
    """
    Compute the cv coefficients c_{2m-1}^{cv} and c_{2m}^{cv}.

    Parameters:
    F1 (numpy.ndarray): Array of shape (N, T1) representing F_{i,t}^{(1)}.
    F2 (numpy.ndarray): Array of shape (N, T2) representing F_{i,t}^{(2)}.
    theta1 (numpy.ndarray): Array of shape (T1,) representing theta_t^{(1)}.
    theta2 (numpy.ndarray): Array of shape (T2,) representing theta_t^{(2)}.
    m (int): The index m.

    Returns:
    tuple: (c_{2m-1}^{cv}, c_{2m}^{cv})
    """
    N, T1 = F1.shape
    _, T2 = F2.shape

    # Compute the inner sums
    inner_sum1_sin = n.mean(F1 * n.sin(m * theta1), axis=1)
    inner_sum2_sin = n.mean(F2 * n.sin(m * theta2), axis=1)
    inner_sum1_cos = n.mean(F1 * n.cos(m * theta1), axis=1)
    inner_sum2_cos = n.mean(F2 * n.cos(m * theta2), axis=1)

    # Compute the outer sums
    c_2m_minus_1_cv = n.mean(inner_sum1_sin * inner_sum2_sin)
    c_2m_cv = n.mean(inner_sum1_cos * inner_sum2_cos)

    return c_2m_minus_1_cv, c_2m_cv


def cv_fourier(F1, F2, theta1, theta2, n_components, degree=True):
    """
    Compute all Fourier coefficients from 1 to n_components.

    Parameters:
    F1 (numpy.ndarray): Array of shape (N, T1) representing F_{i,t}^{(1)}.
    F2 (numpy.ndarray): Array of shape (N, T2) representing F_{i,t}^{(2)}.
    theta1 (numpy.ndarray): Array of shape (T1,) representing theta_t^{(1)}.
    theta2 (numpy.ndarray): Array of shape (T2,) representing theta_t^{(2)}.
    n_components (int): The number of components to compute.

    Returns:
    list: A list of tuples containing (c_{2m-1}^{cv}, c_{2m}^{cv}) for each m from 1 to n_components.
    """
    coefficients = []
    if degree:
        theta1 = n.deg2rad(theta1)
        theta2 = n.deg2rad(theta2)
    for m in range(1, (n_components + 1) // 2):
        c_2m_minus_1_cv, c_2m_cv = cv_fourier_m(F1, F2, theta1, theta2, m)
        coefficients.extend((c_2m_minus_1_cv, c_2m_cv))
    return n.array(coefficients)
