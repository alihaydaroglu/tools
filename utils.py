import os
import numpy as n
from multiprocessing import shared_memory
import cv2
from scipy import ndimage


def split_cells(cells, splits=(0.5, 0.5), seed=10):
    """
    Split cells into non-overlapping subsets according to splits proportions.

    Args:
        cells (ndarray): Array of cell indices to split, or a integer of number of cells.
        splits (tuple): Tuple of fractions summing to 1.0 specifying the split proportions.
        seed (int, optional): Random seed for reproducibility. Defaults to 10.    Returns:
        list[np.ndarray]: One array per split containing the selected cells.
    """
    splits = n.asarray(splits, dtype=float)
    if splits.ndim != 1 or splits.size == 0:
        raise ValueError("splits must be a 1D tuple/list with at least one entry.")
    if n.any(splits < 0):
        raise ValueError("splits must be non-negative.")
    total = splits.sum()
    if total <= 0:
        raise ValueError("Sum of splits must be > 0.")
    splits = splits / total

    if isinstance(cells, (int, n.integer)):
        pool = n.arange(int(cells))
    else:
        pool = n.asarray(cells)
    n_cells = pool.size
    if n_cells == 0:
        return [n.array([], dtype=pool.dtype) for _ in range(splits.size)]

    if seed is not None:
        n.random.seed(seed)
    shuffled = pool.copy()
    n.random.shuffle(shuffled)

    ideal = splits * n_cells
    base = n.floor(ideal).astype(int)
    remainder = n_cells - base.sum()
    if remainder > 0:
        frac = ideal - base
        order = n.argsort(-frac)
        base[order[:remainder]] += 1

    result = []
    start = 0
    for count in base:
        end = start + int(count)
        result.append(shuffled[start:end])
        start = end
    return result


def random_subsets(arr, n_subsets, n_elements, seed=10):
    """
    Generate n_subsets of random indices from arr, each containing n_elements.

    Args:
        arr (ndarray): Input array to sample from.
        n_subsets (int): Number of subsets to generate.
        n_elements (int): Number of elements in each subset.

    Returns:
        list: List of arrays containing the random indices.
    """
    if seed is not None:
        n.random.seed(seed)
    if n_elements > len(arr):
        raise ValueError("n_elements cannot be greater than the length of arr.")

    return n.array([n.random.choice(arr, size=n_elements, replace=False) for _ in range(n_subsets)])

def get_pair_idx(indices, n_dim=None, matrix=None, k=-1):
    """
    Map flat lower-triangle indices back to (row, col) pairs.

    Args:
        indices (int | iterable[int]): Integer index or iterable of indices into flatten_lower_tri output.
        n_dim (int, optional): Size of the (square) matrix. Required if matrix not provided.
        matrix (ndarray, optional): Matrix whose shape is used to infer n_dim.
        k (int, optional): Offset passed to tril_indices (same meaning as numpy.tril_indices). Defaults to -1.

    Returns:
        ndarray: Shape (n_pairs, 2) with (row, col) coordinates.
    """
    if matrix is not None:
        n_dim = matrix.shape[0]
    if n_dim is None:
        raise ValueError("Provide n_dim or matrix.")
    indices = n.atleast_1d(indices).astype(int)
    if indices.ndim != 1:
        raise ValueError("indices must be 1D or scalar.")
    tril_i, tril_j = n.tril_indices(n_dim, k=k)
    max_valid = tril_i.size - 1
    if (indices < 0).any() or (indices > max_valid).any():
        raise IndexError(f"indices out of bounds for lower triangle of size {tril_i.size}.")
    return n.column_stack((tril_i[indices], tril_j[indices]))
def flatten_lower_tri(matrix, k=-1):
    """
    return a flattened version of the lower triangular elements of matrix, excluding the the diagonal by default

    Args:
        matrix (ndarray): square matrix
        k (int, optional): Offset, -1 excludes diagonal, 0 includes diagonal. Defaults to -1.

    Returns:
        ndarray: flattened matrix of size (I think) nx * (nx - 1) / 2?
    """
    trilidx = n.tril_indices_from(matrix, -1)
    flat_matrix = matrix[trilidx]
    return flat_matrix


def make_ticks(min, max, num, round=10):
    """
    Create a list of ticks between min and max, rounded to the nearest round value

    Args:
        min (float): minimum value
        max (float): maximum value
        num (int): number of ticks
        round (int, optional): rounding factor. Defaults to 10.

    Returns:
        list: list of ticks
    """
    step = (max - min) / num
    ticks = n.arange(min, max + step, step)
    # round to the closes even multiple of round
    ticks = n.round(ticks / round) * round
    if ticks[-1] > max:
        ticks = ticks[:-1]
    return ticks


def sort_corr_mat(corr_mat, sorting_order, mask=None):
    """
    Sort a correlation matrix according to a given sorting order

    Args:
        corr_mat (ndarray): correlation matrix
        sorting_order (ndarray): array of indices to sort the correlation matrix by

    Returns:
        ndarray: sorted correlation matrix
    """
    if sorting_order is None:
        return corr_mat
    if mask is not None:
        corr_mat = corr_mat.copy()
        corr_mat[mask] = n.nan
    sorted_corr_mat = corr_mat[sorting_order]
    # print(sorted_corr_mat.shape)
    sorted_corr_mat = sorted_corr_mat[:, sorting_order]
    # print(sorted_corr_mat.shape)
    return sorted_corr_mat


def choose_cells_around_percentile(cells_pool, distribution, target_percentile, n_cells, window_percentile=10):
    """
    Choose n_cells from cells_pool that are centered around a target percentile of distribution.

    Parameters:
    - cells_pool: array of cell indices to choose from
    - distribution: population coupling values for all cells
    - target_percentile: percentile around which to center selection (0-100)
    - n_cells: number of cells to select
    - window_percentile: half-width of percentile window around target
    """
    # Get distribution values for the available cells
    pool_coupling = distribution[cells_pool]

    # Calculate target value and window bounds
    target_value = n.percentile(distribution, target_percentile)
    lower_bound = n.percentile(distribution, max(0, target_percentile - window_percentile))
    upper_bound = n.percentile(distribution, min(100, target_percentile + window_percentile))

    # Find cells within the window
    in_window = (pool_coupling >= lower_bound) & (pool_coupling <= upper_bound)
    candidate_cells = cells_pool[in_window]

    # If we have enough candidates, randomly select n_cells
    if len(candidate_cells) >= n_cells:
        selected_indices = n.random.choice(len(candidate_cells), size=n_cells, replace=False)
        return n.sort(candidate_cells[selected_indices])
    else:
        # If not enough candidates, return all available and warn
        print(f"Warning: Only {len(candidate_cells)} cells available in window, requested {n_cells}")
        return n.sort(candidate_cells)


"""
Split a time-indexed dataset into non-overlapping index chunks separated by a buffer, with optional regrouping into splits.
Parameters:
- n_dataset (int): Total number of timepoints in the dataset.
- n_chunks (int, optional): Number of chunks to create. Mutually exclusive with chunksize. If set, chunksize is computed as floor((n_dataset - n_buffer*(n_chunks-1)) / n_chunks).
- n_buffer (int, default 10): Number of timepoints to skip between consecutive chunks.
- splits (tuple[float], optional): Fractions summing to 1.0 specifying how to regroup whole chunks into that many sets (e.g., train/val/test). Chunks are randomly assigned per these proportions and concatenated within each set.
- chunksize (int, optional): Size of each chunk (timepoints). Mutually exclusive with n_chunks. If set, n_chunks is inferred as n_dataset // (chunksize + n_buffer) - 1.
- cv_fold (bool, default False): Must be False when splits is provided (reserved flag).
- sort (bool, default False): If True and splits is provided, sort indices within each resulting set.
- seed (int, optional): Random seed for reproducible chunk assignment when splits is used.
Returns:
- list of 1D numpy arrays of int indices:
    - Without splits: one array per chunk (contiguous ranges of length chunksize).
    - With splits: one array per split, each containing concatenated indices from its assigned chunks.
Notes:
- Indices range from 0 to n_dataset - 1.
- When using splits, the number of chunks per split is computed by floor(n_chunks * split) per entry.
"""


def chunk_indices(
    n_dataset,
    n_chunks=None,
    n_buffer=10,
    splits=None,
    chunksize=None,
    cv_fold=False,
    sort=False,
    seed=1010,
):
    if seed is not None:
        n.random.seed(seed)
    chunks = []
    if chunksize is None:
        chunksize = (n_dataset - (n_buffer * (n_chunks - 1))) // n_chunks
    else:
        n_chunks = n_dataset // (chunksize + n_buffer) - 1
    i = 0
    # print(n_dataset, n_chunks, chunksize)
    while i + chunksize <= n_dataset:
        chunks.append(n.arange(i, i + chunksize))
        i += chunksize
        i += n_buffer

    if splits is not None:
        assert not cv_fold
        assert n.sum(splits) == 1
        chunks_per_split = [int(n_chunks * split) for split in splits]
        chunk_rand_order = n.random.choice(n.arange(n_chunks), n_chunks, replace=False)
        new_chunks = []
        chunks = n.array(chunks)
        idx = 0
        for i, n_chunks_split in enumerate(chunks_per_split):
            split_chunk_idxs = chunk_rand_order[idx : idx + n_chunks_split]
            idx += n_chunks_split
            new_chunks.append(n.concatenate(chunks[split_chunk_idxs], axis=0))
        chunks = new_chunks
        if sort:
            for i, chunk in enumerate(chunks):
                chunks[i] = n.sort(chunk)
    return chunks


def create_shmem(shmem_params):
    shmem = shared_memory.SharedMemory(create=True, size=shmem_params["nbytes"])
    shmem_params["name"] = shmem.name
    return shmem, shmem_params


def create_shmem_from_arr(sample_arr, copy=False):
    shmem_params = {
        "dtype": sample_arr.dtype,
        "shape": sample_arr.shape,
        "nbytes": sample_arr.nbytes,
    }
    shmem, shmem_params = create_shmem(shmem_params)
    sh_arr = n.ndarray(shmem_params["shape"], shmem_params["dtype"], buffer=shmem.buf)
    if copy:
        sh_arr[:] = sample_arr[:]
    else:
        sh_arr[:] = 0

    return shmem, shmem_params, sh_arr


def load_shmem(shmem_params):
    shmem = shared_memory.SharedMemory(name=shmem_params["name"], create=False)
    sh_arr = n.ndarray(shmem_params["shape"], shmem_params["dtype"], buffer=shmem.buf)
    return shmem, sh_arr


def close_shmem(shmem_params):
    shmem = shared_memory.SharedMemory(name=shmem_params["name"], create=False)
    shmem.close()


def close_and_unlink_shmem(shmem_params):
    print("Don't use me. I cause memory leaks :(")
    if "name" in shmem_params.keys():
        shmem = shared_memory.SharedMemory(name=shmem_params["name"], create=False)
        shmem.close()
        shmem.unlink()


def validate_and_fill_defaults(provided, defaults, name="parameters"):
    if provided is None:
        provided = {}
    if not isinstance(provided, dict):
        raise TypeError(f"{name} must be a dict or None.")

    extra_keys = set(provided.keys()) - set(defaults.keys())
    if extra_keys:
        raise ValueError(f"Unexpected keys in {name}: {extra_keys}")

    # Fill in defaults
    filled = {**defaults, **provided}
    return filled


def get_frametimes(filenames, vid_dir=""):
    all_frames = []
    all_times = []
    for filename in filenames:
        print(filename)
        if type(filename) != str:
            filename = filename[0]

        logfile = filename[:-4] + "_times.txt"
        times = []
        frame_ns = []
        f = open(os.path.join(vid_dir, logfile))
        read = f.read()
        f.close()

        lines = read.split("\n")
        for line in lines[2:-1]:
            line_list = line.split("\t")
            frame_ns.append(int(line_list[1]))
            times.append(float(line_list[-1]))
        all_frames.append(n.array(frame_ns)), all_times.append(n.array(times))
    return all_frames, all_times


def make_alpha_mask(data, min_p=None, max_p=None, filt=0, exp=1.0, min_val=None, max_val=None):
    """
    Create an alpha mask for the data based on percentile thresholds and Gaussian filtering.

    Args:
        data (ndarray): Input data array.
        min_p (float): Minimum percentile threshold.
        max_p (float): Maximum percentile threshold.
        filt (int, optional): Gaussian filter size. Defaults to 0 (no filtering).
        exp (float, optional): Exponent for normalization. Defaults to 1.0.

    Returns:
        ndarray: Normalized and filtered alpha mask.
    """
    mask = data.copy()
    mask[n.isnan(mask)] = 0.0
    if filt > 0:
        mask = ndimage.gaussian_filter(mask, filt, mode="constant")
    if min_val is None:
        min_val = n.nanpercentile(mask, min_p)
    if max_val is None:
        max_val = n.nanpercentile(mask, max_p)
    mask[mask < min_val] = min_val
    mask[mask > max_val] = max_val
    mask = (mask - min_val) / (max_val - min_val)
    mask = mask**exp
    return mask
