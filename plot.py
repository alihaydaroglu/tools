import numpy as n
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from . import mathfuncs as math
from . import utils
from scipy import stats

colors = ["#90be6d", "#e98a15", "#b26c98", "#1b9aaa", "#3a405a"]

# when importing this, set the following rcParams:
# by default, save with dpi 200
# save without whitespace, bbox_in='tight' and pad_inched = 0.1
# save fonts as text, not images
# ALL fonts should be Arial.
# Axis ticks should be 8 pts
# axis labels should be 12 pts
# titles should be 12 pts
# legends should be 10 pts
mpl.rcParams["savefig.dpi"] = 200
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.pad_inches"] = 0.1
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
mpl.rcParams["font.size"] = 12
mpl.rcParams["axes.titlesize"] = 12
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10
mpl.rcParams["legend.fontsize"] = 10
# save text as text, not images
# set default figsize to (4,3)
mpl.rcParams["figure.figsize"] = (4, 3)
# set default dpi to 150
mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["svg.fonttype"] = "none"
# default filetype is svg
mpl.rcParams["savefig.format"] = "svg"


def multiple_timeseries(
    yss,
    ts=None,
    colors=None,
    labels=None,
    alphas=None,
    lws=None,
    zscore=True,
    dy=3.0,
    auto_ylim=True,
    ax=None,
    tick_labels=True,
    legend=False,
    lw=1.0,
    alpha=1.0,
    color=None,
    filt=None,
    swap_yorder=False,
    ylabel_rot=0,
    dy_offset=0,
    yposs=None,
    idx_lims=None,
    tlims=None,
):
    if ax is None:
        f, ax = plt.subplots()
    n_lines = len(yss)
    yticks = []
    lines = []

    if ts is None:
        ts = n.arange(len(yss[0]))
    elif len(n.shape(ts)) == 0:
        ts = n.arange(len(yss[0])) * ts

    ts = n.array(ts)
    print(ts.shape)
    print(tlims)
    if tlims is not None:
        idx0 = n.argmin(n.abs(ts - tlims[0]))
        idx1 = n.argmin(n.abs(ts - tlims[1]))
        idx_lims = (idx0, idx1)
        # print(idx_lims)

    if idx_lims is not None:
        ts = ts[idx_lims[0] : idx_lims[1]]
    for i in range(n_lines):
        color = colors[i] if colors is not None else color
        alpha = alphas[i] if alphas is not None else alpha
        label = labels[i] if labels is not None else None
        lw = lws[i] if lws is not None else lw
        ys = yss[i]

        if idx_lims is not None:
            ys = ys[idx_lims[0] : idx_lims[1]]

        if zscore:
            ys = math.zscore(ys)
        if filt is not None:
            ys = math.filt(ys, filt)
        if yposs is None:
            ypos = dy * i + dy_offset
            if swap_yorder:
                ypos = dy * (n_lines - 1 - i)
        else:
            ypos = yposs[i]
        lines += ax.plot(ts, ys + ypos, color=color, alpha=alpha, linewidth=lw, label=label)
        yticks.append(ypos)
    if labels is not None and legend:
        ax.legend(lines[::-1], labels[::-1], frameon=True, facecolor="white")
    ax.set_yticks(yticks)
    if labels is not None and tick_labels:
        ax.set_yticklabels(labels, rotation=ylabel_rot)
    else:
        ax.set_yticklabels([""] * len(yticks))

    if auto_ylim:
        ax.set_ylim(-dy, dy * (i + 1))

    ax.set_xlim(ts.min(), ts.max())
    return ax

# ...existing code...
def density_scatter(
    x, y,
    *,
    cmap: str = "viridis",
    ax=None,
    s: float = 10,
    cbar: bool = False,
    density: str = 'hist',
    density_bins: int = 64,      # for 'hist'
    gaussian_sigma: float | int = 0,  # smoothing on histogram (pixels)
    knn_k: int = 20,              # for 'knn'
    log_scale: bool = False,      # use logarithmic color scale
    # colorbar inside-axis options
    cbar_loc: str = 'lower right',
    cbar_size: str = '3%',
    cbar_height: str = '20%',
    cbar_borderpad: float = 0.2,
    cbar_orientation: str = 'vertical',
    # identity line options
    identity_line: bool = False,
    max_pts = 5000, # max points to plot, randomly subsampled if more
    # statistics / legend options
    show_stats: bool = False,
    stats_loc: str = 'best',
    stats_fmt: str = 'slope={slope:.3g}, r={r:.3g}, p={p:.1e}',
    stats_frameon: bool = False,
    **scatter_kwargs
):
    """Scatter plot colored by local point density.

    Parameters
    ----------
    x, y : array-like
        1D arrays of the same length.
    density : {'gaussian','hist','knn','uniform'}
        - 'gaussian': scipy.stats.gaussian_kde (slow for large n)
        - 'hist': 2D histogram (+ optional Gaussian blur) then per-point lookup
        - 'knn': k-NN density via cKDTree using 1/(pi r_k^2)
        - 'uniform': constant color (fallback)
    density_bins : int
        Number of bins per axis for 'hist'.
    gaussian_sigma : float
        Gaussian blur sigma (in bins) for 'hist'. 0 disables smoothing.
    knn_k : int
        k for k-NN density.
    log_scale : bool
        If True, use a logarithmic color scale (matplotlib.colors.LogNorm).
    identity_line : bool
        If True, draw a gray dashed identity line (y = x) behind the scatter without
        changing axis limits.
    show_stats : bool
        If True, compute linear regression (scipy.stats.linregress) and add a legend entry
        containing slope, Pearson r, and p-value using stats_fmt.
    stats_loc : str
        Matplotlib legend location for the stats string (if show_stats=True).
    stats_fmt : str
        Format string with placeholders {slope}, {r}, {p}, {intercept}.
    stats_frameon : bool
        Whether the legend frame is shown when displaying stats.
    """
    # Convert and clean inputs
    x = n.asarray(x).ravel()
    y = n.asarray(y).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    
    if x.size > max_pts:
        idx = n.random.choice(x.size, size=max_pts, replace=False)
        x = x[idx]
        y = y[idx]

    mask = n.isfinite(x) & n.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        raise ValueError("No finite points to plot")

    # Compute point density
    xy = n.vstack([x, y])
    z = None
    if density == 'gaussian':
        try:
            kde = stats.gaussian_kde(xy)
            z = kde(xy)
        except Exception:
            # Fallback to fast histogram method if KDE fails (e.g., singular covariance)
            density = 'hist'

    if density == 'hist':
        # 2D histogram on a grid
        H, xedges, yedges = n.histogram2d(x, y, bins=density_bins)
        if gaussian_sigma and gaussian_sigma > 0:
            from scipy.ndimage import gaussian_filter
            H = gaussian_filter(H, gaussian_sigma, mode='constant')
        # Map each point to its bin count (fast)
        ix = n.clip(n.digitize(x, xedges) - 1, 0, H.shape[0] - 1)
        iy = n.clip(n.digitize(y, yedges) - 1, 0, H.shape[1] - 1)
        z = H[ix, iy] + 1e-12  # avoid zeros

    elif density == 'knn':
        # k-NN density estimate using area of circle to k-th neighbor
        from scipy.spatial import cKDTree
        tree = cKDTree(n.c_[x, y])
        dists, _ = tree.query(n.c_[x, y], k=knn_k + 1)  # include self
        rk = dists[:, -1]
        area = n.pi * n.maximum(rk, 1e-12) ** 2
        z = 1.0 / area

    elif density == 'uniform':
        z = n.full_like(x, fill_value=1.0 / max(1, x.size), dtype=float)

    if z is None:
        raise ValueError("Unknown density method. Use 'gaussian', 'hist', 'knn', or 'uniform'.")

    # Sort so densest points are plotted last
    idx = n.argsort(z)
    x_sorted = x[idx]
    y_sorted = y[idx]
    z_sorted = z[idx]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))
        created_fig = True
    else:
        fig = None

    # Apply logarithmic normalization if requested (unless user already provided a norm)
    if log_scale and ('norm' not in scatter_kwargs):
        # Ensure strictly positive vmin for LogNorm
        zpos = z_sorted[z_sorted > 0]
        if zpos.size == 0:
            zpos = n.array([1.0])
        vmin = scatter_kwargs.get('vmin', float(zpos.min()))
        vmax = scatter_kwargs.get('vmax', float(z_sorted.max()))
        scatter_kwargs['norm'] = mpl.colors.LogNorm(vmin=max(vmin, 1e-12), vmax=max(vmax, vmin * 1.000001))

    sc = ax.scatter(x_sorted, y_sorted, c=z_sorted, s=s, cmap=cmap, **scatter_kwargs)

    # Optional identity line (y=x) behind points; restore limits so it doesn't affect view
    if identity_line:
        xlim0 = ax.get_xlim()
        ylim0 = ax.get_ylim()
        lo = float(min(xlim0[0], ylim0[0]))
        hi = float(max(xlim0[1], ylim0[1]))
        try:
            zbase = float(sc.get_zorder())
        except Exception:
            zbase = 1.0
        ax.plot([lo, hi], [lo, hi], color='0.6', linestyle='--', linewidth=1.0, zorder=zbase - 1)
        ax.set_xlim(xlim0)
        ax.set_ylim(ylim0)

    # Optional stats legend (after plotting so limits unaffected)
    if show_stats and x.size > 1:
        try:
            lr = stats.linregress(x, y)
            label = stats_fmt.format(slope=lr.slope, r=lr.rvalue, p=lr.pvalue, intercept=lr.intercept)
            sc.set_label(label)
            # Only draw legend if not already present (or user wants it explicitly)
            existing_legend = ax.get_legend()
            if existing_legend is None:
                ax.legend(loc=stats_loc, frameon=stats_frameon)
        except Exception:
            # Silently ignore regression errors (e.g., constant input)
            pass
    if cbar:
        # Create an inset colorbar inside the plotting axes using fixed bounds to avoid
        # AnchoredLocator issues during save/render.
        fig_for_cb = ax.figure if ax is not None else plt.gcf()

        def _as_frac(v, default_frac):
            # Convert values like '3%' -> 0.03, numbers <=1 kept as-is, >1 treated as percent.
            if isinstance(v, str) and v.endswith('%'):
                try:
                    return float(v[:-1]) / 100.0
                except Exception:
                    return default_frac
            try:
                vf = float(v)
                if vf <= 1.0:
                    return vf
                # Treat e.g. 3 as 3%
                return vf / 100.0
            except Exception:
                return default_frac

        w_frac = _as_frac(cbar_size, 0.03)
        h_frac = _as_frac(cbar_height, 0.4)
        pad = float(cbar_borderpad) if cbar_borderpad is not None else 0.02
        # If pad looks like inches (large), clamp to a small fraction
        if pad > 0.5:
            pad = 0.02

        # Compute bounds in axes fraction coordinates based on location keyword
        loc = (cbar_loc or 'upper right').lower()
        if loc == 'upper right':
            x0 = 1 - w_frac - pad
            y0 = 1 - h_frac - pad
        elif loc == 'upper left':
            x0 = pad
            y0 = 1 - h_frac - pad
        elif loc == 'lower right':
            x0 = 1 - w_frac - pad
            y0 = pad
        elif loc == 'lower left':
            x0 = pad
            y0 = pad
        else:
            # fallback: upper right
            x0 = 1 - w_frac - pad
            y0 = 1 - h_frac - pad

        cbax = ax.inset_axes([x0, y0, w_frac, h_frac])
        cbar_obj = mpl.colorbar.Colorbar(cbax, sc, orientation=cbar_orientation)
        # Optional: keep the colorbar tidy inside the axis
        for spine in cbax.spines.values():
            spine.set_linewidth(0.5)
    return fig, ax, sc
# ...existing code...


def plot_onsets(onset_times, offset_times, ax, alpha=0.5, color="grey"):
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    for i in range(len(onset_times)):
        patch1 = ax.fill_between([onset_times[i], offset_times[i]], *ylim, color=color, alpha=alpha)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    return patch1


def fill_cells_vol(coords, fill_vals, shape=None, empty=n.nan, filt=None, squeeze=True, proj=None):
    """
    create a 3d volume (or a 2d projection of it) of all cells filled with specified values

    Args:
        coords (list): length n_cells, coords output from Suite3D
        fill_vals (ndarray): one value per cell to fill
        shape (tuple, optional): shape of the volume. if empty, try to estimate automatically. Defaults to None.
        empty (float, optional): value to fill pixels that aren't in any cells. Defaults to n.nan.
        filt (ndarray, optional): bool array of size n_cells to filter out unwanted cells. Defaults to None.
        squeeze (bool, optional): honestly i'm not sure. Defaults to True.
        proj (str, optional): whether to project result into a 2d array with a 'max', 'mean' or 'median' projection. Defaults to None.

    Returns:
        vols: 3d or 2d array, where each pixel belonging to a cell is filled with specified values
    """
    expanded = False
    if len(fill_vals.shape) == 1:
        fill_vals = fill_vals[n.newaxis]
        expanded = True
        # print(fil#l_vals.shape)
    n_dims = fill_vals.shape[0]
    vols = []
    # print(filt.shape)
    for idx in range(n_dims):
        if shape is None:
            shape = n.array([n.max(n.concatenate([c[i] for c in coords]) + 1) for i in range(3)])
        vol = n.zeros(shape) * empty
        i = -1
        for coord, fill_val in zip(coords, fill_vals[idx]):
            i += 1
            if filt is not None:
                # print(i,filt[i])
                if not filt[i]:
                    continue
                # print(i, filt[i])
            vol[coord[0], coord[1], coord[2]] = fill_val
        vols.append(vol)

        # print(n.nanmean(vol))
    vols = n.array(vols)
    if expanded and squeeze:
        vols = vols[0]
    if proj == "max":
        vols = n.nanmax(vols, axis=0)
    elif proj == "median":
        vols = n.nanmedian(vols, axis=0)
    elif proj == "mean":
        vols = n.nanmean(vols, axis=0)
    return vols


def fill_cells_plane(coords, fill_vals, shape=None, empty=n.nan, filt=None, squeeze=True):
    """
    Create a 2D (y, x) plane by ignoring z and directly computing the mean projection over z.

    This is a faster alternative to building a full 3D volume and then projecting with proj='mean'.

    Args:
        coords (list): length n_cells, Suite3D-like coords where each element is a tuple/list
            of index arrays (z, y, x). Only y and x are used.
        fill_vals (ndarray): one value per cell to fill. Shape (n_cells,) or (n_dims, n_cells).
        shape (tuple, optional): If 3D, interpreted as (z, y, x) and we use (y, x).
            If 2D, interpreted directly as (y, x). If None, inferred from coords.
        empty (float, optional): value to fill pixels that aren't in any cells. Defaults to n.nan.
        filt (ndarray, optional): bool array of size n_cells to filter out unwanted cells. Defaults to None.
        squeeze (bool, optional): If fill_vals was 1D, return a 2D array when True. Defaults to True.

    Returns:
        planes: 2D array (y, x) or 3D array (n_dims, y, x) depending on fill_vals shape and squeeze.
    """
    expanded = False
    fill_vals = n.asarray(fill_vals)
    if fill_vals.ndim == 1:
        fill_vals = fill_vals[n.newaxis, :]
        expanded = True

    n_dims = fill_vals.shape[0]

    # Determine (y, x) shape
    if shape is None:
        ny = int(n.max(n.concatenate([c[1] for c in coords])) + 1)
        nx = int(n.max(n.concatenate([c[2] for c in coords])) + 1)
    else:
        if len(shape) == 3:
            ny, nx = int(shape[1]), int(shape[2])
        elif len(shape) == 2:
            ny, nx = int(shape[0]), int(shape[1])
        else:
            # Best effort: expect last two entries are (y, x)
            ny, nx = int(shape[-2]), int(shape[-1])

    planes = []
    for d in range(n_dims):
        sum2d = n.zeros((ny, nx), dtype=n.float32)
        cnt2d = n.zeros((ny, nx), dtype=n.int32)

        for i, (coord, v) in enumerate(zip(coords, fill_vals[d])):
            if filt is not None and not filt[i]:
                continue
            ys = coord[1]
            xs = coord[2]
            # Accumulate value per voxel projected onto (y, x)
            n.add.at(sum2d, (ys, xs), v)
            n.add.at(cnt2d, (ys, xs), 1)

        # Compute mean while preserving 'empty' on pixels with zero count
        plane = n.full((ny, nx), empty, dtype=n.float32)
        mask = cnt2d > 0
        plane[mask] = (sum2d[mask] / cnt2d[mask]).astype(n.float32)
        planes.append(plane)

    planes = n.array(planes)
    if expanded and squeeze:
        planes = planes[0]
    return planes


# light wrapper around show_img that sets cmap='RdBu_r', symmetric_cmap=True, and has the additional optional parameter
def show_covmat(covmat, ax=None, fscale = 3, vscale=None, nan_diag=True, sort=None, **kwargs):
    """
    Show a covariance/correlation matrix with RdBu_r colormap, symmetric scaling.
    Args:
        covmat: 2D numpy array
        ax: matplotlib axis (optional)
        vscale: float or None, if set, vmin/vmax = -vscale, vscale
        **kwargs: extra arguments to show_img
    Returns:w
        fig, ax, axim
    """
    if nan_diag:
        covmat = covmat.copy()
        covmat[n.diag_indices_from(covmat)] = n.nan
    if sort is not None:
        covmat = covmat.copy()
        covmat = covmat[:,sort][sort]
    cmap = "RdBu_r"
    symmetric_cmap = True
    if vscale is not None:
        vminmax = (-vscale, vscale)
        vminmax_percentile = None
    else:
        vminmax = None
    return show_img(
        covmat,
        cmap=cmap,
        symmetric_cmap=symmetric_cmap,
        ax=ax,
        figsize=(fscale,fscale),
        vminmax=vminmax,
        **kwargs,
    )



def linear_cmap(
    low_color="white",
    high_color="darkred",
    name="linear",
    nan_color="lightgrey",
    scale="linear",
    vmin=None,
    vmax=None,
    clip=False,
):
    """
    Create a 2-color colormap (low->high) with configurable NaN color and normalization.

    Args:
        low_color (str): Color for the low end of the colormap. Default "white".
        high_color (str): Color for the high end of the colormap. Default "darkred".
        name (str): Name of the colormap.
        nan_color (str): Color for NaN values. Default "lightgrey".
        scale (str): "linear" or "log". Controls the norm used.
        vmin (float|None): Minimum value for normalization.
        vmax (float|None): Maximum value for normalization.
        clip (bool): Whether to clip values outside vmin/vmax.

    Returns:
        (cmap, norm): A matplotlib colormap and a matching Normalize/LogNorm instance.
    """
    if scale not in ("linear", "log"):
        raise ValueError("scale must be 'linear' or 'log'")

    cmap = mpl.colors.LinearSegmentedColormap.from_list(name, [low_color, high_color])
    # Set NaN color
    try:
        # Matplotlib >= 3.3
        cmap = cmap.with_extremes(bad=nan_color)
    except Exception:
        cmap.set_bad(nan_color)

    if scale == "log":
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax, clip=clip)
    else:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=clip)

    return cmap, norm


def show_img(
    im,
    flip=1,
    cmap="Greys_r",
    colorbar=False,
    other_args={},
    figsize=(8, 6),
    dpi=150,
    alpha=None,
    return_fig=True,
    transpose=False,
    bin=None,
    ticks=False,
    ax=None,
    px_py=None,
    exact_pixels=False,
    vminmax_percentile=(0.5, 99.5),
    vminmax=None,
    facecolor="white",
    xticks=None,
    yticks=None,
    flip_y = False,
    return_cax=False,
    cbar_fontsize=9,
    norm=None,
    cbar=False,
    cbar_loc="left",
    cbar_ori="vertical",
    cbar_title="",
    interpolation="nearest",
    ax_off=False,
    cax_kwargs={"frameon": False},
    extent=None,
    spines=False,
    symmetric_cmap=False,
    aspect=None,
    cax_fontcolor="k",
    cax_label_format="%.2f",
):

    f = None
    im = im.copy()
    if type(bin) is tuple:
        im = math.bin1d(math.bin1d(im, bin[0], axis=0), bin[1], axis=1)
    elif bin is not None:
        im = math.bin1d(math.bin1d(im, bin, axis=0), bin, axis=1)
        # print("binned")
    if transpose:
        # print(im.shape)
        im = n.moveaxis(im, (0, 1), (1, 0))
        # print(im.shape)
        if alpha is not None:
            alpha = alpha.T
    if exact_pixels:
        ny, nx = im.shape
        figsize = (nx / dpi, ny / dpi)
        px_py = None
        print(ny, nx, figsize, dpi)

    new_args = {}
    new_args.update(other_args)
    if ax is None:
        f, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if facecolor is not None:
        ax.set_facecolor(facecolor)
    ax.grid(False)
    new_args["interpolation"] = interpolation
    if vminmax_percentile is not None and vminmax is None:
        new_args["vmin"] = n.nanpercentile(im, vminmax_percentile[0])
        new_args["vmax"] = n.nanpercentile(im, vminmax_percentile[1])
    if vminmax is not None:
        new_args["vmin"] = vminmax[0]
        new_args["vmax"] = vminmax[1]
    if symmetric_cmap:
        # print(new_args["vmin"])
        # print(new_args["vmax"])
        vmax_abs = max(n.abs(new_args["vmin"]), n.abs(new_args["vmax"]))
        new_args["vmin"] = -vmax_abs
        new_args["vmax"] = vmax_abs
        # print(new_args["vmin"])
        # print(new_args["vmax"])
    if px_py is not None:
        new_args["aspect"] = px_py[1] / px_py[0]
    if aspect is not None:
        new_args["aspect"] = aspect
    if alpha is not None:
        new_args["alpha"] = alpha.copy()
    if norm is not None:
        new_args["norm"] = norm
        new_args["vmin"] = None
        new_args["vmax"] = None
    if extent is not None:
        new_args["extent"] = extent
    # print(new_args)
    axim = ax.imshow(flip * im, cmap=cmap, **new_args)
    if colorbar:
        plt.colorbar()
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if exact_pixels:
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # plt.tight_layout()
    if norm:
        new_args["vmin"] = norm.vmin
        new_args["vmax"] = norm.vmax
    if cbar:
        if cbar_loc == "left":
            cbar_loc = [0.025, 0.4, 0.02, 0.2]
            cbar_ori = "vertical"
        if cbar_loc == "lower left":
            cbar_loc = [0.025, 0.025, 0.02, 0.2]
            cbar_ori = "vertical"
        elif cbar_loc == "right":
            cbar_loc = [0.88, 0.4, 0.02, 0.2]
            cbar_ori = "vertical"
        elif cbar_loc == "top":
            cbar_loc = [0.4, 0.95, 0.2, 0.02]
            cbar_ori = "horizontal"
        elif cbar_loc == "bottom":
            cbar_loc = [0.4, 0.05, 0.2, 0.02]
            cbar_ori = "horizontal"
        cax = ax.inset_axes(cbar_loc, **cax_kwargs)
        plt.colorbar(axim, cax=cax, orientation=cbar_ori)
        if cbar_ori == "vertical":
            cax.set_yticks(
                [new_args["vmin"], new_args["vmax"]],
                [
                    cax_label_format % new_args["vmin"],
                    cax_label_format % new_args["vmax"],
                ],
                color=cax_fontcolor,
                fontsize=cbar_fontsize,
            )
            cax.set_ylabel(cbar_title, color=cax_fontcolor, fontsize=cbar_fontsize, labelpad=-10)
        if cbar_ori == "horizontal":
            cax.set_xticks(
                [new_args["vmin"], new_args["vmax"]],
                [
                    cax_label_format % new_args["vmin"],
                    cax_label_format % new_args["vmax"],
                ],
                color=cax_fontcolor,
                fontsize=cbar_fontsize,
            )
            cax.set_xlabel(cbar_title, color=cax_fontcolor, fontsize=cbar_fontsize, labelpad=-10)
    if xticks is not None:
        ax.set_xticks(range(len(xticks)), xticks)
    if yticks is not None:
        ax.set_yticks(range(len(yticks)), yticks)
    if ax_off:
        ax.axis("off")
    if not spines:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
    if flip_y:
        ax.invert_yaxis()

    if return_cax:
        return f, ax, axim, cax
    if return_fig:
        return f, ax, axim


def turn_off_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


def compute_alphas(values, min=0, max=1):
    alphas = values.copy()
    alphas[alphas < min] = min
    alphas[alphas > max] = max
    alphas -= min
    alphas /= max - min
    return alphas


def fit_powerlaw(explained_vars, range=(11, 500), normalize=True):
    xs = n.arange(0, explained_vars.size) + 1
    if normalize:
        ys = explained_vars / explained_vars.sum()
    else:
        ys = explained_vars
    alpha, ypred, intercept = get_powerlaw(ys, n.arange(*range).astype(int))
    return xs, ys, ypred, alpha, intercept


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


def plot_powlaw(sig_vars, ax=None, color=None, label="", plaw_range=(11, 500), fit_line=True, lw_curve=4, alpha_curve=0.8, lw_fit=4, alpha_fit=0.3, normalize=True, marker=None, figsize=(3, 3)):
    if ax is None:
        f, ax = plt.subplots(figsize=figsize)
    if color is None:
        color = "k"
    if normalize:
        to_plot = sig_vars / sig_vars.sum()
    else:
        to_plot = sig_vars
    # print(len(sig_vars))
    if len(sig_vars) < plaw_range[1]:
        plaw_range = (plaw_range[0], len(sig_vars))
    alpha = None
    if fit_line:
        xs, alpha, yp, alpha, b = fit_powerlaw(to_plot, plaw_range, normalize=normalize)
    label = label
    if fit_line:
        label += " $\\alpha$ = %.2f" % alpha
    ax.loglog(
        n.arange(1, len(to_plot) + 1),
        to_plot,
        color=color,
        alpha=alpha_curve,
        linewidth=lw_curve,
        label=label,
        marker=marker,
    )
    if fit_line:
        ax.loglog(xs, yp, color=color, alpha=alpha_fit, linewidth=lw_fit)
    return ax, alpha


def hist2d(
    xs,
    ys,
    nbins=51,
    xlims=None,
    ylims=None,
    regression=True,
    ax=None,
    log=True,
    cbar=True,
    cmap="Blues",
    density=False,
    clims=(None, None),
    plot_identity=False,
    xbins=None,
    ybins=None,
    regression_line_params={},
    fix_nans=True,
    lim_percentile=None,
    slope_in_label=False,
    legend_loc="upper right",
):
    if fix_nans:
        nans = n.isnan(xs) | n.isnan(ys)
        if nans.sum() > 0:
            xs = xs[~nans]
            ys = ys[~nans]
    if ax is None:
        f, ax = plt.subplots(figsize=(6, 6))

    if xlims is None:
        xlims = xs.min(), xs.max() * 1.01
    if ylims is None:
        ylims = ys.min(), ys.max() * 1.01

    if xbins is None:
        if lim_percentile:
            xbins = n.linspace(
                n.percentile(xs, lim_percentile),
                n.percentile(xs, 100 - lim_percentile),
                nbins,
            )
        else:
            xbins = n.linspace(*xlims, nbins)
    if ybins is None:
        if lim_percentile:
            ybins = n.linspace(
                n.percentile(ys, lim_percentile),
                n.percentile(ys, 100 - lim_percentile),
                nbins,
            )
        else:
            ybins = n.linspace(*ylims, nbins)

    if log:
        norm = mpl.colors.LogNorm(vmin=clims[0], vmax=clims[1])
    else:
        norm = mpl.colors.Normalize(vmin=clims[0], vmax=clims[1])

    if clims is None:
        clims = (None, None)

    hist = ax.hist2d(xs, ys, bins=(xbins, ybins), cmap=cmap, norm=norm, density=density)

    if cbar:
        plt.colorbar(hist[-1], ax=ax)

    if plot_identity:
        ax.plot(xlims, xlims, color="k", alpha=0.2, lw=3, linestyle="--")

    if regression:
        slopex, interceptx, rx, px, __ = stats.linregress(xs, ys)
        if slope_in_label:
            label = label = "y=%.2fx + %.2f\nR (CoD) : %.2f" % (slopex, interceptx, rx)
        else:
            label = "R: %.2f" % rx
        ax.plot(
            xbins,
            xbins * slopex + interceptx,
            color="k",
            label=label,
            **regression_line_params,
        )
        ax.legend(loc=legend_loc)
    # print(hist[-1].get_clim())

    return ax


def plot_timeseries(timeseries, ax=None):
    if ax is None:
        f, ax = plt.subplots(figsize=(8, 6))
    ax.plot(timeseries.ts, timeseries.data[0])


from pandas.api.types import is_integer_dtype


def pairplot(
    df,
    nbin=101,
    fboxsize=3,
    dpi=150,
    symmetric=False,
    log=True,
    pctile=99.9,
    ticks_only_on_edge=True,
    labels_only_on_edge=True,
):
    ncell, ncol = df.shape
    bins = []
    bounds = []
    for i, col in enumerate(df.columns):
        pmax = n.percentile(df[col], pctile)
        pmin = n.percentile(df[col], 100 - pctile)
        bounds.append((pmin, pmax))
        if is_integer_dtype(df[col].dtype) and df[col].max() < nbin and df[col].min() >= 0:
            bins.append(n.linspace(pmin, pmax, df[col].max() + 1))
        else:
            bins.append(n.linspace(pmin, pmax, nbin))
    f, axs = plt.subplots(ncol, ncol, figsize=(ncol * fboxsize, ncol * fboxsize), dpi=dpi)
    for i in range(ncol):
        jmax = ncol if symmetric else i + 1
        for j in range(ncol):
            col_i = df.columns[i]
            col_j = df.columns[j]
            ax = axs[i][j]
            if j >= jmax:
                ax.axis("off")
                continue
            if i == j:
                ax.hist(df[col_i], bins=bins[i])
                ax.set_yticks([])
                ax.set_xticks(bounds[i])
                if i == 0:
                    ax.set_ylabel(col_i)
                    ax.set_yticks(ax.get_ylim())
                    ax.set_xticklabels(["%.2f" % b for b in bounds[i]])

                    ax.set_yticklabels(["0", "1.0"])
                elif i == ncol - 1:
                    ax.set_xticklabels(["%.2f" % b for b in bounds[i]])
                    ax.set_xlabel(col_j)
                elif not ticks_only_on_edge or not labels_only_on_edge:
                    if not ticks_only_on_edge:

                        #         ax.set_xticks(ax.get_ylim())
                        #         # ax.set_yticklabels(["%.2f" % b for b in bounds[i]])
                        ax.set_xticklabels(["%.2f" % b for b in bounds[i]])
                #     if not labels_only_on_edge:
                #         # ax.set_ylabel(col_i)
                #         ax.set_xlabel(col_j)

                else:
                    ax.set_xticklabels([None for b in bounds[i]])
                if n.prod(bounds[i]) < 0:
                    ax.axvline(0, color="k", alpha=0.25)
            else:
                hist2d(
                    df[col_j],
                    df[col_i],
                    ax=ax,
                    plot_identity=True,
                    cbar=False,
                    slope_in_label=False,
                    cmap="Blues",
                    xbins=bins[j],
                    ybins=bins[i],
                    log=log,
                    regression_line_params={"alpha": 0.5},
                )
                ax.set_xticks(bounds[j])

                if i == ncol - 1 or not labels_only_on_edge:
                    ax.set_xlabel(col_j)
                if i == ncol - 1 or not ticks_only_on_edge:
                    ax.set_xticklabels(["%.2f" % b for b in bounds[j]])
                else:
                    ax.set_xticklabels([None for b in bounds[j]])

                ax.set_yticks(bounds[i])

                if j == 0 or not labels_only_on_edge:
                    ax.set_ylabel(col_i)
                if j == 0 or not ticks_only_on_edge:
                    ax.set_yticklabels(["%.2f" % b for b in bounds[i]])
                else:
                    ax.set_yticklabels([None for b in bounds[i]])

                if n.prod(bounds[j]) < 0:
                    ax.axvline(0, color="k", alpha=0.25)
                if n.prod(bounds[i]) < 0:
                    ax.axhline(0, color="k", alpha=0.25)
    plt.tight_layout()
    plt.show()
    return f, axs


def plot_response_projections(
    trial_resps,
    rep_means,
    axes_pairs,
    subplot_shape,
    figsize=(6, 6),
    dpi=150,
    size_big=30,
    size_small=15,
    cmap="cet_glasbey_dark",
):
    nstim, ncell = rep_means.shape
    f, axs = plt.subplots(*subplot_shape, figsize=figsize, dpi=dpi, layout="constrained")
    if subplot_shape[0] == 1:
        axs = [axs]
    if subplot_shape[1] == 1:
        for i in range(len(axs)):
            axs[i] = [axs[i]]
    try:
        cmap = plt.get_cmap(cmap, lut=nstim)
    except ValueError:
        cmap = plt.get_cmap("tab20", lut=nstim)
    for i in range(len(axs)):
        for j in range(len(axs[0])):
            ax = axs[i][j]
            idx = i * len(axs[0]) + j
            axes = axes_pairs[idx]
            for stimidx in range(nstim):
                ax.scatter(
                    trial_resps[stimidx, :, axes[0]],
                    trial_resps[stimidx, :, axes[1]],
                    s=size_small,
                    alpha=0.5,
                    color=cmap(stimidx),
                    edgecolor=None,
                    linewidth=0,
                )
                ax.scatter(
                    rep_means[stimidx, axes[0]],
                    rep_means[stimidx, axes[1]],
                    s=size_big,
                    color=cmap(stimidx),
                    edgecolor="k",
                    linewidth=0.5,
                )

            ylims = ax.get_ylim()
            xlims = ax.get_xlim()
            ax.axvline(0, color="k", alpha=0.1)
            ax.axhline(0, color="k", alpha=0.1)
            ax.set_ylim(*ylims)
            ax.set_xlim(*xlims)
            #         plt.show()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("PC%02d" % axes[0])
            ax.set_ylabel("PC%02d" % axes[1])
    # plt.tight_layout()
    return f, axs


def scatter(
    xs,
    ys,
    size=5,
    alpha=0.5,
    color=None,
    marker_params={},
    regression=True,
    regression_line_params={},
    ax=None,
    figsize=(5, 5),
    dpi=150,
    binned=False,
    binned_params={},
    label=None,
    binned_line_params={},
    binned_area_params={},
    identity=False,
):
    if ax is None:
        f, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if len(xs.shape) > 1:
        xs = utils.flatten_lower_tri(xs)
        ys = utils.flatten_lower_tri(ys)

    if identity:
        lo = min(n.min(xs), n.min(ys))
        hi = max(n.max(xs), n.max(ys))
        ax.plot([lo, hi], [lo, hi], color="gray", ls="--", lw=1, zorder=-10)
        ax.set_aspect("equal")

    ax.scatter(xs, ys, s=size, c=color, alpha=alpha, linewidth=0,label=label, **marker_params)
    if regression:
        slopex, interceptx, rx, px, __ = stats.linregress(xs, ys)
        xline = n.array([xs.min(), xs.max()])
        ax.plot(
            xline,
            xline * slopex + interceptx,
            color="k",
            label=f"R:{rx:.2f}",
            **regression_line_params,
        )

    if binned:
        n_bins = binned_params.get("n_bins", 20)
        x_edges = binned_params.get("x_edges", None)
        xx, yy = utils.bin_by_coord(
            xs,
            ys,
            n_bins=n_bins,
            bins=x_edges,
            mean_bins=True,
            std_bins=True,
            shift_bins=True,
        )
        ax.plot(xx, yy[:, 0], **binned_line_params)
        ax.fill_between(
            xx,
            yy[:, 0] - yy[:, 1],
            yy[:, 0] + yy[:, 1],
            **binned_area_params,
        )


def plot_embedding_with_colors(
    embedding,
    colors,
    labels=None,
    cmap="viridis",
    bg_color="gainsboro",
    nrow=1,
    fsize=3,
    s=5,
    pminmax=(5, 95),
):
    n_plots = len(colors)
    ncol = n_plots // nrow
    if n_plots % nrow > 0:
        ncol += 1
    f, axs = plt.subplots(nrow, ncol, figsize=(ncol * fsize, nrow * fsize))
    if nrow == 1:
        axs = [axs]
    if ncol == 1:
        for i in range(len(axs)):
            axs[i] = [axs[i]]
    for i in range(nrow):
        for j in range(ncol):
            idx = i * ncol + j
            if idx >= n_plots:
                axs[i][j].axis("off")
                continue
            ax = axs[i][j]
            vmin = n.percentile(colors[idx], pminmax[0])
            vmax = n.percentile(colors[idx], pminmax[1])
            sc = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=colors[idx],
                cmap=cmap,
                s=s,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if labels is not None:
                ax.set_title(labels[idx])
            # plot a colorbar that will be inset in the scatter plot axis, instead of appearing beside
            # print("insetting")
            cax = ax.inset_axes([0.8, 0.2, 0.03, 0.6])  # [x0, y0, width, height]
            cbar = plt.colorbar(sc, cax=cax, orientation="vertical")
            # Set colorbar ticks to min, max, and zero (if zero is between min and max)
            cmin, cmax = vmin, vmax
            ticks = [cmin, cmax]
            if cmin < 0 < cmax:
                ticks.insert(1, 0)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{tick:.1f}" for tick in ticks])
            ax.set_facecolor(bg_color)
    return f, axs


def plot_cell_img(image, med, square_pix=10, **kwargs):
    square_pix = int(square_pix)
    if len(image.shape) == 3:
        image = image[med[0]]
        med = med[1:]
    # if transpose:
    #     print(image.shape)

    #     xx, yy = med
    #     image = image.T

    yy, xx = med
    ny, nx = image.shape
    # print(square_pix)

    xmin = max(xx - square_pix, 0)
    xmin_lim = (xx - xmin) - square_pix
    xmax = min(xx + square_pix, nx - 1)
    xmax_lim = 2 * square_pix + xmin_lim

    ymin = max(yy - square_pix, 0)
    ymin_lim = (yy - ymin) - square_pix
    ymax = min(yy + square_pix, ny - 1)
    ymax_lim = 2 * square_pix + ymin_lim
    # print(med)
    # print(ymin, ymax, xmin, xmax)
    # print(ymin_lim, ymax_lim, xmin_lim, xmax_lim)

    crop = image[ymin:ymax, xmin:xmax]

    # print(kwargs)
    f, ax, __ = show_img(crop, **kwargs)
    ax.scatter(
        [square_pix + xmin_lim],
        [square_pix + ymin_lim],
        marker="o",
        color="none",
        edgecolor="red",
        s=100,
        linewidth=2,
    )
    ax.set_ylim(ymin_lim, ymax_lim)
    ax.set_xlim(xmin_lim, xmax_lim)
    # return crop

    return f, ax

def diverging_cmap(low_color="blue", high_color="red", mid_color="white", name="diverging", nan_color="lightgrey"):
    """
    Creates a diverging colormap with a specified color at the center (zero value)
    and a color for NaN values.

    Args:
        low_color (str): Color for the low end of the colormap.
        high_color (str): Color for the high end of the colormap.
        mid_color (str): Color for the center (zero) of the colormap.
        name (str): Name of the colormap.
        nan_color (str): Color to display for NaN values.

    Returns:
        matplotlib.colors.LinearSegmentedColormap: The created colormap.
    """
    cdict = {
        "red": [
            (0.0, mcolors.to_rgb(low_color)[0], mcolors.to_rgb(low_color)[0]),
            (0.5, mcolors.to_rgb(mid_color)[0], mcolors.to_rgb(mid_color)[0]),
            (1.0, mcolors.to_rgb(high_color)[0], mcolors.to_rgb(high_color)[0]),
        ],
        "green": [
            (0.0, mcolors.to_rgb(low_color)[1], mcolors.to_rgb(low_color)[1]),
            (0.5, mcolors.to_rgb(mid_color)[1], mcolors.to_rgb(mid_color)[1]),
            (1.0, mcolors.to_rgb(high_color)[1], mcolors.to_rgb(high_color)[1]),
        ],
        "blue": [
            (0.0, mcolors.to_rgb(low_color)[2], mcolors.to_rgb(low_color)[2]),
            (0.5, mcolors.to_rgb(mid_color)[2], mcolors.to_rgb(mid_color)[2]),
            (1.0, mcolors.to_rgb(high_color)[2], mcolors.to_rgb(high_color)[2]),
        ],
    }
    cmap = mcolors.LinearSegmentedColormap(name, cdict)
    cmap.set_bad(nan_color)
    return cmap


def plot_spatial_corr_dist(
    corr_by_dist,
    bins_xy,
    bins_z,
    reductions,
    figsize=(8, 5),
    ylabel="",
    mean_over_z=True,
    noz=False,
    colors=("mediumblue", "darkred", "silver"),
):

    percentiles = []
    for i in reductions:
        if type(i) != str:
            percentiles.append(i)
    percentiles = n.sort(percentiles)

    # cbins_z = (bins_z[1:] + bins_z[:-1]) / 2
    cbins_xy = (bins_xy[1:] + bins_xy[:-1]) / 2

    f, axs = plt.subplots(1, 2, figsize=(8, 5), layout="constrained")
    ax = axs[0]

    n_percentiles = len(percentiles)
    cmap = diverging_cmap(*colors)
    for i, percentile in enumerate(percentiles):
        xs = cbins_xy[1:]
        if noz:
            to_plot = corr_by_dist[percentile]  # [2:-2]
            xs = cbins_xy
            print(to_plot.shape)
            print(xs.shape)
        elif mean_over_z:
            to_plot = corr_by_dist[percentile][1:-1, 2:-1].mean(axis=0)
        else:
            to_plot = corr_by_dist[percentile][1:-1, 2:-1][0]
        ax.plot(
            xs,
            to_plot,
            color=cmap(i / n_percentiles),
            label=percentile,
        )
    ax.set_xlabel("xy-distance")
    ax.set_ylabel(ylabel)

    ax = axs[1]
    for i, percentile in enumerate(percentiles):
        if percentile < 5 or percentile > 95:
            continue
        if noz:
            to_plot = corr_by_dist[percentile][2:-2]
            # print(to_plot.shape)
        elif mean_over_z:
            to_plot = corr_by_dist[percentile][1:-1, 2:-1].mean(axis=0)
        else:
            to_plot = corr_by_dist[percentile][1:-1, 2:-1][0]
        ax.plot(
            cbins_xy[1:],
            to_plot,
            color=cmap(i / n_percentiles),
            label=percentile,
        )
    ax.set_xlabel("xy-distance")
    ax.set_ylabel(ylabel)

    return f, axs

class CategoricalColormap(mcolors.Colormap):
    # untested code from deepseek
    def __init__(self, colors_rgba, categories, name="categorical_cmap"):
        super().__init__(name=name, N=len(categories))
        self.lookup = []
        for cat, color in zip(categories, colors_rgba):
            if isinstance(cat, float) and n.isnan(cat):
                # Check for NaN
                cond = lambda X, c=cat: n.isnan(X)
            else:
                # Check for exact match, including inf
                cond = lambda X, c=cat: X == c
            self.lookup.append((cond, color))

    def __call__(self, X, alpha=None, bytes=False):
        X = n.asarray(X)
        rgba = n.zeros(X.shape + (4,), dtype=n.float32)
        unassigned = n.ones(X.shape, dtype=bool)

        for condition, color in self.lookup:
            mask = condition(X) & unassigned
            rgba[mask] = color
            unassigned &= ~mask

        if alpha is not None:
            rgba[..., 3] = alpha

        if bytes:
            rgba = (rgba * 255).astype(n.uint8)

        return rgba.squeeze()


def categorical_cmap(colors, categories):
    colors_rgba = [mcolors.to_rgba(c) for c in colors]
    return CategoricalColormap(colors_rgba, categories)

def save_gif_from_timeseries(
    arr,
    gif_path,
    framerate=10,
    cmap="Greys_r",
    vminmax_percentile=(0.5, 99.5),
    vminmax=None,
    symmetric_cmap=False,
    show_img_kwargs=None,
    dpi=100,
    close_fig=True,
    decimate=None,
    dt=1,
):
    """
    Plot each frame of a 3D array (nt, ny, nx) using show_img and save as a GIF.
    Optionally decimate (filter+downsample) along the time axis by 'decimate'.
    Args:
        arr: np.ndarray, shape (nt, ny, nx)
        gif_path: str, output path for gif
        framerate: int, frames per second
        cmap: str, matplotlib colormap
        vminmax_percentile: tuple, percentiles for vmin/vmax if vminmax is None
        vminmax: tuple, (vmin, vmax) for all frames (overrides percentile)
        symmetric_cmap: bool, use symmetric color scaling
        show_img_kwargs: dict, extra kwargs for show_img
        dpi: int, figure dpi
        close_fig: bool, whether to close each figure after saving
        decimate: int or None, if set, filter and decimate along time axis by this factor
        dt: float, time step between frames (before decimation)
    """
    import numpy as np
    from scipy.signal import decimate as scipy_decimate

    arr = np.asarray(arr)
    nt = arr.shape[0]
    if show_img_kwargs is None:
        show_img_kwargs = {}
    # Decimate if requested
    if cmap == "RdBu_r" or cmap == "RdBu":
        symmetric_cmap = True
    if decimate is not None and decimate > 1:
        # Decimate along time axis for each (y, x)
        nt_dec = int(np.ceil(arr.shape[0] / decimate))
        arr_dec = np.zeros((nt_dec, arr.shape[1], arr.shape[2]), dtype=arr.dtype)
        for y in range(arr.shape[1]):
            for x in range(arr.shape[2]):
                arr_dec[:, y, x] = scipy_decimate(arr[:, y, x], decimate, ftype="fir", zero_phase=True, axis=0)
        arr = arr_dec
        nt = arr.shape[0]
        dt = dt * decimate
    # Compute vmin/vmax if not given
    if vminmax is None:
        vmin = np.nanpercentile(arr, vminmax_percentile[0])
        vmax = np.nanpercentile(arr, vminmax_percentile[1])
        vminmax = (vmin, vmax)
        # if symmetric cmap:
        if symmetric_cmap:
            vmax_abs = max(abs(vmin), abs(vmax))
            vminmax = (-vmax_abs, vmax_abs)

    images = []
    for t in range(nt):
        f, ax, axim = show_img(arr[t], cmap=cmap, vminmax=vminmax, return_fig=True, dpi=dpi, **show_img_kwargs)
        ax.set_title(f"t = {t * dt:.2f}")
        f.canvas.draw()
        img = np.frombuffer(f.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(f.canvas.get_width_height()[::-1] + (3,))
        images.append(img)
        if close_fig:
            import matplotlib.pyplot as plt

            plt.close(f)
    duration = 1.0 / framerate
    import imageio

    imageio.mimsave(gif_path, images, duration=duration)
    print(f"Saved gif to {gif_path} ({nt} frames, {framerate} fps, dt={dt})")


def save_cortical_movie(
    coords,
    values,
    suite3d_shape,
    savedir,
    pix_weights=None,
    frame_indices=None,
    step=1,
    vmin=None,
    vmax=None,
    cmap=None,
    nan_color="white",
    frame_prefix='frame_',
    transpose=True,
    make_gif=False,
    gif_name="out.gif",
    fps=10,
    save_timestamps=False,
    ts_framerate=None,
    ts_fontsize=18,
    ts_color="k",
    ts_bg="white",
    # dpi=None,
):
    """
    Fast saver: convert per-cell values (n_cells x n_frames) into 2D images by projecting
    cell pixel coords to (y,x) and computing the mean per pixel using bincount.

    Args:
        coords (list): list of length n_cells where each entry is a tuple/list of arrays
            (z_coords, y_coords, x_coords). Only y and x are used.
        values (ndarray): shape (n_cells, n_frames) or (n_cells,) for single-frame repeated use.
        suite3d_shape (tuple): (z,y,x) or (y,x) to infer ny,nx.
        savedir (path-like): directory to create and write PNG frames into.
        frame_indices (iterable or None): indices of frames to write. If None and values has 2 dims,
            uses np.arange(0, values.shape[1], step).
        step (int): step between source frames when building frame_indices (used when frame_indices is None).
        vmin, vmax (float|None): color limits; if None are inferred from values percentiles.
        cmap (matplotlib cmap or str or None): colormap to use. If None, uses a white->darkred linear cmap.
        nan_color (str): color to use for NaN pixels.
        transpose (bool): whether to transpose images before colormapping (matches previous code's .T usage).
        make_gif (bool): if True, also create a GIF named gif_name in savedir after writing PNGs.
        gif_name (str): filename for GIF when make_gif True.
        fps (int): frames per second for GIF.

    Returns:
        list of file paths written (in order). If make_gif True, also returns the gif path as second element.
    """
    import numpy as _np
    import imageio as _imageio
    from pathlib import Path

    savedir = Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    # infer ny,nx from suite3d_shape
    if len(suite3d_shape) == 3:
        ny, nx = int(suite3d_shape[1]), int(suite3d_shape[2])
    elif len(suite3d_shape) == 2:
        ny, nx = int(suite3d_shape[0]), int(suite3d_shape[1])
    else:
        ny, nx = int(suite3d_shape[-2]), int(suite3d_shape[-1])

    # precompute pixel -> cell mapping
    pix_idx_list = []
    cell_id_list = []
    pix_weights_list = []
    for cell_idx, coord in enumerate(coords):
        ys = _np.asarray(coord[1], dtype=_np.int64)
        xs = _np.asarray(coord[2], dtype=_np.int64)
        if ys.size == 0:
            continue
        lin = ys * nx + xs
        pix_idx_list.append(lin)
        cell_id_list.append(_np.full(lin.shape, cell_idx, dtype=_np.int32))
        # handle optional per-pixel weights for this cell
        if pix_weights is not None:
            try:
                w = _np.asarray(pix_weights[cell_idx], dtype=float)
            except Exception:
                raise ValueError("pix_weights must be an indexable sequence with one array per cell in coords")
            if w.shape[0] != lin.shape[0]:
                raise ValueError(f"pix_weights[{cell_idx}] length {w.shape[0]} does not match number of pixels {lin.shape[0]} for coords[{cell_idx}]")
            pix_weights_list.append(w)
        else:
            # placeholder; will default to ones when concatenated
            pix_weights_list.append(_np.ones(lin.shape, dtype=float))

    if len(pix_idx_list) == 0:
        raise ValueError("No pixel coordinates found in coords; nothing to save.")

    pix_idx = _np.concatenate(pix_idx_list)
    cell_id = _np.concatenate(cell_id_list)
    pixel_weights = _np.concatenate(pix_weights_list) if len(pix_weights_list) > 0 else None
    n_pixels = ny * nx
    # print(pixel_weights)

    # frame indices to loop
    vals = _np.asarray(values)
    if frame_indices is None:
        if vals.ndim == 1:
            frame_indices = [0]
        else:
            frame_indices = _np.arange(0, vals.shape[1], step)
    else:
        frame_indices = list(frame_indices)

    # prepare colormap and norm
    if cmap is None:
        cmap = linear_cmap("whitesmoke", "darkred", nan_color=nan_color)[0]
    elif isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # ensure cmap handles bad values
    try:
        cmap = cmap.with_extremes(bad=nan_color)
    except Exception:
        try:
            cmap.set_bad(nan_color)
        except Exception:
            pass

    # infer vmin/vmax if not provided
    if vmin is None or vmax is None:
        if vals.ndim == 2:
            vmin = _np.nanpercentile(vals, 0.5) if vmin is None else vmin
            vmax = _np.nanpercentile(vals, 99.5) if vmax is None else vmax
        else:
            vmin = 0.0 if vmin is None else vmin
            vmax = 1.0 if vmax is None else vmax

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    written = []
    frames_for_gif = []
    for i_idx, fi in enumerate(frame_indices):
        if vals.ndim == 1:
            frame_vals = vals
        else:
            frame_vals = vals[:, int(fi)]

        # handle NaN cell values: don't count them
        valid_cells = ~_np.isnan(frame_vals)
        values_for_pixels = frame_vals[cell_id]
        valid_for_pixels = valid_cells[cell_id].astype(_np.int32)
        # apply per-pixel weights if provided; compute weighted mean per pixel
        if pixel_weights is None:
            weights_for_pixels = _np.ones_like(values_for_pixels, dtype=float)
        else:
            weights_for_pixels = pixel_weights

        # avoid NaNs in values_for_pixels by zeroing entries for invalid cells
        vals_pix = values_for_pixels.copy()
        invalid_mask = valid_for_pixels == 0
        if invalid_mask.any():
            vals_pix[invalid_mask] = 0.0

        # Zero-out weights for invalid (NaN) cell values so they don't contribute
        effective_weights = weights_for_pixels * valid_for_pixels.astype(float)
        # print(weights_for_pixels)

        sum_per_pixel = _np.bincount(pix_idx, weights=vals_pix * weights_for_pixels, minlength=n_pixels)
        sum_weights_per_pixel = _np.bincount(pix_idx, weights=None, minlength=n_pixels)

        mean_flat = _np.full(n_pixels, _np.nan, dtype=float)
        mask = sum_weights_per_pixel > 0
        mean_flat[mask] = sum_per_pixel[mask] / sum_weights_per_pixel[mask]
        img = mean_flat.reshape(ny, nx)

        # img[img==0] = _np.nan  # optional: treat zero as NaN for better visualization

        # prepare RGBA 8-bit image for safe PNG writing
        arr = img.T if transpose else img
        masked = _np.ma.masked_invalid(arr)
        rgba = cmap(norm(masked))
        rgba8 = (rgba * 255).astype(_np.uint8)

        out_path = savedir / f"{frame_prefix}{i_idx:06d}.png"
        _imageio.imwrite(str(out_path), rgba8)
        written.append(str(out_path))
        if make_gif:
            frames_for_gif.append(rgba8)
        # optional: save timestamp image for this frame
        if save_timestamps:
            if ts_framerate is None:
                raise ValueError("ts_framerate must be provided when save_timestamps=True")
            # compute time in seconds using the actual frame index fi
            time_s = float(fi) / float(ts_framerate)
            # format like 't = 025.3 s' (zero-padded width before decimal)
            timestr = f"t = {time_s:06.1f} s"
            # create a small PNG with the text using PIL
            try:
                from PIL import Image, ImageDraw, ImageFont

                # estimate size
                font = ImageFont.load_default()
                # try to use a larger truetype font if available
                try:
                    font = ImageFont.truetype("arial.ttf", ts_fontsize)
                except Exception:
                    try:
                        font = ImageFont.truetype("DejaVuSans.ttf", ts_fontsize)
                    except Exception:
                        font = ImageFont.load_default()

                # compute text size robustly across Pillow versions
                dummy_img = Image.new("RGBA", (1, 1))
                draw = ImageDraw.Draw(dummy_img)
                tw = th = None
                try:
                    # Pillow >= 8.0: textbbox
                    bbox = draw.textbbox((0, 0), timestr, font=font)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                except Exception:
                    try:
                        # ImageFont has getsize in many versions
                        tw, th = font.getsize(timestr)
                    except Exception:
                        try:
                            # older fallback
                            tw, th = draw.textsize(timestr, font=font)
                        except Exception:
                            tw, th = (200, 24)
                pad = 6
                img_ts = Image.new("RGBA", (tw + pad * 2, th + pad * 2), color=ts_bg)
                draw = ImageDraw.Draw(img_ts)
                draw.text((pad, pad), timestr, font=font, fill=ts_color)
                ts_path = savedir / f"{frame_prefix}timestamp_{i_idx:06d}.png"
                img_ts.save(str(ts_path))
            except Exception:
                # fallback: write a tiny matplotlib figure (slower)
                fig, ax = plt.subplots(figsize=(tw / 100.0 + 0.1, th / 100.0 + 0.1))
                ax.text(0.5, 0.5, timestr, ha="center", va="center", fontsize=ts_fontsize, color=ts_color)
                ax.axis("off")
                fig.tight_layout(pad=0)
                fig.canvas.draw()
                arr = _np.frombuffer(fig.canvas.tostring_rgb(), dtype=_np.uint8)
                arr = arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                ts_path = savedir / f"timestamp_{i_idx:06d}.png"
                _imageio.imwrite(str(ts_path), arr)
                plt.close(fig)

    if make_gif:
        gif_path = savedir / gif_name
        _imageio.mimsave(str(gif_path), frames_for_gif, fps=fps)
        return written, str(gif_path)

    return written


def save_tiled_gif_from_timeseries(
    arr_list,
    gif_path,
    grid_shape,
    framerate=10,
    cmap="Greys_r",
    vminmax_percentile=(0.5, 99.5),
    vminmax=None,
    symmetric_cmap=False,
    show_img_kwargs=None,
    dpi=100,
    close_fig=True,
    decimate=None,
    dt=1,
    titles=None,
):
    """
    Plot each frame of multiple 3D arrays (nt, ny, nx) in a tiled grid and save as a single GIF.
    Args:
        arr_list: list/array of np.ndarray, each shape (nt, ny, nx)
        gif_path: str, output path for gif
        grid_shape: tuple, (n_rows, n_cols)
        framerate: int, frames per second
        cmap: str, matplotlib colormap
        vminmax_percentile: tuple, percentiles for vmin/vmax if vminmax is None
        vminmax: tuple, (vmin, vmax) for all frames (overrides percentile)
        symmetric_cmap: bool, use symmetric color scaling
        show_img_kwargs: dict, extra kwargs for show_img
        dpi: int, figure dpi
        close_fig: bool, whether to close each figure after saving
        decimate: int or None, if set, filter and decimate along time axis by this factor
        dt: float, time step between frames (before decimation)
        titles: list of str or None, optional titles for each subplot
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import decimate as scipy_decimate
    import imageio

    arr_list = [np.asarray(arr) for arr in arr_list]
    n_movies = len(arr_list)
    n_rows, n_cols = grid_shape
    assert n_movies <= n_rows * n_cols, "More movies than grid slots"
    nt = min(arr.shape[0] for arr in arr_list)
    # Decimate if requested
    if decimate is not None and decimate > 1:
        arr_list_dec = []
        for arr in arr_list:
            nt_dec = int(np.ceil(arr.shape[0] / decimate))
            arr_dec = np.zeros((nt_dec, arr.shape[1], arr.shape[2]), dtype=arr.dtype)
            for y in range(arr.shape[1]):
                for x in range(arr.shape[2]):
                    arr_dec[:, y, x] = scipy_decimate(arr[:, y, x], decimate, ftype="fir", zero_phase=True, axis=0)
            arr_list_dec.append(arr_dec)
        arr_list = arr_list_dec
        nt = min(arr.shape[0] for arr in arr_list)
        dt = dt * decimate
    # Compute vmin/vmax if not given
    if vminmax is None:
        all_vals = np.concatenate([arr[:nt].ravel() for arr in arr_list])
        vmin = np.nanpercentile(all_vals, vminmax_percentile[0])
        vmax = np.nanpercentile(all_vals, vminmax_percentile[1])
        vminmax = (vmin, vmax)
        if symmetric_cmap:
            vmax_abs = max(abs(vmin), abs(vmax))
            vminmax = (-vmax_abs, vmax_abs)
    if show_img_kwargs is None:
        show_img_kwargs = {}
    images = []
    for t in range(nt):
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3), dpi=dpi)
        axs = np.array(axs).reshape(n_rows, n_cols)
        for idx, arr in enumerate(arr_list):
            row = idx // n_cols
            col = idx % n_cols
            # print(arr.shape)
            ax = axs[row, col]
            show_img(arr[t], cmap=cmap, vminmax=vminmax, return_fig=False, ax=ax, **show_img_kwargs)
            if titles is not None and idx < len(titles):
                ax.set_title(titles[idx])
            ax.set_xticks([])
            ax.set_yticks([])
        # Hide unused axes
        for idx in range(n_movies, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axs[row, col].axis("off")

        # If a raster is to be shown below, draw the red line at the correct time index
        # (This is a placeholder for future raster support in the GIF version)
        # If you add a raster subplot, use the following logic:
        # if decimate is not None and decimate > 1:
        #     t_raster = int(round(t * decimate))
        # else:
        #     t_raster = t
        # raster_ax.axvline(t_raster, color="red", lw=2, alpha=0.7)

        fig.suptitle(f"t = {t * dt:.2f}")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(img)
        if close_fig:
            plt.close(fig)
    duration = 1.0 / framerate
    imageio.mimsave(gif_path, images, duration=duration)
    print(f"Saved tiled gif to {gif_path} ({nt} frames, {framerate} fps, dt={dt})")


def save_tiled_mp4_with_raster_from_timeseries(
    arr_list,
    raster_arr,
    mp4_path,
    grid_shape,
    framerate=10,
    cmap="Greys_r",
    vminmax_percentile=(0.5, 99.5),
    vminmax=None,
    symmetric_cmap=False,
    show_img_kwargs=None,
    dpi=100,
    close_fig=True,
    decimate=None,
    dt=1,
    titles=None,
    raster_show_img_kwargs=None,
    raster_height=2,
    raster_pad=0.5,
):
    """
    Like save_tiled_gif_with_raster_from_timeseries, but saves as MP4 using ffmpeg.
    Args:
        arr_list: list/array of np.ndarray, each shape (nt, ny, nx)
        raster_arr: np.ndarray, shape (Ny, Nt) (e.g. a raster or heatmap)
        mp4_path: str, output path for mp4
        grid_shape: tuple, (n_rows, n_cols)
        framerate: int, frames per second
        cmap: str, matplotlib colormap for movies
        vminmax_percentile: tuple, percentiles for vmin/vmax if vminmax is None
        vminmax: tuple, (vmin, vmax) for all frames (overrides percentile)
        symmetric_cmap: bool, use symmetric color scaling for movies
        show_img_kwargs: dict, extra kwargs for show_img for the raster
        dpi: int, figure dpi
        close_fig: bool, whether to close each figure after saving
        decimate: int or None, if set, filter and decimate along time axis by this factor
        dt: float, time step between frames (before decimation)
        titles: list of str or None, optional titles for each subplot
        raster_show_img_kwargs: dict, kwargs for show_img for the raster
        raster_height: float, height of the raster subplot (inches)
        raster_pad: float, vertical space between movies and raster (inches)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import decimate as scipy_decimate
    import imageio

    arr_list = [np.asarray(arr) for arr in arr_list]
    n_movies = len(arr_list)
    n_rows, n_cols = grid_shape
    assert n_movies <= n_rows * n_cols, "More movies than grid slots"
    nt = min(arr.shape[0] for arr in arr_list)
    # Decimate if requested
    if decimate is not None and decimate > 1:
        arr_list_dec = []
        for arr in arr_list:
            nt_dec = int(np.ceil(arr.shape[0] / decimate))
            arr_dec = np.zeros((nt_dec, arr.shape[1], arr.shape[2]), dtype=arr.dtype)
            for y in range(arr.shape[1]):
                for x in range(arr.shape[2]):
                    arr_dec[:, y, x] = scipy_decimate(arr[:, y, x], decimate, ftype="fir", zero_phase=True, axis=0)
            arr_list_dec.append(arr_dec)
        arr_list = arr_list_dec
        nt = min(arr.shape[0] for arr in arr_list)
        dt = dt * decimate
    # Compute vmin/vmax if not given
    if vminmax is None:
        all_vals = np.concatenate([arr[:nt].ravel() for arr in arr_list])
        vmin = np.nanpercentile(all_vals, vminmax_percentile[0])
        vmax = np.nanpercentile(all_vals, vminmax_percentile[1])
        vminmax = (vmin, vmax)
        if symmetric_cmap:
            vmax_abs = max(abs(vmin), abs(vmax))
            vminmax = (-vmax_abs, vmax_abs)
    if show_img_kwargs is None:
        show_img_kwargs = {}
    if raster_show_img_kwargs is None:
        raster_show_img_kwargs = {}
    # Figure layout: movies in grid, raster below (using gridspec)
    fig_height = (n_rows + 1) * 3 + raster_pad
    fig_width = n_cols * 3
    import matplotlib.gridspec as gridspec

    with imageio.get_writer(mp4_path, fps=framerate, codec="libx264", format="ffmpeg") as writer:
        for t in range(nt):
            fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
            gs = gridspec.GridSpec(n_rows + 1, n_cols, height_ratios=[3] * n_rows + [3], hspace=0.3)
            axs = np.empty((n_rows, n_cols), dtype=object)
            for idx, arr in enumerate(arr_list):
                row = idx // n_cols
                col = idx % n_cols
                ax = fig.add_subplot(gs[row, col])
                axs[row, col] = ax
                show_img(arr[t], cmap=cmap, vminmax=vminmax, return_fig=False, ax=ax, **show_img_kwargs)
                if titles is not None and idx < len(titles):
                    ax.set_title(titles[idx])
                ax.set_xticks([])
                ax.set_yticks([])
            # Hide unused axes
            for idx in range(n_movies, n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                fig.add_subplot(gs[row, col]).axis("off")
            # Raster image (bottom row, spanning all columns)
            raster_ax = fig.add_subplot(gs[-1, :])
            Ny, Nt = raster_arr.shape
            raster_show_img_kwargs = dict(raster_show_img_kwargs)  # copy
            raster_show_img_kwargs.setdefault("aspect", "auto")
            raster_show_img_kwargs.setdefault("extent", [-0.5, Nt - 0.5, Ny - 0.5, -0.5])
            show_img(raster_arr, ax=raster_ax, return_fig=False, **raster_show_img_kwargs)
            # Draw red line at correct time index in raster
            if decimate is not None and decimate > 1:
                t_raster = int(round(t * decimate))
            else:
                t_raster = t
            raster_ax.axvline(t_raster, color="red", lw=2, alpha=0.7)
            raster_ax.set_xlim(-0.5, Nt - 0.5)
            raster_ax.set_ylim(Ny - 0.5, -0.5)
            raster_ax.set_aspect("auto")
            raster_ax.set_title("Raster (vertical line = current frame)")
            fig.suptitle(f"t = {t * dt:.2f}")
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(img)
            if close_fig:
                plt.close(fig)
    print(f"Saved tiled mp4 with raster to {mp4_path} ({nt} frames, {framerate} fps, dt={dt})")


def save_tiled_gif_with_raster_from_timeseries(
    arr_list,
    raster_arr,
    gif_path,
    grid_shape,
    framerate=10,
    cmap="Greys_r",
    vminmax_percentile=(0.5, 99.5),
    vminmax=None,
    symmetric_cmap=False,
    show_img_kwargs=None,
    dpi=100,
    close_fig=True,
    decimate=None,
    dt=1,
    titles=None,
    raster_show_img_kwargs=None,
    raster_height=2,
    raster_pad=0.5,
):
    """
    Like save_tiled_mp4_with_raster_from_timeseries, but saves as GIF.
    Args:
        arr_list: list/array of np.ndarray, each shape (nt, ny, nx)
        raster_arr: np.ndarray, shape (Ny, Nt) (e.g. a raster or heatmap)
        gif_path: str, output path for gif
        grid_shape: tuple, (n_rows, n_cols)
        framerate: int, frames per second
        cmap: str, matplotlib colormap for movies
        vminmax_percentile: tuple, percentiles for vmin/vmax if vminmax is None
        vminmax: tuple, (vmin, vmax) for all frames (overrides percentile)
        symmetric_cmap: bool, use symmetric color scaling for movies
        show_img_kwargs: dict, extra kwargs for show_img (movies)
        dpi: int, figure dpi
        close_fig: bool, whether to close each figure after saving
        decimate: int or None, if set, filter and decimate along time axis by this factor
        dt: float, time step between frames (before decimation)
        titles: list of str or None, optional titles for each subplot
        raster_show_img_kwargs: dict, kwargs for show_img for the raster
        raster_height: float, height of the raster subplot (inches)
        raster_pad: float, vertical space between movies and raster (inches)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import decimate as scipy_decimate
    import imageio

    arr_list = [np.asarray(arr) for arr in arr_list]
    n_movies = len(arr_list)
    n_rows, n_cols = grid_shape
    assert n_movies <= n_rows * n_cols, "More movies than grid slots"
    nt = min(arr.shape[0] for arr in arr_list)
    # Decimate if requested
    if decimate is not None and decimate > 1:
        arr_list_dec = []
        for arr in arr_list:
            nt_dec = int(np.ceil(arr.shape[0] / decimate))
            arr_dec = np.zeros((nt_dec, arr.shape[1], arr.shape[2]), dtype=arr.dtype)
            for y in range(arr.shape[1]):
                for x in range(arr.shape[2]):
                    arr_dec[:, y, x] = scipy_decimate(arr[:, y, x], decimate, ftype="fir", zero_phase=True, axis=0)
            arr_list_dec.append(arr_dec)
        arr_list = arr_list_dec
        nt = min(arr.shape[0] for arr in arr_list)
        dt = dt * decimate
    # Compute vmin/vmax if not given
    if vminmax is None:
        all_vals = np.concatenate([arr[:nt].ravel() for arr in arr_list])
        vmin = np.nanpercentile(all_vals, vminmax_percentile[0])
        vmax = np.nanpercentile(all_vals, vminmax_percentile[1])
        vminmax = (vmin, vmax)
        if symmetric_cmap:
            vmax_abs = max(abs(vmin), abs(vmax))
            vminmax = (-vmax_abs, vmax_abs)
    if show_img_kwargs is None:
        show_img_kwargs = {}
    if raster_show_img_kwargs is None:
        raster_show_img_kwargs = {}
    # Figure layout: movies in grid, raster below (using gridspec)
    fig_height = (n_rows + 1) * 3 + raster_pad
    fig_width = n_cols * 3
    import matplotlib.gridspec as gridspec

    images = []
    for t in range(nt):
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
        gs = gridspec.GridSpec(n_rows + 1, n_cols, height_ratios=[3] * n_rows + [3], hspace=0.3)
        axs = np.empty((n_rows, n_cols), dtype=object)
        for idx, arr in enumerate(arr_list):
            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col])
            axs[row, col] = ax
            show_img(arr[t], cmap=cmap, vminmax=vminmax, return_fig=False, ax=ax, **show_img_kwargs)
            if titles is not None and idx < len(titles):
                ax.set_title(titles[idx])
            ax.set_xticks([])
            ax.set_yticks([])
        # Hide unused axes
        for idx in range(n_movies, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            fig.add_subplot(gs[row, col]).axis("off")
        # Raster image (bottom row, spanning all columns)
        raster_ax = fig.add_subplot(gs[-1, :])
        Ny, Nt = raster_arr.shape
        raster_show_img_kwargs = dict(raster_show_img_kwargs)  # copy
        raster_show_img_kwargs.setdefault("aspect", "auto")
        raster_show_img_kwargs.setdefault("extent", [-0.5, Nt - 0.5, Ny - 0.5, -0.5])
        show_img(raster_arr, ax=raster_ax, return_fig=False, **raster_show_img_kwargs)
        # Draw red line at correct time index in raster
        if decimate is not None and decimate > 1:
            t_raster = int(round(t * decimate))
        else:
            t_raster = t
        raster_ax.axvline(t_raster, color="red", lw=2, alpha=0.7)
        raster_ax.set_xlim(-0.5, Nt - 0.5)
        raster_ax.set_ylim(Ny - 0.5, -0.5)
        raster_ax.set_aspect("auto")
        raster_ax.set_title("Raster (vertical line = current frame)")
        fig.suptitle(f"t = {t * dt:.2f}")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(img)
        if close_fig:
            plt.close(fig)
    duration = 1.0 / framerate
    imageio.mimsave(gif_path, images, duration=duration)
    print(f"Saved tiled gif with raster to {gif_path} ({nt} frames, {framerate} fps, dt={dt})")


def plot_raster_and_curve(
    raster,
    curves,
    dt=1.0,
    bin=(1, 1),
    raster_size=(10, 5),
    vminmax=(0, 1),
    dpi=200,
    curve_height=0.5,
    curve_panel_min_height=0.5,
    curve_panel_max_height=6.0,
    raster_cmap="Greys",
    curve_kwargs={},
    raster_kwargs={},
    labels=None,
    colors=None,
    dy=3,
    ax_raster=None,
    ax_curve=None,
    return_fig=True,
    interp="nearest",
    tick_locs=None,
    nospine=True,
    lw=1.0,
    scaler=1.0,  # <-- new argument
):
    """
    Plot a raster (ncell, ntime) and curves (ncurves, ntime) in a two-panel figure.
    The raster panel is exactly raster_size in inches. The curves panel height is proportional to ncurves.
    Args:
        raster: (ncell, ntime)
        curves: (ncurves, ntime)
        dt: sampling interval (seconds per sample)
        bin: (by, bx) binning for raster (integers)
        raster_size: (width, height) in inches for raster panel
        dpi: figure dpi
        curve_height: height in inches per curve (default 0.5)
        curve_panel_min_height: minimum height for curve panel (inches)
        curve_panel_max_height: maximum height for curve panel (inches)
        raster_cmap: colormap for raster
        curve_kwargs: dict, extra kwargs for multiple_timeseries
        raster_kwargs: dict, extra kwargs for plt.imshow
        ax_raster, ax_curve: optional axes to plot into
        return_fig: if True, return (fig, (ax_raster, ax_curve))
    Returns:
        fig, (ax_raster, ax_curve) if return_fig else None
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from .plot import multiple_timeseries

    raster = np.asarray(raster)
    curves = np.asarray(curves)
    ncell, ntime = raster.shape
    print(ncell)
    ncurves = curves.shape[0]
    # Bin raster if needed
    by, bx = bin
    if by > 1:
        ncell_binned = ncell // by
        raster = raster[: ncell_binned * by].reshape(ncell_binned, by, ntime).mean(axis=1)
        # ncell = raster.shape[0]
    if bx > 1:
        ntime_binned = ntime // bx
        raster = raster[:, : ntime_binned * bx].reshape(raster.shape[0], ntime_binned, bx).mean(axis=2)
        ntime = raster.shape[1]
        dt = dt * bx
    # Figure size: raster panel is raster_size, curve panel height is proportional to ncurves
    curve_panel_height = min(max(curve_height * ncurves, curve_panel_min_height), curve_panel_max_height)
    fig_height = raster_size[1] + curve_panel_height
    fig_width = raster_size[0]
    if ax_raster is None or ax_curve is None:
        fig = plt.figure(figsize=(fig_width * scaler, fig_height * scaler), dpi=dpi)
        gs = fig.add_gridspec(2, 1, height_ratios=[raster_size[1], curve_panel_height], hspace=0.15)
        ax_raster = fig.add_subplot(gs[0, 0])
        ax_curve = fig.add_subplot(gs[1, 0], sharex=ax_raster)
    else:
        fig = ax_raster.figure
    # Plot raster using plt.imshow directly
    if raster_kwargs is None:
        raster_kwargs = {}
    print(raster.shape)
    # interpolation = 'nearest'
    im = ax_raster.imshow(
        raster,
        cmap=raster_cmap,
        aspect="auto",
        interpolation=interp,
        origin="upper",
        vmin=vminmax[0] if vminmax is not None else None,
        vmax=vminmax[1] if vminmax is not None else None,
        **raster_kwargs,
    )
    ax_raster.set_xlim(-0.5, raster.shape[1] - 0.5)
    ax_raster.set_ylim(raster.shape[0] - 0.5, -0.5)
    ax_raster.set_ylabel("Cell")
    ax_raster.set_yticks([0, raster.shape[0]], [0, ncell - 1])
    ax_raster.set_xticks([])
    if nospine:
        for ax in ax_raster, ax_curve:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
    # Plot curves (do not bin curves, just scale x axis appropriately)
    if curve_kwargs is None:
        curve_kwargs = {}
    # Adjust x scale so curves match raster length after binning
    times = np.linspace(0, (raster.shape[1] - 1) * dt / bx, curves.shape[1])
    # print(curves.shape[1])
    # print(dt)
    # print(raster.shape)
    multiple_timeseries(
        n.arange(curves.shape[1]) / bx,
        [curves[i] for i in range(ncurves)],
        ax=ax_curve,
        zscore=True,
        dy=dy,
        lw=lw,
        colors=colors,
        labels=labels,
        **curve_kwargs,
    )
    # Set x-tick labels at user-specified fractions or default locations
    ntime = raster.shape[1]
    t0 = 0
    tmax = (ntime - 1) * dt
    if tick_locs is not None:
        # tick_locs: list/array of fractions between 0 and 1
        xticks = [int(loc * (ntime - 1)) for loc in tick_locs]
        xticklabels = [f"{int(loc * tmax)}" for loc in tick_locs]
    else:
        tmax10 = int(tmax // 10) * 10
        thalf = tmax10 / 2
        xticks = [0, int(thalf // dt), int(tmax10 // dt)]
        xticklabels = [f"0", f"{int(thalf)}", f"{int(tmax10)}"]
    ax_curve.set_xticks(xticks)
    ax_curve.set_xticklabels(xticklabels)

    ax_curve.set_xlabel("Time (s)")
    # ax_curve.set_ylabel("Curves")
    # ax_curve.set_xlim(0, times[-1])
    plt.tight_layout()
    if return_fig:
        return fig, (ax_raster, ax_curve)


def plot_polygon_from_outline(outline, color="k", lw=1, ls=None, alpha=1.0, ax=None):
    """
    Plot a polygon from an outline (list of (x, y) tuples).

    Args:
        outline: list of (x, y) tuples defining the polygon outline
        color: color of the polygon
        lw: line width
        ls: line style (default is solid)
        alpha: transparency level
        ax: matplotlib Axes object to plot on; if None, uses current Axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    if ax is None:
        ax = plt.gca()

    polygon = Polygon(outline, closed=True, edgecolor=color, facecolor="none", lw=lw, ls=ls, alpha=alpha)
    ax.add_patch(polygon)
    return polygon


def plot_percentiles(xs, ys, percentiles, stds=None, cmap=None, ax=None, legend=False, **kwargs):
    """
    Plot multiple percentiles as lines on the same axes.

    Args:
        xs: x values (common for all percentiles)
        ys: list of y values for each percentile
        percentiles: list of percentile values (e.g., [5, 25, 50, 75, 95])
        stds: list of standard deviations for each percentile (same shape as ys); if provided, adds shaded error bars
        cmap:
        ax: matplotlib Axes object to plot on; if None, uses current Axes
        **kwargs: additional keyword arguments for plt.plot
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()

    if cmap is None:
        cmap = diverging_cmap()

    for i, (p, y) in enumerate(zip(percentiles, ys)):
        color = cmap(i / len(percentiles))
        ax.plot(xs, y, label=f"{p}th", color=color, **kwargs)

        # Add shaded error bars if stds is provided
        if stds is not None:
            std = stds[i]
            ax.fill_between(xs, np.array(y) - std, np.array(y) + std, color=color, alpha=0.2, linewidth=0)

    if legend:
        ax.legend()


def pairplot(
    data,
    *,
    nc=None,
    labels=None,
    dim_labels=None,
    axs=None,
    square_axes=False,
    share_limits=False,
    percentile_limit=None,
    density_threshold=100,
    **density_scatter_kwargs,
):
    """Pairwise scatter / histogram matrix.

    Diagonal panels show histograms; off-diagonal panels show scatter plots
    (coloured by density when n_points > density_threshold).

    Parameters
    ----------
    data : ndarray (n_samples, n_dims) or list of such arrays
        One or more datasets to overlay.  Each must have the same number of
        columns (dimensions).
    nc : int, optional
        How many dimensions to include.  Defaults to all columns.
    labels : list of str, optional
        One label per dataset in *data* (used in legend).
    dim_labels : list of str, optional
        Axis labels for each dimension.  Defaults to '0', '1', ...
    axs : 2-D array of Axes, optional
        Pre-existing nc x nc Axes grid.  Created if not supplied.
    square_axes : bool
        If True, each off-diagonal panel uses the same numerical range on
        both axes (the union of the two dimensions' ranges).
    share_limits : bool
        If True, every panel shares one global limit (implies square_axes).
    percentile_limit : float or None
        None  -> axis limits = data range +/- 5% padding.
        Value in (50, 100] -> symmetric percentile clip, e.g. 90 gives the
        [10th, 90th] percentile window (passing 10 gives the same result).
    density_threshold : int
        Use density_scatter when a dataset exceeds this many points.
    **density_scatter_kwargs
        Forwarded to density_scatter for dense panels.

    Returns
    -------
    axs : ndarray of Axes, shape (nc, nc)
    """
    # normalise input
    if isinstance(data, np.ndarray):
        series_list = [data]
    else:
        series_list = [np.asarray(d) for d in data]

    if nc is None:
        nc = series_list[0].shape[1]
    series_list = [s[:, :nc] for s in series_list]

    n_series = len(series_list)
    if labels is None:
        labels = [None] * n_series
    if dim_labels is None:
        dim_labels = [str(i) for i in range(nc)]

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = [default_colors[i % len(default_colors)] for i in range(n_series)]

    # axis limits
    if share_limits:
        square_axes = True
        global_lim = _pairplot_lim(
            np.concatenate([s.ravel() for s in series_list]), percentile_limit
        )
        dim_lims = [global_lim] * nc
    else:
        dim_lims = [
            _pairplot_lim(
                np.concatenate([s[:, i] for s in series_list]), percentile_limit
            )
            for i in range(nc)
        ]

    # create axes
    if axs is None:
        fig, axs = plt.subplots(nc, nc, figsize=(2 * nc, 2 * nc))
        plt.tight_layout(pad=0.5)
    axs = np.atleast_2d(axs)

    for i in range(nc):
        for j in range(nc):
            ax = axs[i, j]

            if i == j:
                xlim = dim_lims[i]
                for s, lab, col in zip(series_list, labels, colors):
                    kw = dict(bins=30, alpha=0.6, color=col, range=xlim)
                    if lab is not None:
                        kw['label'] = lab
                    ax.hist(s[:, i], **kw)
                ax.set_xlim(xlim)

            else:
                # scatter: dim j on x-axis, dim i on y-axis
                if square_axes:
                    lo = min(dim_lims[i][0], dim_lims[j][0])
                    hi = max(dim_lims[i][1], dim_lims[j][1])
                    xlim = ylim = (lo, hi)
                else:
                    xlim = dim_lims[j]
                    ylim = dim_lims[i]

                for s, lab, col in zip(series_list, labels, colors):
                    xd, yd = s[:, j], s[:, i]
                    if len(xd) > density_threshold:
                        density_scatter(xd, yd, ax=ax, **density_scatter_kwargs)
                    else:
                        sc_kw = dict(s=5, alpha=0.6, color=col)
                        if lab is not None:
                            sc_kw['label'] = lab
                        ax.scatter(xd, yd, **sc_kw)

                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

            # edge labels only
            if i == nc - 1:
                ax.set_xlabel(dim_labels[j])
            else:
                ax.tick_params(labelbottom=False)
            if j == 0:
                ax.set_ylabel(dim_labels[i])
            else:
                ax.tick_params(labelleft=False)

    if any(l is not None for l in labels):
        axs[0, nc - 1].legend(loc='upper right')

    return axs


def _pairplot_lim(data, percentile_limit):
    """Compute a (lo, hi) axis limit from a flat data array."""
    if percentile_limit is None:
        lo, hi = float(data.min()), float(data.max())
        pad = (hi - lo) * 0.05
        if pad == 0:
            pad = 0.5
        return (lo - pad, hi + pad)
    else:
        p = max(min(float(percentile_limit), 100.0), 50.0)
        lo = float(np.percentile(data, 100.0 - p))
        hi = float(np.percentile(data, p))
        return (lo, hi)
    return ax
