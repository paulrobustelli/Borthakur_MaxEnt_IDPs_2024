
import numpy as np
import scipy
import itertools
from itertools import chain
from scipy.stats import gaussian_kde
from package.utils.utils import combinations
from functools import partial
import warnings
import matplotlib.pyplot as plt
import matplotlib
import os
import functools

def num_str(s, return_num=True, return_str=False):
    s = ''.join(filter(str.isdigit, s))

    if return_str and return_num:
        return s, int(s)

    if return_str:
        return s

    if return_num:
        return int(s)


def dKL(p, q, axis: "tuple or int" = None):
    # kl = p * np.log(p / q)
    # masked = np.ma.masked_array(kl, kl == np.nan)
    # p, q = common_nonzero([p, q])
    indices = np.prod(np.stack([i == 0. for i in [p, q]]), axis=0).astype(bool)
    p, q = [np.ma.masked_array(i, indices) for i in [p, q]]
    return np.sum(p * np.log(p / q), axis=axis).data


# def log(x):
#     return np.log(np.ma.masked_array(x, np.isclose(x, 0) | x == np.inf | x == np.nan))

def dJS(p, q, axis: "tuple or int" = None):
    m = 0.5 * (p + q)
    return 0.5 * (dKL(p, m, axis=axis) + dKL(q, m, axis=axis))


class DensityComparator():
    """
    Estimate and compare discrete (histogram) and continuous (kernel density) of coupled datasets.

    """

    def __init__(self, data: list, weights: list = None):

        self.bounds = None
        self.data_list = data
        self.weights_list = weights

    @property
    def data_list(self, array: bool = False):

        return self.data_list_

    @data_list.setter
    def data_list(self, x):

        assert isinstance(x, list), "data_list must be type list"

        assert all((isinstance(i, np.ndarray) for i in x)), "all data should be type np.ndarray"

        x = [i.squeeze() for i in x]

        assert len(set([i.shape[-1] for i in x])) == 1, "All data arrays must be the same dimension"

        # print(x[0].ndim)

        assert x[0].ndim == 2, "Arrays should be 2D"

        self.dim = x[0].shape[-1]

        self.n_datasets = len(x)

        self.data_list_ = x

        self.set_bounds()

        return


    @property
    def weights_list(self):
        return self.weights_list_

    @weights_list.setter
    def weights_list(self, x):

        if x is not None:

            x = [i.squeeze() for i in x]

            data = self.data_list

            for i, (d, w) in enumerate(zip(data, x)):
                assert len(d) == len(w), f"The number of data samples must match the number of weights : index {i}"

            self.weights_list_ = x

        else:
            self.weights_list_ = None

        return


    def set_bounds(self):

        assert self.data_list is not None, "Must have data_list_ attribute in order to estimate bounds"

        self.bounds = np.array([get_extrema(i) for i in np.concatenate(self.data_list).T])

        return

    def estimate_kde(self,
                     bins: int = 80,
                     norm: bool = True,
                     weight: bool = False,
                     bw_method=None):

        self.bins = bins

        if weight:
            assert self.weights_list is not None, "Must have weights list in order to estimate weighted KDE"

            kdes = [gaussian_kde(i.T, weights=j, bw_method=bw_method) for i, j in zip(self.data_list, self.weights_list)]

        else:
            kdes = [gaussian_kde(i.T, bw_method=bw_method) for i in self.data_list]

        self.kde_grid = product(*[np.linspace(i[0], i[1], bins) for i in self.bounds])

        setattr(self, "kdes_weighted" if weight else "kdes",
                [self.sample_kde(kde, self.kde_grid, norm=norm) for kde in kdes])

        return

    def estimate_hist(self, bins: int = 80, norm: bool = True, weight: bool = False):

        self.bins = bins

        if weight:
            assert self.weights_list is not None, "Must have weights list in order to estimate weighted KDE"

            hists = [pmf(i, bins=bins, weights=j, norm=norm, range=self.bounds) for i, j in
                     zip(self.data_list, self.weights_list)]


        else:
            hists = [pmf(i, bins=bins, norm=norm, range=self.bounds) for i in self.data_list]

        self.hist_bin_centers = [i[-1] for i in hists]
        self.hist_dtrajs = [i[2] for i in hists]
        setattr(self, "hists_weighted" if weight else "hists", [i[0] for i in hists])

        return

    @staticmethod
    def sample_kde(kde, bounds, norm: bool = True):
        sample = kde.pdf(bounds.T)
        return sample / sample.sum() if norm else sample

    @property
    def n_datasets(self):
        return self.n_datasets_

    @n_datasets.setter
    def n_datasets(self, x):
        assert isinstance(x, int), "Number of datasets should be an integer"
        self.n_datasets_ = x
        self.data_pairs = combinations(np.arange(x)).astype(int)
        return

    @staticmethod
    def cos_similarity(x, y, axis: "tuple of ints or int" = None):
        return np.sum(x * y, axis=axis) / np.sqrt(np.sum(x ** 2, axis=axis) * np.sum(y ** 2, axis=axis))

    @property
    def bins(self):
        return self.bins_

    @bins.setter
    def bins(self, x: int):
        if hasattr(self, "bins_"):
            if self.bins_ != x:
                warnings.warn(
                    f"Bins to use in densitiy estimators has already been set to {self.bins_}. Changing to {x}. Consider recomputing all densities")
        self.bins_ = x
        return

    def compare(self, attr: str, weight: bool = False, metric: callable = None,
                pairs: np.ndarray = None, weight0: bool = None, weight1: bool = None):

        pairs = self.data_pairs if pairs is None else pairs

        assert attr in ("hists", "kdes"), "Density to compare must be either 'kdes' or 'hists' regardless of weighting"

        if "hists" in attr:
            warnings.warn((
                              "Using densities defined by histograms in the computation of a comparision metric can cause counter intuitive results "
                              "because empty bins are masked out to prevent nans")
                          )

        if all(i is None for i in (weight0, weight1)):

            attr = attr + "_weighted" if weight else attr

            assert hasattr(self, attr), f"Class must have {attr} in order to compare"

            density = getattr(self, attr)

            d0, d1 = np.stack([density[i] for i in pairs[:, 0]]), np.stack([density[i] for i in pairs[:, 1]])


        else:

            assert all(i is not None for i in (
            weight0, weight1)), "Must specify weighting for both datasets if weighting is specified for either"

            densities = []

            for i in (weight0, weight1):
                attr_ = attr + "_weighted" if i else attr

                assert hasattr(self, attr_), f"Class must have {attr_} in order to compare"

                densities.append(getattr(self, attr_))

            d0, d1 = [np.stack([d[i] for i in p]) for p, d in zip(pairs.T, densities)]

        metric = partial(self.cos_similarity if metric is None else metric, axis=(1, 2) if d0.ndim > 2 else -1)

        return metric(d0, d1)

    def plot_kde(self,
                 weight: bool = False,
                 title: str = None,
                 dscrs: list = None,
                 dscr: str = None,
                 kwargs: dict = {}):

        assert hasattr(self, "kdes_weighted" if weight else "kdes"), "Must estimate KDEs before plotting"

        if dscrs is not None:
            assert len(dscrs) == self.n_datasets, "Number of labels must match the number of datasets"
        else:
            dscrs = self.n_datasets * [""]

        title = ("Weighted Kernel Densities" if weight else "Kernel Densities") if title is None else title

        if dscr is not None:
            title = f"{title} : {dscr}"

        density = getattr(self, "kdes_weighted" if weight else "kdes")

        args = dict(figsize=(6, 1.8), title_pad=1.11, sharex=True, sharey=True, cbar_label="Density")
        args.update(kwargs)

        subplots_proj2d(self.kde_grid, c=np.stack(density),
                        rows=1, cols=self.n_datasets,
                        dscrs=dscrs, title=title,
                        **args)
        return

    def plot_hist(self,
                  weight: bool = False,
                  title: str = None,
                  dscrs: list = None,
                  dscr: str = None,
                  kwargs: dict = {}):

        if weight:
            assert self.weights_list is not None, "Must provide weights for weighted histogram plot"

        if dscrs is not None:
            assert len(dscrs) == self.n_datasets, "Number of labels must match the number of datasets"
        else:
            dscrs = self.n_datasets * [""]

        weights = self.weights_list if self.weights_list is not None and weight else self.n_datasets * [None]

        title = ("Weighted Histogram Densities" if weight else "Histogram Densities") if title is None else title

        if dscr is not None:
            title = f"{title} : {dscr}"

        args = dict(figsize=(6, 1.8), title_pad=1.11, sharex=True, sharey=True)
        args.update(kwargs)

        subplots_fes2d(x=self.data_list,
                       cols=self.n_datasets,
                       title=f"Reweighted : {title}" if weight else title,
                       dscrs=dscrs,
                       weights_list=self.weights_list if weight else None,
                       rows=1,
                       extent=self.bounds,
                       **args)
        return



def lsdir(dir,
          keyword: "list or str" = None,
          exclude: "list or str" = None,
          match: callable = all,
          indexed: bool = False):
    """ full path version of os.listdir with files/directories in order

        dir ::: path to a directory (str), required
        keyword ::: filter out strings that DO NOT contain this/these substrings (list or str)=None
        exclude ::: filter out strings that DO contain this/these substrings (list or str)=None
        indexed ::: filter out strings that do not contain digits.
                    Is passed to sort_strs function (bool)=False"""

    if dir[-1] == "/":
        dir = dir[:-1]

    listed_dir = os.listdir(dir)

    listed_dir = filter_strs(listed_dir, keyword=keyword, exclude=exclude, match=match)

    #print(listed_dir)

    return [f"{dir}/{i}" for i in sort_strs(listed_dir, indexed=indexed)]


def filter_strs(strs: list,
                keyword: "list or str" = None,
                exclude: "list or str" = None,
                match: callable = all):
    if keyword is not None:
        strs = keyword_strs(strs, keyword=keyword, exclude=False, match=match)

    if exclude is not None:
        strs = keyword_strs(strs, keyword=exclude, exclude=True, match=match)

    return strs


def keyword_strs(strs: list,
                 keyword: "list or str",
                 exclude: bool = False,
                 match: callable = all):
    if isinstance(keyword, str):
        if exclude:
            filt = lambda string: keyword not in string

        else:
            filt = lambda string: keyword in string

    else:
        if exclude:
            filt = lambda string: match(kw not in string for kw in keyword)

        else:
            filt = lambda string: match(kw in string for kw in keyword)

    return list(filter(filt, strs))


def sort_strs(strs: list, max=False, indexed: bool = False):
    """ strs ::: a list or numpy array of strings.
        max ::: bool, to sort in terms of greatest index to smallest.
        indexed ::: bool, whether or not to filter out strings that don't contain digits.
                    if set to False and string list (strs) contains strings without a digit, function
                    will return unsorted string list (strs) as an alternative to throwing an error."""

    # we have to ensure that each str in strs contains a number otherwise we get an error
    assert len(strs) > 0, "List of strings is empty"
    check = np.vectorize(lambda s: any(map(str.isdigit, s)))

    if isinstance(strs, list):
        strs = np.array(strs)

    # the indexed option allows us to filter out strings that don't contain digits.
    ## This prevents an error
    if indexed:
        strs = strs[check(strs)]
        assert len(strs) > 0, "List of strings is empty after filtering strings without digits"

    # if indexed != True, then we don't filter the list of input strings and simply return it
    ##because an attempt to sort on indices (digits) that aren't present results in an error
    else:
        if not all(check(strs)):
            warnings.warn("Not all strings contain a number, returning unsorted input list to avoid throwing an error. "
                          "If you want to only consider strings that contain a digit, set indexed to True ")

            return strs

    get_index = np.vectorize(functools.partial(num_str, return_str=False, return_num=True))
    indices = get_index(strs).argsort()

    if max:
        return strs[np.flip(indices)]

    else:
        return strs[indices]



def pmf1d(x: np.ndarray,
          bins: int,
          weights: np.ndarray = None,
          norm: bool = True,
          range: tuple = None):
    count, edge = np.histogram(x, bins=bins, weights=weights, range=range)
    p = count / count.sum() if norm else count
    idx = np.digitize(x, edge[1:-1])
    pi = p.flatten()[idx]
    return p, pi, idx, edge[:-1] + np.diff(edge) / 2


def pmfdd(arrays: "a list of arrays or N,d numpy array",
          bins: int,
          weights: np.ndarray = None,
          norm: bool = True,
          range: tuple = None,
          statistic: str = None):
    """each array in arrays should be the same length"""

    if statistic is None:
        statistic = "count" if weights is None else "sum"

    if isinstance(arrays, list):
        assert all(isinstance(x, np.ndarray) for x in arrays), \
            "Must input a list of arrays"
        arrays = [i.flatten() for i in arrays]
        assert len({len(i) for i in arrays}) == 1, "arrays are not all the same length"
        arrays = np.stack(arrays, axis=1)
    else:
        assert isinstance(arrays, np.ndarray), \
            "Must input a list of arrays or a single N,d array"
        arrays = arrays.squeeze()

    count, edges, idx = scipy.stats.binned_statistic_dd(arrays,
                                                        values=weights,
                                                        statistic=statistic,
                                                        bins=bins,
                                                        expand_binnumbers=True,
                                                        range=range)


    if range is not None:
        idx = np.stack([np.digitize(value, edge[1:-1]) for edge, value in zip(edges, arrays.T)]) + 1

    idx = np.ravel_multi_index(idx - 1, tuple([bins for i in arrays.T]))

    p = count / count.sum() if norm else count
    pi = p.flatten()[idx]
    return p, pi, idx, (edge[:-1] + np.diff(edge) / 2 for edge in edges)


def pmf(x: "list of arrays or array",
        bins: int,
        weights: np.ndarray = None,
        norm: bool = True,
        range: tuple = None):
    """
    returns : p, pi, idx, bin_centers

    """
    if isinstance(x, np.ndarray):
        x = x.squeeze()
        if x.ndim > 1:
            return pmfdd(x, bins, weights, norm, range)
        else:
            return pmf1d(x, bins, weights, norm, range)
    if isinstance(x, list):
        if len(x) == 1:
            return pmf1d(x[0], bins, weights, norm, range)
        else:
            return pmfdd(x, bins, weights, norm, range)


def product(x: np.ndarray, y: np.ndarray):
    return np.asarray(list(itertools.product(x, y)))


def sample_array(arr: np.ndarray, n, figs: int = 1):
    N = len(arr)
    return arr[np.round(np.linspace(n, N - n, n)).astype(int)].round(figs)


def truncate_colormap(cmap:str, minval=0.0, maxval=1.0, n=100):
    cmap = plt.get_cmap(cmap)
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def fes2d(x: np.ndarray,
          y: np.ndarray = None,
          xlabel: str = None,
          ylabel: str = None,
          title: str = None,
          cbar: bool = True,
          cmap: str = "jet",
          vmax: float = None,
          vmin: float = None,
          cluster_centers=None,
          bins: int = 180,
          weights: np.ndarray = None,
          density: bool = False,
          extent: list = None,
          n_contours: int = 200,
          alpha_contours: float = 1,
          contour_lines: bool = False,
          alpha_lines: float = 0.6,
          scatter: bool = False,
          scatter_alpha: float = .2,
          scatter_cmap: str = "bone",
          scatter_size: float = 0.05,
          scatter_stride: int = 100,
          scatter_min: float = 0.2,
          scatter_max: float = 0.8,
          comp_type: str = None,
          mask: bool = True,
          font_scale: float = 1,
          cbar_shrink: float = 1,
          nxticks: int = 4,
          nyticks: int = 4,
          tick_decimals: int = 2,
          extend_border: float = 1e-5,
          hide_ax: bool = False,
          ax=None,
          aspect="auto",
          mask_thresh: float = 0,
          ):
    x, y = (np.squeeze(i) if i is not None else None for i in (x, y))

    if y is None:
        assert (x.ndim == 2) and (x.shape[-1] == 2), \
            ("Must provide 1d data vectors for x and y"
             "or provide x as a N,2 array with data vectors as columns")

        x, y = x.T

    if extent is None:
        extent = [[x.min() - extend_border, x.max() + extend_border],
                  [y.min() - extend_border, y.max() + extend_border]]

    counts, x_edges, y_edges = np.histogram2d(x, y, bins=bins, range=extent,
                                              weights=weights, density=density)

    xticks, yticks = (i[:-1] + np.diff(i) / 2 for i in (x_edges, y_edges))

    mask_index = counts <= mask_thresh

    if mask:
        counts = np.ma.masked_array(counts, mask_index)

    else:
        counts[mask_index] = counts[~mask_index].min()

    F = -np.log(counts)
    F += -F.min()

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(5, 3))

    # ax.margins(extend_border, tight=False)

    if extend_border == 0:
        ax.set_aspect(aspect=aspect)
        # extent = None

    s = ax.contourf(F.T,
                    n_contours,
                    cmap=cmap,
                    extent=tuple(chain(*extent)),  # if extent is not None and extend_border != 0 else None,
                    zorder=-1,
                    alpha=alpha_contours,
                    vmax=vmax,
                    vmin=vmin
                    )

    if contour_lines:
        ax.contour(s, colors="black", cmap=None, linewidths=1, alpha=alpha_lines)

    fmtr = lambda x, _: f"{x:.2f}"

    xlabel = f"{comp_type} 1" if (comp_type is not None) and (xlabel is None) else xlabel
    ylabel = f"{comp_type} 2" if (comp_type is not None) and (ylabel is None) else ylabel

    ax.set_title(title, size=14 * font_scale)

    ax.set_xlabel(xlabel, fontsize=14 * font_scale)
    ax.set_ylabel(ylabel, fontsize=14 * font_scale)

    # ax.set_xticks(np.linspace(nxticks, bins - nxticks, nxticks), sample_array(xticks,
    #                                                                           nxticks,
    #                                                                           figs=tick_decimals))
    # ax.set_yticks(np.linspace(nyticks, bins - nyticks, nyticks), sample_array(yticks,
    #                                                                           nyticks,
    #                                                                           figs=tick_decimals))
    # print(sample_array(yticks,
    #                                                                           nyticks,
    #                                                                           figs=tick_decimals))
    ax.tick_params(axis="x", labelsize=10 * font_scale)
    ax.tick_params(axis="y", labelsize=10 * font_scale)

    if extend_border != 0:
        ax.set_xlim(extent[0][0], extent[0][1])
        ax.set_ylim(extent[1][0], extent[1][1])

    ax.set_xticks(np.linspace(extent[0][0], extent[0][1], nxticks),
                  np.linspace(extent[0][0], extent[0][1], nxticks).round(tick_decimals))

    ax.set_yticks(np.linspace(extent[1][0], extent[1][1], nyticks),
                  np.linspace(extent[1][0], extent[1][1], nyticks).round(tick_decimals))

    # ax.set_xticklabels(sample_array(xticks,
    #                               nxticks,
    #                               figs=tick_decimals).astype(str))
    #
    # ax.set_yticklabels(sample_array(yticks,
    #                               nyticks,
    #                               figs=tick_decimals).astype(str))

    if cbar:
        cbar = plt.colorbar(s, ax=ax, format=fmtr,
                            shrink=cbar_shrink,
                            ticks=np.linspace(F.min(), F.max(), 4, endpoint=True))
        cbar.set_label("Free Energy / (kT)", size=14 * font_scale)
        cbar.ax.tick_params(labelsize=8)

    # ax.set_aspect(aspect=aspect, share=True)

    if scatter:
        c = F.flatten()[pmf([x, y], bins=bins)[2]]
        # ax1 = ax.twinx().twiny()
        # ax1.set_xticks([])
        # ax1.set_yticks([])
        ax.scatter(x[::scatter_stride], y[::scatter_stride],
                   cmap=truncate_colormap(scatter_cmap,
                                          minval=scatter_min,
                                          maxval=scatter_max,
                                          n=len(c[::scatter_stride])),
                   alpha=scatter_alpha,
                   c=c[::scatter_stride],
                   s=scatter_size)

        # ax1.autoscale_view()
        if hide_ax:
            pass
            # ax1.set_axis_off()
            # ax1.axis("off")
            # ax.set_frame_on(False)

    if cluster_centers is not None:
        # ax2 = ax.twinx().twiny()
        # ax2.set_xticks([])
        # ax2.set_yticks([])

        for j, i in enumerate(cluster_centers):
            ax.annotate(f"{j + 1}", [i[k] for k in range(2)],
                        color="black", size=str(10 * font_scale))
        if hide_ax:
            pass
            # ax2.set_axis_off()
            # ax2.axis("off")
            # ax2.set_frame_on(False)

    if hide_ax:
        ax.set_axis_off()
        ax.axis("off")
        plt.gca().set_frame_on(False)
    # ax.autoscale_view()

    return s


def get_extrema(x, extend: float = 0):
    return [x.min() - extend, x.max() + extend]


def subplots_fes2d(x: np.ndarray,
                   rows: int,
                   cols: int,
                   dscrs: list,
                   indices_list: list = None,
                   y: np.ndarray = None,
                   ylabel=None,
                   xlabel=None,
                   title: str = None,
                   title_pad: float = 1,
                   font_scale: float = .6,
                   cmap: str = "jet",
                   mask: bool = False,
                   mask_thresh: float = 0,
                   share_extent: bool = True,
                   sharex: bool = False,
                   sharey: bool = False,
                   n_contours: int = 200,
                   alpha_contours: float = 1,
                   contour_lines: bool = False,
                   alpha_lines: bool = 0.6,
                   bins: int = 100,
                   weights_list: list = None,
                   extend_border: float = 0,
                   density: bool = False,
                   scatter: bool = False,
                   scatter_alpha: float = .2,
                   scatter_cmap: str = "bone",
                   scatter_size: float = 0.05,
                   scatter_stride: int = 100,
                   scatter_min: float = 0.2,
                   scatter_max: float = 0.8,
                   figsize: tuple = (6, 5)):
    x = np.stack([x, y], -1) if y is not None else x

    indices_list = list(range(len(x))) if indices_list is None else indices_list

    extent = [get_extrema(i, extend_border) for i in x.T] if share_extent else None

    fig, axes = plt.subplots(rows, cols, sharey=sharey, sharex=sharex, figsize=figsize)

    if weights_list is None:
        for ax, indices, dscr in zip(axes.flat, indices_list, dscrs):
            s = fes2d(x[indices],
                      cbar=False,
                      cmap=cmap,
                      extent=extent,
                      mask=mask,
                      mask_thresh=mask_thresh,
                      bins=bins,
                      density=density,
                      n_contours=n_contours,
                      alpha_contours=alpha_contours,
                      contour_lines=contour_lines,
                      alpha_lines=alpha_lines,
                      title=dscr,
                      ax=ax,
                      font_scale=font_scale,
                      cbar_shrink=1,
                      extend_border=extend_border,
                      scatter=scatter,
                      scatter_alpha=scatter_alpha,
                      scatter_cmap=scatter_cmap,
                      scatter_size=scatter_size,
                      scatter_stride=scatter_stride,
                      scatter_min=scatter_min,
                      scatter_max=scatter_max,
                      )
    else:
        for ax, indices, dscr, weights in zip(axes.flat, indices_list, dscrs, weights_list):
            s = fes2d(x[indices],
                      cbar=False,
                      cmap=cmap,
                      extent=extent,
                      mask=mask,
                      mask_thresh=mask_thresh,
                      bins=bins,
                      density=density,
                      weights=weights,
                      n_contours=n_contours,
                      alpha_contours=alpha_contours,
                      contour_lines=contour_lines,
                      alpha_lines=alpha_lines,
                      title=dscr,
                      ax=ax,
                      font_scale=font_scale,
                      cbar_shrink=1,
                      extend_border=extend_border,
                      scatter=scatter,
                      scatter_alpha=scatter_alpha,
                      scatter_cmap=scatter_cmap,
                      scatter_size=scatter_size,
                      scatter_stride=scatter_stride,
                      scatter_min=scatter_min,
                      scatter_max=scatter_max,
                      )

    fig.subplots_adjust(right=1.05, top=.9)
    fmtr = lambda x, _: f"{x:.1f}"
    c0, c1 = s.get_clim()
    cbar = fig.colorbar(s,
                        format=fmtr,
                        orientation='vertical',
                        ax=axes.ravel().tolist(),
                        aspect=20,
                        pad=.03,
                        panchor=(1, .5),
                        ticks=np.linspace(c0, c1, 4, endpoint=True)
                        )

    cbar.ax.tick_params(labelsize=12 * font_scale)
    cbar.set_label("Free Energy / (kT)", size=14 * font_scale)
    fig.supylabel(ylabel)
    fig.supxlabel(xlabel)
    fig.suptitle(title, y=title_pad)
    return

def proj2d(x: np.ndarray,
           c: np.ndarray,
           y: np.ndarray = None,
           xlabel: str = None,
           ylabel: str = None,
           title: str = None,
           cbar: bool = True,
           cbar_label: str = None,
           cbar_labels: str = None,
           cmap: str = "jet",
           alpha: float = 1,
           cluster_centers: np.ndarray = None,
           center_font_color: str = "black",
           bins: int = 180,
           extent: list = None,
           comp_type: str = None,
           font_scale: float = 1,
           dot_size: float = 0.5,
           cbar_shrink: float = 1,
           nxticks: int = 4,
           nyticks: int = 4,
           tick_decimals: int=2,
           vmin: float = None,
           vmax: float = None,
           ax=None,
           aspect="auto",
           state_map: bool = False,
           ):
    x, y = (np.squeeze(i) if i is not None else None for i in (x, y))

    if y is None:
        assert (x.ndim == 2) and (x.shape[-1] == 2), \
            ("Must provide 1d data vectors for x and y"
             "or provide x as a N,2 array with data vectors as columns")
        x, y = x.T

    if extent is None:
        extent = [[x.min(), x.max()], [y.min(), y.max()]]

    counts, x_edges, y_edges = np.histogram2d(x, y, bins=bins, range=extent)

    xticks, yticks = (i[:-1] + np.diff(i) / 2 for i in (x_edges, y_edges))

    fmtr = lambda x, _: f"{x:.2f}"

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(5, 3))

    if cbar:
        if state_map:
            nstates = c.max() + 1
            color_list = getattr(plt.cm, cmap) if isinstance(cmap, str) else cmap
            boundaries = np.arange(nstates + 1).tolist()
            listed_colormap = matplotlib.colors.ListedColormap(
                [color_list(i) for i in range(color_list.N)])
            norm = matplotlib.colors.BoundaryNorm(boundaries, listed_colormap.N, clip=True)
            s = ax.scatter(x, y, c=c, s=dot_size, cmap=cmap, norm=norm, alpha=alpha)
            tick_locs = (np.arange(nstates) + 0.5)
            ticklabels = np.arange(1, nstates + 1).astype(str).tolist()\
                         if cbar_labels is None else cbar_labels
            cbar = plt.colorbar(s, ax=ax, format=fmtr, shrink=cbar_shrink, )
            cbar.set_label(label="State" if cbar_label is None else cbar_label,
                           size=12 * font_scale)
            cbar.set_ticks(tick_locs)
            cbar.set_ticklabels(ticklabels)

        else:
            s = ax.scatter(x, y, c=c, s=dot_size, cmap=cmap,
                           alpha=alpha, vmin=vmin, vmax=vmax)
            c0, c1 = s.get_clim()
            cbar = plt.colorbar(s, ax=ax, format=fmtr, shrink=cbar_shrink,
                                ticks=np.linspace(c0, c1, 4, endpoint=True))
            cbar.set_label(cbar_label, size=12 * font_scale)
        cbar.ax.tick_params(labelsize=9 * font_scale)

    else:
        s = ax.scatter(x, y, c=c, s=.5, cmap=cmap,
                       alpha=alpha, vmin=vmin, vmax=vmax)

    ax.set_aspect(aspect)

    xlabel = f"{comp_type} 1" if (comp_type is not None) and (xlabel is None) else xlabel
    ylabel = f"{comp_type} 2" if (comp_type is not None) and (ylabel is None) else ylabel

    ax.set_title(title, size=14 * font_scale)

    ax.set_xlabel(xlabel, fontsize=14 * font_scale)
    ax.set_ylabel(ylabel, fontsize=14 * font_scale)

    ax.set_xticks(np.linspace(xticks[nxticks], xticks[bins - nxticks], nxticks),
                  sample_array(xticks, nxticks, figs=tick_decimals))
    ax.set_yticks(np.linspace(yticks[nyticks], yticks[bins - nyticks], nyticks),
                  sample_array(yticks, nyticks, figs=tick_decimals))

    # ax.set_xticks(np.linspace(nxticks, bins - nxticks, nxticks), sample_array(xticks, nxticks))
    # ax.set_yticks(np.linspace(nyticks, bins - nyticks, nyticks), sample_array(yticks, nyticks))

    ax.tick_params(axis="x", labelsize=10 * font_scale)
    ax.tick_params(axis="y", labelsize=10 * font_scale)

    if cluster_centers is not None:
        for j, i in enumerate(cluster_centers):
            ax.annotate(f"{j + 1}", [i[k] for k in range(2)],
                         color=center_font_color, size=str(10 * font_scale))

    return s


def subplots_proj2d(x: np.ndarray,
                    c: np.ndarray,
                    rows: int,
                    cols: int,
                    dscrs: list,
                    indices_list: list = None,
                    cmap: str = "jet",
                    dot_size: float = 0.5,
                    y: np.ndarray = None,
                    ylabel=None,
                    xlabel=None,
                    title: str = None,
                    title_pad=1.05,
                    cbar_label: "str or list" = None,
                    font_scale: float = .6,
                    share_extent: bool = True,
                    sharey: bool = False,
                    sharex: bool = False,
                    vmin: float = None,
                    vmax: float = None,
                    bins: int = 100,
                    figsize: tuple = (6, 5),
                    ):
    x = np.stack([x, y], -1) if y is not None else x

    c = c.squeeze()

    if indices_list is None:
        if x.ndim == 3:
            indices_list = list(range(len(x)))
        else:
            assert x.ndim == 2, "x must be 2 or 3 dimensional"
            indices_list = len(c) * [None]

    if c.ndim == 2:
        assert c.shape[0] == len(indices_list), "If each plot has a different coloring, number must match number of datasets"
        color_indices = list(range(len(c)))
    else:
        assert c.ndim == 1, "c must be 1 or two dimensional"
        color_indices = len(indices_list) * [None]

    extent = list(map(get_extrema, x.T)) if share_extent else None

    fig, axes = plt.subplots(rows, cols, sharey=sharey, sharex=sharex, figsize=figsize)

    # if isinstance(cbar_label, str):
    #     cbar_label = len(indices_list) * [cbar_label]
    # else:
    #     assert isinstance(cbar_label, list) and (len(cbar_label) == len(indices_list))

    for ax, indices, color_index, dscr in zip(axes.flat, indices_list, color_indices, dscrs):
        s = proj2d(x[indices],
                   c=c[color_index],
                   cmap=cmap,
                   dot_size=dot_size,
                   cbar=False,
                   cbar_label=cbar_label,
                   extent=extent,
                   bins=bins,
                   title=dscr,
                   ax=ax,
                   font_scale=font_scale,
                   vmin=vmin,
                   vmax=vmax,
                   cbar_shrink=1)

    fig.subplots_adjust(right=1.05, top=.9)
    fmtr = lambda x, _: f"{x:.3f}"
    c0, c1 = c.min(), c.max()
    cbar = fig.colorbar(s,
                        format=fmtr,
                        orientation='vertical',
                        ax=axes.ravel().tolist(),
                        aspect=20,
                        pad=.03,
                        panchor=(1, .5),
                        ticks=np.linspace(c0, c1, 4, endpoint=True)
                        )

    cbar.ax.tick_params(labelsize=12 * font_scale)
    cbar.set_label(cbar_label, size=14 * font_scale)
    fig.supylabel(ylabel)
    fig.supxlabel(xlabel)
    fig.suptitle(title, y=title_pad)
    return


def make_symbols():
    unicharacters = ["\u03B1",
                     "\u03B2",
                     "\u03B3",
                     "\u03B4",
                     "\u03B5",
                     "\u03B6",
                     "\u03B7",
                     "\u03B8",
                     "\u03B9",
                     "\u03BA",
                     "\u03BB",
                     "\u03BC",
                     "\u03BD",
                     "\u03BE",
                     "\u03BF",
                     "\u03C0",
                     "\u03C1",
                     "\u03C2",
                     "\u03C3",
                     "\u03C4",
                     "\u03C5",
                     "\u03C6",
                     "\u03C7",
                     "\u03C8",
                     "\u03C9",
                     "\u00C5"]
    keys = "alpha,beta,gamma,delta,epsilon,zeta,eta,theta,iota,kappa,lambda,mu,nu,xi,omicron,pi,rho,final_sigma,sigma,tau,upsilon,phi,chi,psi,omega,angstrom"
    return dict(zip(keys.split(","), unicharacters))


symbols = make_symbols().__getitem__
