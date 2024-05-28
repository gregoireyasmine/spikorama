import numpy as np
from .filter import gaussian_filter
from raster import rasterize


def align_rasters(spike_times, starts, dt, length):
    """
    Transform an array of spike event times in a raster-like array
    :param spike_times: an array of spike timings
    :param starts: start times on which spikes should be aligned
    :param dt: timebin of the raster array
    :param length: duration of the raster
    :return: an array of aligned rasters of shape (len(starts), int(length/dt))
    """
    return np.stack([rasterize(spike_times, start, start+length, dt) for start in starts])


def align(values, starts, dt, length=8):
    """
    Take a time series and aligns its values from different onsets
    :param values: the time series to align
    :param starts: the array of onsets (in time units)
    :param dt: the time bin of the series
    :param length: the duration of the alignment
    :return: Aligned time series array of shape (len(starts), int(length/dt))
    """
    alignment = []
    for i in range(len(starts)):
        startbin = int(starts[i]/dt)
        stopbin = startbin + int(length/dt)
        if startbin > 0 and stopbin < len(values):
            alignment.append(values[startbin: stopbin])
    return np.stack(alignment)


def trial_average_fr(rasters, dt, smoothing_resolution=None):
    """
    Computes trial-averaged firing rate given an array of aligned rasters
    :param rasters: an array of aligned rasters, shape (num_trials, num_timebins)
    :param dt: the timebin of the rasters
    :param smoothing_resolution: if not None, indicates the std of the gaussian kernel used to smooth the firing rate
    :return: a 1-D time series corresponding to trial-averaged firing rate
    """
    if rasters.ndim == 1:
        big_raster = np.expand_dim(rasters, 0)
    else:
        big_raster = rasters.copy()
    firing_rate = big_raster.mean(axis=0)/dt
    if smoothing_resolution is not None:
        firing_rate = gaussian_filter(firing_rate, smoothing_resolution, dt)
    return firing_rate


def std_between_traces(rasters, dt, smoothing_resolution=None):
    if rasters.ndim == 1:
        big_raster = np.expand_dim(rasters, 0)
    else:
        big_raster = rasters.copy()
    standard_deviation = np.std(big_raster, 0)/dt
    if smoothing_resolution is not None:
        standard_deviation = gaussian_filter(standard_deviation, smoothing_resolution, dt)
    return standard_deviation



# TODO : implement correlograms, CV, ISI, Fano, etc
