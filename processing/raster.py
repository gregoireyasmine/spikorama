import numpy as np


def rasterize(spikes, start, stop, dt):
    """
    Transforms spike event times array in a raster-like array with given timebin and start/stop times
    :param spikes: an array of spike timings
    :param start: start time of the raster array
    :param stop: end time of the raster array
    :param dt: timebin of the raster array
    :return: a 1D raster where each values indicates if a spike append in the timebin
    """
    num_bins = int(np.round((stop - start) / dt))  # Rounds to closest integer
    spike_times = np.where((spikes > start) & (spikes < stop))[0]
    spike_bins = ((spikes[spike_times] - start) / dt).astype(int)
    raster = np.bincount(spike_bins, minlength=num_bins)
    return raster


