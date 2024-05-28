from scipy.ndimage import gaussian_filter1d
import numpy as np


def gaussian_filter(values: np.ndarray, deltat, dt):
    """
    A gaussian kernel moving average
    :param values: time series to filter
    :param deltat: std of the gaussian kernel in seconds
    :param dt: timebin
    :return: filtered copy of values
    """
    return gaussian_filter1d(values, axis=-1, sigma=deltat/dt)


def linear_filter(values: np.ndarray, deltat, dt):
    """
    A simple moving average
    :param values: one or multiple stacked time series to filter
    :param deltat: length of the moving average window in seconds
    :param dt: timebin
    :return: filtered copy of values
    """
    if values.ndim == 1:
        v = np.expand_dims(values, 0)
    else:
        v = values
    window_size = int(deltat / dt)
    smoothed = np.zeros_like(v)
    start_indices = np.maximum(np.arange(v.shape[1]) - window_size // 2, 0)
    end_indices = np.minimum(np.arange(v.shape[1]) + window_size // 2, v.shape[1] - 1)
    for i in range(v.shape[1]):
        smoothed[:, i] = np.mean(v[:, start_indices[i]:end_indices[i] + 1], axis=1)
    if values.ndim == 1:
        return smoothed[0]
    else:
        return smoothed


def alpha_filter(values: np.ndarray, deltat, dt):
    """
    A moving average ignoring future values and amortizing past values
    :param values: one or multiple stacked time series to filter
    :param deltat: time at which past values are amortized by 90%
    :param dt: timebin
    :return: filtered copy of values
    """
    alpha = 1 - 0.9**(1/(deltat/dt))  # this allows for an amortization of 90% at deltat
    if values.ndim == 1:
        v = values[None, :]
    else:
        v = values
    smoothed = np.zeros_like(v)
    smoothed[0] = v[0]
    for i in range(1, v.shape[1]):
        smoothed[i] = alpha * v[:, i] + (1 - alpha) * smoothed[i - 1]
    if values.ndim == 1:
        return smoothed[0]
    else:
        return smoothed


def downsample(values, old_dt, new_dt, axis=0):
    """
    Returns downsampled version of values with sampling frequency new_dt/old_dt of the input array along the specified axis.
    If old_dt > new_dt the function has no effect (no upsampling implemented).
    :param values: a N-D time series numpy array
    :param old_dt: the timebin of the time series
    :param new_dt: the desired timebin of the output
    :param axis: the axis along which to downsample (default is 0)
    :return: the downsamples array
    """
    if new_dt > old_dt:
        new_indices = np.arange(0, values.shape[axis], int(new_dt / old_dt))
        slices = [slice(None) if i != axis else new_indices for i in range(values.ndim)]
        downsampled = np.take(values, new_indices, axis=axis)
        return downsampled
    else:
        return values