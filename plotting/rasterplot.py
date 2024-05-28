import matplotlib.pyplot as plt
import numpy as np


def plot_raster(spikes, ax=None, s=10, linewidths=0.5, title=None):
    """
    Plots a raster from binary spike data

    Args:
        spikes (array-like): Binary spike data. Each row is a trial and each column is a timestep
        ax (matplotlib.axes.Axes, optional): Axis where to plot

    Returns:
        matplotlib.axes.Axes: Modified axis
    """
    if ax is None:
        fig, ax = plt.subplots()

    num_units, num_time_steps = spikes.shape
    spike_times = np.where(spikes)

    for unit in range(num_units):
        ax.scatter(spike_times[1][spike_times[0] == unit],
                   np.full(np.sum(spike_times[0] == unit), unit),
                   marker='|', color='k', linewidths=linewidths, s=s)

    ax.set_xlabel('Temps')
    ax.set_ylabel('Trial')
    if title is None:
        title = 'Raster Plot'
    ax.set_title(title)
    ax.set_ylim(-0.5, num_units - 0.5)
    ax.set_xlim(0, num_time_steps)
    ax.invert_yaxis()
    return ax
