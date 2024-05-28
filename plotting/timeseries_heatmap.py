import numpy as np
import matplotlib.pyplot as plt


def plot_time_series_heatmap(data, binsize, ax=None, cbar_title=None, title=None):
    """
    Plot a time series graph with time binning and a colormap.

    Args:
    - data : 2D array of shape (n_trials, n_timepoints) containing the time series data
    - binsize : size of the bins for time binning
    - ax : Matplotlib axis object, optional

    Returns:
    - No return value, plots the graph on the given axis or creates a new axis and displays the graph
    """

    times = np.arange(data.shape[1])     # Calculate the total time and the time points
    total_time = times[-1] - times[0]

    nb_bins = int(np.ceil(total_time / binsize))    # Determine the total number of bins

    bins = np.linspace(times[0], times[-1], nb_bins+1)   # Calculate the bin boundaries

    # Calculate the mean values of the bins for each trial
    mean_bin_values = np.array([[np.mean(time_series[np.logical_and(times >= start_bin, times < end_bin)]) for start_bin, end_bin in zip(bins[:-1], bins[1:])] for time_series in data])

    # Find the minimum and maximum values of the mean values
    min_value = np.nanmin(mean_bin_values)
    max_value = np.nanmax(mean_bin_values)

    # Create a colormap with the minimum and maximum values
    norm = plt.Normalize(min_value, max_value)

    # Check if an axis is provided
    if ax is None:
        # Create a new figure and a new axis
        fig, ax = plt.subplots()

    # Plot the bins with the colormap
    img = ax.imshow(mean_bin_values, aspect='auto', extent=[times[0], times[-1], 0, data.shape[0]], origin='lower', cmap=custom_cmap, norm=norm,
                    interpolation=None)

    # Add a color bar
    cb = plt.colorbar(img, ax=ax, orientation='vertical', label=cbar_title)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.5)  # Set the width of the colorbar border
    cb.ax.yaxis.set_tick_params(width=0.5)  # Set the thickness of the colorbar ticks

    # Add labels and a title
    ax.set_xlabel('Timestep', fontsize=10)
    ax.set_ylabel('Trial', fontsize=10)
    if title is not None:
        ax.set_title(title, fontsize=11)
    # Display the graph if no axis is provided
    if ax is None:
        plt.show()
