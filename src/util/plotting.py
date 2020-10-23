import os
import logging
import pandas
import seaborn as sns

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt


__author__ = 'Otilia Stretcu'


def plot_heatmap(data, annotate=True, fmt='.2f', output_path=None, output_filename='heatmap.png', title='',
                 row_names=None, col_names=None, xticklabels="auto", yticklabels="auto", xlabel='', ylabel='',
                 use_seaborn=False, interpolation='gaussian', clim=None):
    sns.set()
    fig = plt.figure(figsize=(15, 15))

    # Seaborn below is prettier, but takes longer to plot.
    if use_seaborn:
        if row_names is not None:
            if col_names is not None:
                data = pandas.DataFrame(data, index=row_names, columns=col_names)
            else:
                data = pandas.DataFrame(data, index=row_names)
        ax = sns.heatmap(data, annot=annotate, fmt=fmt, annot_kws={"size":8},
                         xticklabels=xticklabels, yticklabels=yticklabels)
    else:
        ax = plt.imshow(data, interpolation=interpolation, aspect='auto')
        plt.colorbar()
        if clim is not None:
            plt.clim(clim[0], clim[1])
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    if output_path:
        fig.savefig(os.path.join(output_path, output_filename))
    plt.close(fig)
    return ax


def plot_time_signal(data, output_path=None, output_filename='signal.png', title='',
                     figsize=(15, 15)):
    fig = plt.figure(figsize=figsize)
    ax = plt.plot(data)
    plt.title(title)
    if output_path:
        fig.savefig(os.path.join(output_path, output_filename))
    plt.close(fig)
    return ax


def plot_time_signal_all_channels(data,
                                  output_path=None,
                                  output_filename='signal.png',
                                  title=''):
    num_channels = data.shape[0]
    num_cols = 3
    num_rows = int(num_channels // num_cols) + (1 if num_channels % num_cols > 0 else 0)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 30))

    for ch in range(num_channels):
        row = int(ch // num_cols)
        col = int(ch % num_cols)
        ax = axes[row][col]
        ax.plot(data[ch])
        ax.set_title('sensor %d' % (ch + 1))

    plt.title(title)
    plt.tight_layout()

    if output_path:
        path = os.path.join(output_path, output_filename)
        fig.savefig(path)
        logging.info('Saved plot at: %s', path)
    plt.close(fig)
