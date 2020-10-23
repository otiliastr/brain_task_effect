import numpy as np

__author__ = 'Otilia Stretcu'


def normalize(data, axis, offset=None, scale=None, return_offset=False):
    """
    Normalizes the data along the provided axis.
    If offset and scale are provided, we compute (data-offset) / scale,
    otherwise the offset is the mean, and the scale is the std (i.e. we z-score).
    Args:
        data(np.ndarray or list of embedded lists): Array-like data.
        axis(int): Axis along which to normalize.
        offset(np.ndarray or list of embedded lists): It matches the
            shape of the data along the provided axis.
        scale(np.ndarray or list of embedded lists): It matches the
            shape of the data along the provided axis.
    Returns:
        Normalized data, and optionally the offset and scale.

    """
    if offset is None:
        offset = np.expand_dims(data.mean(axis=axis), axis=axis)
    if scale is None:
        scale = np.expand_dims(data.std(axis=axis), axis=axis)
        scale[scale == 0.0] = 1.0
    if return_offset:
        return (data - offset) / scale, offset, scale
    return (data - offset) / scale
