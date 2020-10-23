import numpy as np

__author__ = 'Otilia Stretcu'


def get_data_shape(data):
    if isinstance(data, np.ndarray):
        return data.shape
    if isinstance(data, (list, tuple)):
        len_data = len(data)
        if len_data == 0:
            return tuple([0,])
        len_inner = get_data_shape(data[0])
        return tuple([len_data] + list(len_inner))
    return tuple([1])


def slice_data(data, indexes, dim=0):
    if dim == -1:
        dim = len(get_data_shape(data)) - 1
    if isinstance(data, np.ndarray):
        slice_indexes = [slice(l) for l in data.shape]
        slice_indexes[dim] = [int(i) for i in indexes]
        return data[slice_indexes]
    elif isinstance(data, list):
        if dim == 0:
            return [data[i] for i in indexes]
        return [slice_data(elem, indexes, dim-1) for elem in data]
    elif isinstance(data, tuple):
        if dim == 0:
            return tuple([data[i] for i in indexes])
        return tuple([slice_data(elem, indexes, dim-1) for elem in data])
    raise ValueError('Unsupported data type %s.' % data.__class__)


def pretty_print(data, float_format='%10.8f'):
    if isinstance(data, dict):
        return ', '.join(['%s: %s' % (str(k), pretty_print(v))
                          for k, v in data.items()])
    if isinstance(data, list):
        return ', '.join([pretty_print(v) for v in data])
    if np.isscalar(data) and isinstance(data, float):
        return float_format % data
    if hasattr(data, '__name__'):
        return data.__name__
    return str(data)


def get_dim_size(data, dim):
    if dim == -1:
        dim = len(get_data_shape(data)) - 1
    assert dim >= 0
    if isinstance(data, np.ndarray):
        return data.shape[dim]
    if isinstance(data, (list, tuple)):
        if dim == 0:
            return len(data)
        return get_dim_size(data[0], dim - 1)
    raise TypeError('Unsupported data type %s encountered.' % type(data))


def get_num_dims(data):
    """
    Returns the number of dimensions of data container.
    Args:
        data(array-like object):
    Returns:
        Number of dimensions.
    """
    return len(get_data_shape(data))
