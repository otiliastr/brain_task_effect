from .container_ops import pretty_print

__author__ = 'Otilia Stretcu'


class Parameters(dict):
    """A container for storing experiment the configuration.

    It is an extension of a Python dictionary that allows the parameters to be accessed using dots
    (e.g. params.learning_rate).
    It can also be saved using pickle and compared with another Parameters object by content, not
    by hash code."""
    def __init__(self, param_dict):
        super(Parameters, self).__init__()
        self.update(param_dict)

    def id(self):
        raise ValueError('Must implement a param config id.')

    def __eq__(self, other):
        """Override the default Equals behavior"""
        def _equal(param1, param2):
            if isinstance(param1, Parameters) and isinstance(param2, Parameters):
                return param1 == param2
            return pretty_print(param1) == pretty_print(param2)

        if isinstance(other, self.__class__):
            if not self.keys() == other.keys():
                return False
            same_params = all([_equal(self[k], other[k]) for k in self.keys()])
            return same_params
        return False

    def __ne__(self, other):
        """Define a non-equality test"""
        return not self.__eq__(other)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError

    def __setattr__(self, key, value):
        self[key] = value


