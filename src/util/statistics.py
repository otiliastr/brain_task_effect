from __future__ import absolute_import

import random

__author__ = 'Otilia Stretcu'

__all__ = ['sample_reservoir']


def sample_reservoir(generator, k, random_state=None):
    """
    Samples k items from a generator using reservior sampling.

    It goes through the elements in a generator in order, and iteratively decides on k elements to keep. The final
    k elements that are returned are sampled with equal probability from all elements in the generator.

    Args:
        generator(generator):    A sample generator.
        k(int):                  Number of samples to keep from the generator, sampled uniformly.
        random_state(int):       Seed or random state for the random number generator.

    Returns: a list containing k samples from the generator.

    """
    random.seed(a=random_state)
    sample = []
    i = 0
    for item in generator:
        if i < k:
            sample.append(item)
        else:
            j = random.randrange(0, i)
            if j < k:
                sample[j] = item
        i = i + 1
    return sample
