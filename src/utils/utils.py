import numpy as np


# Choose the nearest power of two greater than the most common height
def next_power_of_two(n):
    """Returns the nearest power of two greater than or equal to n."""
    return 2 ** int(np.ceil(np.log2(n)))
