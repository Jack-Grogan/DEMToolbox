import numpy as np

class Split():
    """Class to store the information of a split of particles.

    Attributes
    ----------
    split_array: (np.ndarray)
        2D Array of split particles. The first column is the particle
        id and the second column is the class of the particle.
    split_dimension: (str)
        Dimension of the split. Can be 'x', 'y', 'z', or 'r'.
    n_class_0: (int)
        Number of particles in class 0. 
    n_class_1: (int)
        Number of particles in class 1.
    """
    def __init__(self, split_array, split_dimension):
        self.split_array = split_array
        self.split_dimension = split_dimension
        self.n_class_0 = np.sum(split_array[:, 1] == 0)
        self.n_class_1 = np.sum(split_array[:, 1] == 1)