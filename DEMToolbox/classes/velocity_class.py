class VelocityField():
    """ Class to store velocity field data.

    Attributes
    ----------
    mesh_column: (str)
        Column name of the 2D mesh data.
    slice_column: (str)
        Column name of the slice data.
    velocity_column: (str)
        Column name of the velocity data.
    velocity_vectors: (np.ndarray)
        Array of velocity vectors.
    velocity_magnitude: (np.ndarray)
        Array of velocity magnitudes
    """
    def __init__(self, mesh_column, slice_column, velocity_column,
                 velocity_vectors, velocity_magnitude):
        self.mesh_column = mesh_column
        self.slice_column = slice_column
        self.velocity_column = velocity_column
        self.velocity_vectors = velocity_vectors
        self.velocity_magnitude = velocity_magnitude
        