class Slice():
    """Class to store the information of a slice of particles.

    Attributes
    ----------
    name: (str)
        Name of the slice.
    n_inside_particles: (int)
        Number of particles inside the slice.
    n_outside_particles: (int)
        Number of particles outside the slice.
    normal: (np.ndarray)
        Normal vector of the slice.
    point: (np.ndarray)
        Point on the slice.
    thickness: (float)
        Thickness of the slice.
    """
    def __init__(self, name, n_inside_particles, n_outside_particles, 
                 normal, point, thickness):
        self.name = name
        self.n_inside_particles = n_inside_particles
        self.n_outside_particles = n_outside_particles
        self.normal = normal
        self.point = point
        self.thickness = thickness
