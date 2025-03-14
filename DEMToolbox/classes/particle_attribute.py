class ParticleAttribute():
    """Class to store particle attribute data
    
    Attributes
    ----------
    field: (str)
        Name of the field that the attribute is associated with.
    attribute: (str)
        Name of the attribute.
    data: (np.ndarray)
        2d Array of attribute data. The first column is the field
        value and the second column is the attribute value.
    """
    def __init__(self, field, attribute, data):
        self.field = field
        self.attribute = attribute
        self.data = data
