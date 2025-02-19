import numpy as np

class Column:

    def __init__(self, ID, neuron_ids = None, point_coordinates = None, neuron_types = None, columns = None):
        self.ID = ID
        self.neuron_ids = neuron_ids
        self.neuron_types = neuron_types
        self.point_coordinates = point_coordinates
        self.columns = columns

    def centroid(self):
        """Get the center of mass fo points which make up this column"""
        if self.coords is None:
            raise AttributeError('Column has no coordinates')
        else:
            return self.coords.mean(axis=0)


