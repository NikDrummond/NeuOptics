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


class Columns:
    def __init__(self, columns=None):
        """
        Initialize the Columns container.

        Parameters:
            clusters (iterable, optional): An initial list of cluster objects.
        """
        # Using a dictionary for fast lookup by cluster ID
        self.columns = {}
        if columns:
            for cl in columns:
                self.add_column(cl)

    def add_column(self, column_obj):
        """
        Add a cluster object to the container.

        Parameters:
            cluster_obj (cluster): The cluster instance to add.
        """
        self.clusters[column_obj.ID] = column_obj

    def remove_cluster(self, column_id):
        """
        Remove a cluster from the container by its ID.

        Parameters:
            cluster_id: The unique identifier of the cluster to remove.
        """
        if column_id in self.columns:
            del self.clusters[column_id]
        else:
            print(f"Cluster with ID {column_id} not found.")

    def get_column(self, column_id):
        """
        Retrieve a cluster by its ID.

        Parameters:
            cluster_id: The unique identifier of the cluster.

        Returns:
            The cluster instance if found, else None.
        """
        return self.clusters.get(column_id, None)

    def all_centroids(self, as_array:bool = False) -> dict | np.ndarray:
        """
        Compute the centroid for each cluster.

        Returns:
            A dictionary mapping each cluster's ID to its centroid.
        """
        if as_array:
            cents = np.array([
            cluster_obj.centroid()
            for cluster_id, cluster_obj in self.clusters.items()
            ])
        else:
            cents = {
            cluster_id: cluster_obj.centroid()
            for cluster_id, cluster_obj in self.clusters.items()
        }
        return cents

    def __iter__(self):
        """
        Allow iteration over the cluster objects.
        """
        return iter(self.columns.values())

    def __len__(self):
        """
        Return the number of clusters.
        """
        return len(self.columns)