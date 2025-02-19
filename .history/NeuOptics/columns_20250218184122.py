
class Column:

    def __init__(self, ID, n_ids, coords, types, columns):
        self.ID = ID
        self.n_ids = n_ids
        self.coords = coords
        self.types = types
        self.columns = columns

    def centroid(self):
        """Get the center of mass fo points which make up this column"""
        return self.coords.mean(axis=0)


class Columns:
    def __init__(self, Columns=None):
        """
        Initialize the Columns container.

        Parameters:
            clusters (iterable, optional): An initial list of cluster objects.
        """
        # Using a dictionary for fast lookup by cluster ID
        self.clusters = {}
        if clusters:
            for cl in clusters:
                self.add_cluster(cl)

    def add_cluster(self, cluster_obj):
        """
        Add a cluster object to the container.

        Parameters:
            cluster_obj (cluster): The cluster instance to add.
        """
        self.clusters[cluster_obj.ID] = cluster_obj

    def remove_cluster(self, cluster_id):
        """
        Remove a cluster from the container by its ID.

        Parameters:
            cluster_id: The unique identifier of the cluster to remove.
        """
        if cluster_id in self.clusters:
            del self.clusters[cluster_id]
        else:
            print(f"Cluster with ID {cluster_id} not found.")

    def get_cluster(self, cluster_id):
        """
        Retrieve a cluster by its ID.

        Parameters:
            cluster_id: The unique identifier of the cluster.

        Returns:
            The cluster instance if found, else None.
        """
        return self.clusters.get(cluster_id, None)

    def all_centroids(self):
        """
        Compute the centroid for each cluster.

        Returns:
            A dictionary mapping each cluster's ID to its centroid.
        """
        return {
            cluster_id: cluster_obj.centroid()
            for cluster_id, cluster_obj in self.clusters.items()
        }

    def __iter__(self):
        """
        Allow iteration over the cluster objects.
        """
        return iter(self.clusters.values())

    def __len__(self):
        """
        Return the number of clusters.
        """
        return len(self.clusters)