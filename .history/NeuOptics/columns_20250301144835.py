from numpy import ndarray, array
# columns class - add vmf back in later.
class Columns:
    """
    Represents a collection of spatially distributed columns with associated metadata.

    Attributes:
        Column_centres (np.ndarray): Coordinates of the column centres (shape: Nx3).
        Column_ids (np.ndarray): Unique identifiers for each column.
        Column_points (dict): Coordinates of points associated with each column.
        Column_types (dict): Types or categories assigned to each column.
        Column_N_ids (dict): IDs of neurons or entities contained in each column.
        Synapse_counts (dict): Number of synapses from each input id within column (in order of input ids)
    
    """

    def __init__(
        self,
        coords: ndarray,
        col_ids: ndarray,
        col_point_coords: dict,
        types_in_columns: dict,
        ids_in_columns: dict,
        Synapse_counts: dict
    ):
        """
        Initializes the Columns object with spatial and categorical data.

        Args:
            coords (np.ndarray): Array of shape (N, 3) representing column center coordinates.
            col_ids (np.ndarray): Unique column identifiers.
            col_point_coords (dict): Coordinates of points associated with columns.
            types_in_columns (dict): Types/categories of columns.
            ids_in_columns (dict): IDs of neurons or entities in each column.
            Synapse_counts (dict): Number of synapses from each input id within column (in order of input ids)

        """
        self.Column_centres = coords
        self.Column_ids = col_ids
        self.Column_points = col_point_coords
        self.Column_types = types_in_columns
        self.Column_N_ids = ids_in_columns
        self.Synapse_counts = Synapse_counts

    def update_centres(self):
        """Re-calculate the column centres
        """
        self.Column_centres = array([self.Column_points[i].mean(axis = 0) for i in self.Column_ids])

    def column_count(self) -> int:
        return