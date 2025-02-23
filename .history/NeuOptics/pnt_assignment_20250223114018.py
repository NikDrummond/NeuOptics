from .columns import Columns
from GeoJax import normalise
from scipy.stats import vonmises_fisher as vmf
import numpy as np

def fit_vonMises_Fisher(cols:Columns, bind:bool = True) -> tuple[dict, dict] | None:
    """Fit a von_Mises Fisher distribution to each column

    Parameters
    ----------
    cols : Columns
        _description_
    bind : bool, optional
        If True add a mu and kappa attribute to cols. Both are dictionaries of col_id:mu or kappa. 
        If False will return the dictionaries directly. By default True


    Returns
    -------
    tuple[dict, dict] | None
        If bind is False, returns dictionary of column_id:mu and column_id:kappa (in this order!)
    """
    mus = dict()
    kappas = dict()
    for i in cols.Column_ids:
        coords = cols.Column_points[i]
        if coords.shape[0] == 1:
            kappas[i] = np.nan
            ar = np.empty(3)
            ar[:] = np.nan
            mus[i] = ar
        else:
            coords = normalise(coords)
            mu, kappa = vmf.fit(coords)
            mus[i] = mu
            kappas[i] = kappa 
    if bind:
        cols.mu = mus
        cols.kappa = kappas
    else:
        return mus, kappas
    
def _col_pnt_likelihood(cols:Columns,col_ind:int,coords:np.ndarray,norm:bool = True) -> float | np.ndarray:
    """Assuming cols object has mu and kappa as attributes, return likelihood, or normalised likelihood of a 
    point being within this column.

    Parameters
    ----------
    cols : Columns
        SSet of columns
    col_ind : int
        Specific column to check points against
    coords : np.ndarray
        Set of 3D point coordinates
    norm : bool, optional
        If True, normalises the likelihood by the likelihood at the centre of the column, by default True

    Returns
    -------
    float | np.ndarray
        If a single coordinate, it's likelihood, otherwise an array of likelihoods in same order as given coordinates
    """
    
    # normalise points
    coords = normalise(coords)
    mu = cols.mu[col_ind]
    kappa = cols.kappa[col_ind]

    prob = vmf.pdf(coords, mu,kappa)
    if norm:
        prob = prob / vmf.pdf(mu,mu,kappa)

    return prob

def vmf_likelihood_matrix(cols: Columns, coords: np.ndarray, norm:bool = True):
    """Calulate the likelihood of a point being within the vonMises-Fisher fit of a column
    
    Generate columns by coordinates likelihood matrix of points being within columns

    Assumes cols has mu and kappa attributes, if you've not done this, add them with fit_vonMises_Fisher and bind = True

    Parameters
    ----------
    cols : Columns
        Set of Columns
    coords : np.ndarray
        Set of points
    norm : bool, optional
        Whether or not to normalise the returned likelihood, by default True

    Returns
    -------
    np.ndarray
        columns by points matrix of likelihoods
    """
    # initialise data
    n = len(cols.Column_ids)
    if coords.ndim == 1:
        m = 1
    else:
        m = coords.shape[0]
    
    data = np.zeros((n,m))
    for i in range(len(cols.Column_ids)):
        curr_col = cols.Column_ids[i]
        try:
            data[i] = _col_pnt_likelihood(cols,curr_col,coords, norm = norm)
        except:
            ar = np.empty(m)
            ar[:] = np.nan
            data[i] = ar

    return data

from scipy.optimize import linear_sum_assignment
def hungarian_tie_handling(prob_matrix):
    """
    Finds the optimal set of [row, column] indices that maximize the sum of assigned values
    while handling ties systematically:
    - Rows with all zeros are ignored (assigned np.nan).
    - Non-zero ties are resolved by testing all possibilities for global optimization.
    - Ensures that no column is assigned to more than one row.

    Parameters:
        prob_matrix (numpy.ndarray): An n x m array of probabilities (values between 0 and 1).

    Returns:
        list: Optimal assignments (row, column) with np.nan for rows that are ignored.
    """
    n, m = prob_matrix.shape
    assignments = [None] * n  # Initialize assignments
    used_columns = set()  # Track assigned columns

    for i, row in enumerate(prob_matrix):
        max_val = np.max(row)
        tied_indices = np.where(row == max_val)[0]

        if max_val == 0:
            # Case 1: All values in the row are zero
            assignments[i] = np.nan
            continue

        # Track all valid columns for this row
        valid_columns = [col for col in tied_indices if col not in used_columns]
        if not valid_columns:
            # No valid column to assign
            assignments[i] = np.nan
            continue

        # Test all valid columns and pick the best option
        best_score = -np.inf
        best_assignment = None

        for col in valid_columns:
            # Create a temporary matrix where this row assigns to `col`
            temp_matrix = prob_matrix.copy()
            temp_matrix[i, :] = 0  # Zero out the row
            temp_matrix[i, col] = max_val  # Assign the tied value

            # Solve the assignment problem for the rest
            cost_matrix = -temp_matrix  # Negate for maximization
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            current_score = -cost_matrix[row_indices, col_indices].sum()

            if current_score > best_score:
                best_score = current_score
                best_assignment = col

        # Assign the best column, if found
        if best_assignment is not None:
            assignments[i] = (i, best_assignment)
            used_columns.add(best_assignment)
        else:
            # Fallback: assign the first available valid column
            fallback_column = valid_columns[0]
            assignments[i] = (i, fallback_column)
            used_columns.add(fallback_column)

    return assignments