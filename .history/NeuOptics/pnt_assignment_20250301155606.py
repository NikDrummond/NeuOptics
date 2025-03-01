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

def max_pnt_assignment(cols:Columns, coordinates: np.ndarray)-> np.array:
    """Return the index of column which each point in coordinates is assigned to, based on maximum liklelihood

    Given Columns object must have the mu and kappa attributes so we can evaluate coordinates against the vonMises-Fisher pdf.

    We leave it to the user to make sure that the columns and the point coordinates make sense to be mapped together!
    You will ALWAYS get an assignment regardless of whether or not mapping the points to the columns makes sense. 

    Parameters
    ----------
    cols : NeuOptics.Columns
        Column object to assign points to
    coordinates : np.ndarray
        Coordinates to assign to columns

    Returns
    -------
    np.array
        In the same order as coordinates, index of column which each coordinate is assigned to.
    """
    assert (hasattr(cols, 'mu')) & (hasattr(cols, 'kappa')), 'Columns object must have mu and kappa attribute, \n Please fit the vonMises-Fisher using neuOptics.fit_vonMises_Fisher'
    mat = vmf_likelihood_matrix(cols,coordinates)
    return np.nanargmax(mat, axis = 0)