from .columns import Columns
from GeoJax import normalize
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

def calculate_threshold(kappa, p, overflow_threshold:float = 400):
    """
    Calculate the cosine similarity threshold for a von Mises–Fisher distribution on S².
    
    For high kappa, to avoid numerical overflow, we use the approximation:
        t ≈ 1 + log(1-p)/kappa
    
    Parameters:
        mu (array-like): The mean direction (unit vector in R^3). (Provided for consistency.)
        kappa (float): The concentration parameter of the distribution (> 0).
        p (float): The desired probability mass within the cutoff (0 < p < 1).
                   For example, 0.997 is analogous to ±3 standard deviations.
    
    Returns:
        t (float): The threshold value, i.e., the minimum cosine similarity (μᵀx)
                   required for a point x to be within the cutoff.
    """
    if kappa <= 0:
        raise ValueError("kappa must be greater than 0.")
    
    # Choose a cutoff to switch to the approximation.
    # np.exp(700) is about the upper limit before overflow, so we use a much lower threshold.
    if kappa > overflow_threshold:  # You can adjust this threshold as needed.
        # Use the stable approximation
        t = 1.0 + np.log(1-p) / kappa
    else:
        numerator = np.exp(kappa) - p * (np.exp(kappa) - np.exp(-kappa))
        t = (1.0 / kappa) * np.log(numerator)
    return t

def points_within_cutoff(mu, t, points):
    """
    Determine which points on the unit sphere lie within the cutoff defined by threshold t.
    
    Parameters:
        mu (array-like): The mean direction (unit vector in R^3).
        kappa (float): The concentration parameter (not used directly in this function).
        t (float): The cosine similarity threshold (from calculate_threshold).
        points (numpy.ndarray): An array of shape (N, 3) containing N 3D points on the unit sphere.
    
    Returns:
        numpy.ndarray: A boolean array of length N where True indicates that the
                       corresponding point satisfies μᵀx >= t.
    """
    mu = np.asarray(mu)
    # Ensure mu is normalized.
    mu = mu / np.linalg.norm(mu)
    
    points = np.asarray(points)
    # Compute dot products (cosine similarities) for all points.
    cos_similarities = points.dot(mu)
    
    # Return boolean array: True if cosine similarity >= t.
    return cos_similarities >= t

def point_on_threshold(mu, t):
    """
    Given a unit vector mu and a cosine similarity threshold t, return a point x on the unit sphere
    that is exactly at the angular distance alpha = arccos(t) from mu. That is, x satisfies:
        μᵀx = t.
    
    Parameters:
        mu (array-like): A unit vector in R^3 representing the center direction.
        t (float): The cosine similarity threshold (between -1 and 1).
    
    Returns:
        numpy.ndarray: A point on the unit sphere that is at angular distance α = arccos(t) from mu.
    """
    # Ensure mu is a unit vector
    mu = np.asarray(mu, dtype=np.float64)
    mu = mu / np.linalg.norm(mu)
    
    # Calculate the angular distance alpha from t.
    alpha = np.arccos(t)
    
    # Choose an arbitrary vector not collinear with mu for constructing a tangent vector.
    if abs(mu[0]) < 0.9:
        arbitrary = np.array([1, 0, 0], dtype=np.float64)
    else:
        arbitrary = np.array([0, 1, 0], dtype=np.float64)
    
    # Compute a unit vector u in the tangent plane at mu.
    u = np.cross(mu, arbitrary)
    u = u / np.linalg.norm(u)
    
    # Construct the point at the threshold
    x = np.cos(alpha)*mu + np.sin(alpha)*u
    # The result x is on the unit sphere.
    return x

def likelihood_threshold(x,mu,kappa,norm = True):
    if norm:
        return vmf.pdf(x,mu,kappa)/vmf.pdf(mu,mu,kappa)
    else:
        return vmf.pdf(x,mu,kappa)