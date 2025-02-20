from .columns import Columns
from GeoJax import norma
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
            coords = GeoJax.normalise(coords)
            mu, kappa = vmf.fit(coords)
            mus[i] = mu
            kappas[i] = kappa 
    if bind:
        cols.mu = mus
        cols.kappa = kappas
    else:
        return mus, kappas