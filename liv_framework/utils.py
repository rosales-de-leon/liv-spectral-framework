import numpy as np
import astropy.units as u
from astropy.constants import c


def compute_blob_radius(t_var, delta_D, z):
    """
    Compute blob radius from variability timescale.

    Parameters
    ----------
    t_var : Quantity
        Variability timescale.
    delta_D : float
        Doppler factor.
    z : float
        Redshift.

    Returns
    -------
    Quantity
        Blob radius.
    """
    return (c * t_var * delta_D / (1 + z)).to("cm")


def format_scaled(value, exponent, precision=2):
    """
    Format number scaled by a common exponent.
    """
    if np.isnan(value):
        return "--"

    return f"{value / 10**exponent:.{precision}f}"


def symlog_grid(xmin, xmax, xbreak, n_half):
    """
    Build symmetric log grid.
    """

    xneg = -np.logspace(
        np.log10(abs(xmin)),
        np.log10(abs(xbreak)),
        n_half,
    )

    xpos = np.logspace(
        np.log10(abs(xbreak)),
        np.log10(abs(xmax)),
        n_half,
    )

    return np.concatenate((xneg, [0.0], xpos))
