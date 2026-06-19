import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from gammapy.modeling.models import EBLAbsorptionNormSpectralModel, PowerLawSpectralModel, SkyModel

def find_minimum_epsilon(n, xi_n, liv_type="subluminal"):
    """
    Finds the minimum of the LIV-modified epsilon_threshold curve.

    Parameters:
    - n: Order of LIV correction (0 = no LIV, 1 = linear, 2 = quadratic).
    - xi_n: LIV coefficient.
    - liv_type: "subluminal" or "superluminal".

    Returns:
    - E_gamma_min (in TeV)
    - epsilon_min (in eV)
    """

    # Constants
    m_e = 0.511 * u.MeV
    E_pl = 1.22e19 * u.GeV
    E_LIV = E_pl/xi_n

    # Case 0 = Standard physics
    if n == 0:
        return None, None

    # Case n=2, subluminal → analytic minimum
    if n == 2 and liv_type == "subluminal":
        E_gamma_min = ((4/3)**0.25) * np.sqrt(m_e * E_LIV)
        E_gamma_min = E_gamma_min.to(u.TeV)

        eps_min = (m_e.to(u.GeV)**2 / E_gamma_min.to(u.GeV) +
                   0.25 * E_gamma_min.to(u.GeV)**3 / E_LIV.to(u.GeV)**2)
        eps_min = eps_min.to(u.eV)

        return E_gamma_min, eps_min

    # For other cases (e.g., n=1 or superluminal n=2) → grid search
    sign = +1 if liv_type == "subluminal" else -1
    E_gamma = np.logspace(-1, 6.5, 2000) * u.TeV  # Extend search up to 10^6 TeV
    E_gamma_GeV = E_gamma.to(u.GeV)
    m_e_GeV = m_e.to(u.GeV)

    epsilon_threshold = (m_e_GeV**2 / E_gamma_GeV) + \
                        sign * 0.25 * ((E_gamma_GeV / E_LIV.to(u.GeV))**n) * E_gamma_GeV
    epsilon_threshold_eV = epsilon_threshold.to(u.eV)

    min_index = np.argmin(epsilon_threshold_eV)
    E_gamma_min = E_gamma[min_index].to(u.TeV)
    eps_min = epsilon_threshold_eV[min_index]

    return E_gamma_min, eps_min
    
def plot_epsilon_threshold(n_values, xi_n_values, liv_type="subluminal"):
    """
    Plots epsilon_threshold as function of E_gamma for LIV corrections.

    Parameters:
    - n_values: list of LIV correction orders.
    - xi_n_values: list of LIV coefficients.
    - liv_type: "subluminal", "superluminal", or "both".
    """

    valid_liv_types = ["subluminal", "superluminal", "both"]
    if liv_type not in valid_liv_types:
        raise ValueError(f"liv_type must be one of {valid_liv_types}.")

    # Constants
    m_e = 0.511 * u.MeV
    E_pl = 1.22e19 * u.GeV
    E_gamma = np.logspace(-1, 6.5, 2000) * u.TeV
    E_gamma_GeV = E_gamma.to(u.GeV)
    m_e_GeV = m_e.to(u.GeV)

    plt.figure(figsize=(8, 6))
    
    

    for n in n_values:
        if n == 0:
            epsilon_threshold = m_e_GeV**2 / E_gamma_GeV
            plt.plot(E_gamma, epsilon_threshold.to(u.eV), 'k--', label="Standard (No LIV)")
        else:
            for xi_n in xi_n_values:
                E_LIV = E_pl/xi_n

                if liv_type in ["subluminal", "both"]:
                    epsilon_thr_sub = (m_e_GeV**2 / E_gamma_GeV) + \
                                      0.25 * ((E_gamma_GeV / E_LIV.to(u.GeV))**n) * E_gamma_GeV
                    plt.plot(E_gamma, epsilon_thr_sub.to(u.eV),
                             label=rf"$\xi_n={xi_n:.0e}$")

                if liv_type in ["superluminal", "both"]:
                    epsilon_thr_super = (m_e_GeV**2 / E_gamma_GeV) - \
                                        0.25 * ((E_gamma_GeV / E_LIV.to(u.GeV))**n) * E_gamma_GeV
                    plt.plot(E_gamma, epsilon_thr_super.to(u.eV),
                             label=rf"$\xi_n={xi_n:.0e}$")

    # Config
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$E_{\gamma}$ [TeV]")
    plt.ylabel(r"$\epsilon_{\rm th}$ [eV]")
    plt.legend(loc=1, fontsize=9)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.xlim([1e-1, 4e+6])
    plt.ylim([1e-9, 1e+9])
    plt.show()
    
    
def xi_to_eqg_limit(xi_limit):
    """
    Convert xi_n limit to E_QG limit.

    Parameters
    ----------
    xi_limit : float
        absolute value of xi_n limit
    n : int
        LIV order (1 or 2)

    Returns
    -------
    E_QG in eV
    """
    return E_PLANCK * (1.0 / np.abs(xi_limit))
