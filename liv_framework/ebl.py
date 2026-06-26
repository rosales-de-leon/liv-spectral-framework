import numpy as np
import matplotlib.pyplot as plt
import re
from astropy.constants import c, G, M_sun, m_e, m_p, h
import astropy.units as u

from scipy.integrate import quad
from scipy.interpolate import interp1d, RegularGridInterpolator, PchipInterpolator, make_interp_spline
from numba import jit, float32
from concurrent.futures import ProcessPoolExecutor

from gammapy.modeling.models import SpectralModel
from gammapy.modeling import Parameter


# ===============================
# LOADER FOR DOMINGUEZ MODEL PHOTON DENSITY
# ===============================

# --- Dominguez ---
def load_ebl_data(filename):
    data = np.loadtxt(filename)
    redshifts = np.array([0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.9])
    wavelengths = data[:, 0] * u.micron
    intensity_matrix = data[:, 1:] * (u.nW / u.m**2 / u.sr)
    return redshifts, wavelengths, intensity_matrix


## Grid interpolator for computing the photon energy density for dominguez model 
from astropy import constants as const

def precompute_photon_density_grid(redshifts, wavelengths, lambda_I_lambda):
    # Define the grid in redshift and energy
    epsilon_grid = np.logspace(np.log10(1e-3), np.log10(1e2), 500)  # en eV
    z_grid = np.linspace(0, 4.0, 100)
    n_grid = np.zeros((len(epsilon_grid), len(z_grid)))
    
    # Change units to ev / cm^2 s sr
    lambda_I_lambda = lambda_I_lambda.to(u.eV / (u.cm**2 * u.s * u.sr))
    
    # Interpolation for z
    z_interp_functions = []
    for i in range(len(wavelengths)):
        f = interp1d(
            redshifts, lambda_I_lambda[i, :].value,
            kind='linear', bounds_error=False, fill_value=0.0
        )
        z_interp_functions.append(f)
    
    # Computation of n(epsilon, z)
    for j, z in enumerate(z_grid):
        # Interpolation λIλ for each λ at a given z
        lambda_I_lambda_z = np.array([f(z) for f in z_interp_functions]) * lambda_I_lambda.unit
        
        for i, epsilon in enumerate(epsilon_grid):
            # energy to wavelength
            lambda_ebl = (const.h * const.c / (epsilon * u.eV)).to(u.micron)
            
            # Interpolation of λIλ(λ, z)
            I_lambda = np.interp(
                lambda_ebl.value, wavelengths.value, lambda_I_lambda_z.value,
                left=0.0, right=0.0  # No extrapolations
            ) * lambda_I_lambda.unit
            
            # n(ε, z):
            n_epsilon = (4 * np.pi * u.sr / const.c.to(u.cm / u.s)) * (1 / (epsilon * u.eV)**2) * I_lambda
            
            n_grid[i, j] = n_epsilon.value

    # Final interpolation in log(ε) y z
    n_interp = RegularGridInterpolator((np.log(epsilon_grid), z_grid), n_grid)
    return n_interp, epsilon_grid

# ===============================
# LOADER FOR FRANCESCHINI MODEL PHOTON DENSITY
# ===============================

def load_franceschini_data(file_path):
    # 1. READ & CLEAN DATA
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()

    cleaned_lines = []
    for line in raw_lines:
        if line.strip().startswith("#"): continue
        # Handle unicode minus signs found in the text file
        line = re.sub(r'[\u2013\u2212\u2014\u00ad]', '-', line)
        cleaned_lines.append(line.strip())

    data_list = []
    for line in cleaned_lines:
        try:
            row = list(map(float, line.split()))
            if row: data_list.append(row)
        except ValueError: continue
    data = np.array(data_list)

    # 2. CONVERSION TO COMOVING PHOTON DENSITY
    z_vals = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    n_z = len(z_vals)

    # epsilon_common must cover the full range of the table (approx 1e-3 to 12 eV)
    epsilon_common = np.logspace(-4, 2, 1000)
    n_grid = np.zeros((len(epsilon_common), n_z))

    for i in range(n_z):
        z = z_vals[i]
        
        # Extract energy and density for current redshift
        # Table format: log_eps_z0, log_density_z0, log_eps_z1, log_density_z1...
        eps_table = 10**data[:, 2*i]
        log_eps_dnde = data[:, 2*i + 1]
        
        # Calculate Proper Differential Photon density (dn/de)
        dnde_proper = (10**log_eps_dnde) / eps_table
        
        # Convert to comoving frame: n_comov = n_proper / (1+z)^3
        dnde_comov = dnde_proper / (1 + z)**3
        
        # DEFINE PHYSICAL BOUNDARIES for this specific redshift 
        eps_min_z = eps_table.min()
        eps_max_z = eps_table.max()

        # Interpolation in log-log space to maintain numerical stability
        interp_func = interp1d(
            np.log10(eps_table),
            np.log10(np.maximum(dnde_comov, 1e-40)),
            kind='linear',
            bounds_error=False,
            fill_value=-40 # Represents effectively zero in log space
        )
        
        # Evaluate on the common grid
        dens_interp = 10**interp_func(np.log10(epsilon_common))
        
        # STRICT MASKING: Force to 0.0 outside the actual model limits
        # This prevents the "spike" at low epsilon
        mask = (epsilon_common >= eps_min_z) & (epsilon_common <= eps_max_z)
        n_grid[:, i] = np.where(mask, dens_interp, 0.0)

    return epsilon_common, z_vals, n_grid


# ==========================================================
# PRECOMPUTE INTERPOLATOR
# ==========================================================

def precompute_photon_density_grid_franceschini(file_path):
    epsilon_grid, z_grid, n_grid = load_franceschini_data(file_path)

    # Use log(epsilon) for the grid to match the integral's log-space stepping
    n_interp = RegularGridInterpolator(
        (np.log(epsilon_grid), z_grid),
        n_grid,
        method='linear',
        bounds_error=False,
        fill_value=0.0 # Any value outside the epsilon_common range is 0
    )

    return n_interp, epsilon_grid
    
    
def get_ebl_interpolator(model, file_path=None):
    """
    Returns the photon density interpolator and epsilon grid.
    Standardized to return comoving frame data for all models.
    """
    model_low = model.lower()
    
    if model_low == "dominguez":
        
        redshifts, wavelengths, intensity_matrix = load_ebl_data(file_path)
        
        n_interp, epsilon_grid = precompute_photon_density_grid(redshifts, wavelengths, intensity_matrix)
        return n_interp, epsilon_grid

    elif model_low == "franceschini":
        # The corrected precompute function now returns comoving density
        n_interp, epsilon_grid = precompute_photon_density_grid_franceschini(file_path)
        return n_interp, epsilon_grid

    else:
        raise ValueError("Model must be 'dominguez' or 'franceschini'")
        
### Breit-Wheeler cross section
@jit(float32(float32))
def sigma_breit_wheeler_s_numba(s):
    s_thresh = 4.0 * (511000.0)**2  # eV^2    
#    s_thresh = 1044479707588.7295  # eV^2    
    if s <= s_thresh:
        return 0
    beta = np.sqrt(1 - s_thresh / s)
    
    #sigma_T = 6.6524e-25 # Thomson cross-section in cm^2
    #beta = np.clip(beta, 1e-5, 1 - 1e-5)  # Avoid problematic values around 0 and 1
    term1 = (1 - beta**2) * (2 * beta * (beta**2 - 2) + (3 - beta**4) * np.log((1 + beta) / (1 - beta)))
    
    return term1 # Return as a dimensionless number in cm^2
    
# ---------------------------------------------------------
### COMPUTATION OF OPACITY WITH LIV EFFECTS
# ---------------------------------------------------------
def tau_liv(E_gamma, z, n_interp, epsilon_grid, xi_n=0.0, n_order=1):
    """
    Optical depth including LIV effects using comoving EBL density.

    xi_n:
        = 0  → standard EBL
        > 0  → superluminal
        < 0  → subluminal
    """

    # --- Energy ---
    E_gamma = E_gamma.to(u.eV)
    E_gamma_val = E_gamma.value

    # --- Constants ---
    s_thresh = (4 * const.m_e**2 * const.c**4).to(u.eV**2).value
    sigma_T = 6.6524e-25

    prefactor = (3/16) * sigma_T

    # --- LIV handling ---
    if xi_n == 0:
        sign = 0
    else:
        sign = np.sign(xi_n)
        E_pl = (1.22e19 * u.GeV).to(u.eV)
        E_LIV = (E_pl / abs(xi_n)).to(u.eV).value
        

    def delta_liv(E_local):
        if xi_n == 0:
            return 0.0
        if n_order == 1:
            return sign * (E_local**3) / E_LIV
        elif n_order == 2:
            return sign * (E_local**4) / (E_LIV**2)
        else:
            raise ValueError("n_order must be 1 or 2")
            
    # ---------------------------------------------------------
    # Vacuum photon decay for superluminal LIV
    # ---------------------------------------------------------

    if sign > 0:

        E_max_path = E_gamma_val * (1 + z)

        if delta_liv(E_max_path) >= s_thresh:
            return 1e+30  #huge value instead of +inf 


    # --- Integrals ---
    def redshift_integral(z_int):
        E_local = E_gamma_val * (1 + z_int)

        def epsilon_integral_log(log_epsilon):
            epsilon = np.exp(log_epsilon)
            
            # Simplified EBL lookup: comoving 
            try:
                n_epsilon = n_interp((np.log(epsilon), z_int))
            except:
                return 0.0
            
            
            delta = delta_liv(E_local)

            s_min = s_thresh
            s_max = 4 * E_local * epsilon + delta

            if s_max <= s_min:
                return 0.0
            
            ### condition if there is a negligible contribution
            if np.log(s_max) - np.log(s_min) < 1e-8:
                return 0.0

            
            def s_integral_log(u):
                s = np.exp(u)
                sigma = sigma_breit_wheeler_s_numba(s)
                return sigma * (s - delta) * s  # extra s from log-substitution

            u_min = np.log(s_min)
            u_max = np.log(s_max)

            s_int, _ = quad(s_integral_log, u_min, u_max,
                            epsabs=1e-4, epsrel=1e-3, limit=60)

            return (n_epsilon / epsilon) * s_int  # 1/epsilon from log-substitution

        delta = delta_liv(E_local)
        epsilon_min = (s_thresh - delta) / (4 * E_local)
        epsilon_max = np.max(epsilon_grid)

        if epsilon_min >= epsilon_max:
            return 0.0

        if epsilon_min <= 0:
            epsilon_min = max(epsilon_min, epsilon_grid[0])   # this is a small positive value

        epsilon_int, _ = quad(
            epsilon_integral_log,
            np.log(epsilon_min),
            np.log(epsilon_max),
            epsabs=1e-4, epsrel=1e-3, limit=60
        )

        # Cosmology
        Omega_m, Omega_lambda = 0.3, 0.7
        H_z = np.sqrt(Omega_m * (1 + z_int)**3 + Omega_lambda)

        return epsilon_int / H_z
    
    # --- Outer integral ---
    int_final, _ = quad(redshift_integral, 0, z,
                        epsabs=1e-4, epsrel=1e-3, limit=60)

    H0 = 70 * (u.km / u.s / u.Mpc)

    final_prefactor = (
        (prefactor / (8 * E_gamma_val**2)) *
        (const.c / H0).to(u.cm) *
        (u.eV**2 / u.cm)
    ).value

    return final_prefactor * int_final
    


# ===============================
# MULTIPROCESSING FUNCTION
# ===============================
def compute_tau_single(args):
    """Unpack arguments and call simplified tau_liv."""
    E, xi_val, z_val, n_int, eps_grid, order = args
    return tau_liv(E, z_val, n_int, eps_grid, xi_n=xi_val, n_order=order)


# ===============================
# PARALLEL COMPUTATION
# ===============================
def compute_model(n_interp, epsilon_grid, model_name, xi_n_array, E_gamma_vals, z, n_order=1):

    model_results = []

    # Parallel processing
    with ProcessPoolExecutor() as executor:

        for xi_n_val in xi_n_array:

            print(f"[{model_name}] Processing xi_n = {xi_n_val:.3e}")

            args_list = [
                (
                    E,
                    xi_n_val,
                    z,
                    n_interp,
                    epsilon_grid,
                    n_order,
                )
                for E in E_gamma_vals
            ]

            tau_vals = list(
                executor.map(
                    compute_tau_single,
                    args_list
                )
            )

            model_results.append(tau_vals)

    return np.asarray(model_results)    
# ---------------------------------------------------------
### DOMINGUEZ EBL+LIV SPECTRAL MODEL IN GAMMAPY
# ---------------------------------------------------------
class EBL_LIV_CombinedAbsorptionSpectralModel(SpectralModel):

    tag = "EBL_LIV_CombinedAbsorptionSpectralModel"

    redshift = Parameter("redshift", 0.069, frozen=True)
    xi_n = Parameter("xi_n", 1.0, frozen=False)
    n_order = Parameter("n_order", 2.0, frozen=True)

    def __init__(self, file_subluminal, file_superluminal,
                 redshift, xi_n_init, n_order):

        pars = {
            "redshift": redshift,
            "xi_n": xi_n_init,
            "n_order": float(n_order),
        }

        super().__init__(**pars)

        # ---------------------------------------------------------
        # 1) LOAD BOTH TABLES
        # ---------------------------------------------------------
        E_sub, xi_sub, tau_sub = self._load_table(file_subluminal, sign=-1)
        E_sup, xi_sup, tau_sup = self._load_table(file_superluminal, sign=+1)

        # ---------------------------------------------------------
        # 2) CHOOSE COMMON ENERGY GRID
        # ---------------------------------------------------------
        # Use superluminal grid as reference
        self.energy_grid = E_sup
        logE_ref = np.log(E_sup.value)

        # ---------------------------------------------------------
        # 3) INTERPOLATE SUBLUMINAL ONTO SUPERLUMINAL GRID
        # ---------------------------------------------------------
        tau_sub_interp = np.zeros((len(E_sup), len(xi_sub)))

        for i in range(len(xi_sub)):
            pchip = PchipInterpolator(
                np.log(E_sub.value),
                np.log(np.maximum(tau_sub[:, i], 1e-300))
            )
            tau_sub_interp[:, i] = np.exp(pchip(logE_ref))

        # ---------------------------------------------------------
        # 4) MERGE XI AND TAU MATRICES
        # ---------------------------------------------------------
        xi_all = np.concatenate([xi_sub, xi_sup])
        tau_all = np.hstack([tau_sub_interp, tau_sup])

        # Sort by xi
        sort_idx = np.argsort(xi_all)
        self.xi_n_values = xi_all[sort_idx]
        tau_all = tau_all[:, sort_idx]

        # ---------------------------------------------------------
        # 5) BUILD LOG–LOG INTERPOLATOR
        # ---------------------------------------------------------
        logE = np.log(self.energy_grid.value)
        logXi = np.log(np.abs(self.xi_n_values))

        tau_safe = np.where(tau_all > 1e-300, tau_all, 1e-300)
        logTau = np.log(tau_safe)

        self.logTau_interp_2D = RegularGridInterpolator(
            (logE, self.xi_n_values),
            logTau,
            bounds_error=False,
            fill_value=None,
        )

    # -------------------------------------------------------------
    def _load_table(self, filename, sign=+1):

        import re

        # ---------------------------------------------------------
        # Read header
        # ---------------------------------------------------------
        with open(filename, "r") as f:
            header = f.readline().strip()

        col_names = header.replace("#", "").split()

        # ---------------------------------------------------------
        # Extract xi values AND keep original column names
        # ---------------------------------------------------------
        xi_vals = []
        tau_col_indices = []

        pattern = r"tau_xi_([0-9.eE+-]+)"

        for i, name in enumerate(col_names):
            match = re.match(pattern, name)
            if match:
                xi_str = match.group(1)
                xi_val = float(xi_str)

                xi_vals.append(sign * xi_val)
                tau_col_indices.append(i)

        xi_vals = np.array(xi_vals)

        if len(xi_vals) == 0:
            raise ValueError("No tau_xi_* columns found!")

        # ---------------------------------------------------------
        # Load numerical data
        # ---------------------------------------------------------
        data = np.loadtxt(filename)

        energy = data[:, 0] * u.TeV

        # Directly extract tau matrix using indices
        tau_matrix = data[:, tau_col_indices]

        return energy, xi_vals, tau_matrix


    # -------------------------------------------------------------
    def evaluate(self, energy, **kwargs):

        logE_val = np.log(energy.to(u.TeV).value)
        xi = self.xi_n.value

        pts = np.column_stack([
            logE_val,
            np.full_like(logE_val, xi)
        ])

        logtau = self.logTau_interp_2D(pts)
        tau = np.exp(logtau)

        return np.exp(-tau)

# ---------------------------------------------------------
### FRANCESCHINI EBL+LIV SPECTRAL MODEL IN GAMMAPY
# ---------------------------------------------------------
class EBL_LIV_FranceschiniAbsorptionSpectralModel(SpectralModel):
    """
    Gammapy custom spectral model for EBL absorption with LIV effects
    using the Franceschini model from a precomputed text file.
    """
    tag = "EBL_LIV_FranceschiniAbsorptionSpectralModel"

    redshift = Parameter("redshift", 0.069, frozen=True)
    xi_n = Parameter("xi_n", 0.0, frozen=False)
    n_order = Parameter("n_order", 1.0, frozen=True)

    def __init__(self, file_path, redshift, xi_n_init=0.0, n_order=1):
        
        # Initialize parameters
        pars = {
            "redshift": redshift,
            "xi_n": xi_n_init,
            "n_order": float(n_order),
        }
        super().__init__(**pars)

        # ---------------------------------------------------------
        # 1) LOAD THE SINGLE TABLE
        # ---------------------------------------------------------
        energy, xi_values, tau_matrix = self._load_table(file_path)

        self.energy_grid = energy
        self.xi_n_values = xi_values

        # ---------------------------------------------------------
        # 2) BUILD LOG–LOG INTERPOLATOR
        # ---------------------------------------------------------
        # We interpolate log(tau) over log(E) and linear xi
        logE = np.log(self.energy_grid.value)
        
        # Ensure xi_values are strictly increasing for RegularGridInterpolator
        sort_idx = np.argsort(self.xi_n_values)
        xi_sorted = self.xi_n_values[sort_idx]
        tau_sorted = tau_matrix[:, sort_idx]

        # Log-Log safety: avoid zeros in tau
        tau_safe = np.where(tau_sorted > 1e-150, tau_sorted, 1e-150)
        logTau = np.log(tau_safe)

        # Interpolator: (logE, xi)
        self.logTau_interp_2D = RegularGridInterpolator(
            (logE, xi_sorted),
            logTau,
            method='linear',
            bounds_error=False,
            fill_value=None, # It will extrapolate or we can force 1e8 for explosion
        )

    # -------------------------------------------------------------
    def _load_table(self, filename):
        """
        Parses the header to find xi_n values and loads the matrix.
        """
        with open(filename, "r") as f:
            header = f.readline().strip()

        # Extract xi values from column names like 'tau_xi_-1e+06'
        pattern = r"tau_xi_([-+]?[0-9.eE+-]+)"
        xi_vals = np.array([float(x) for x in re.findall(pattern, header)])

        # Load numerical data
        data = np.loadtxt(filename)
        
        # First column is Energy in TeV
        energy = data[:, 0] * u.TeV
        
        # Rest of columns are tau values
        # data[:, 1:] matches the order of xi_vals extracted from the header
        tau_matrix = data[:, 1:]

        return energy, xi_vals, tau_matrix

    # -------------------------------------------------------------
    def evaluate(self, energy, **kwargs):
        """
        Evaluate the model: exp(-tau)
        """
        # Convert input energy to log(TeV)
        logE_val = np.log(energy.to(u.TeV).value)
        xi = self.xi_n.value

        # Prepare points for 2D interpolation: (logE, xi)
        # Handle single value or array energy inputs
        if np.isscalar(logE_val):
            pts = np.array([logE_val, xi])
        else:
            pts = np.column_stack([
                logE_val,
                np.full_like(logE_val, xi)
            ])

        logtau = self.logTau_interp_2D(pts)
        tau = np.exp(logtau)

        # For superluminal explosion: if extrapolation goes very high, 
        # exp(-tau) will naturally go to 0.
        return np.exp(-tau)
