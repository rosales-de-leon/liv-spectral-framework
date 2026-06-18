import os
import json
import numpy as np
import astropy.units as u

from astropy.constants import m_e

from agnpy.spectra import (
    PowerLaw,
    BrokenPowerLaw,
    LogParabola,
    ExpCutoffPowerLaw,
)

from agnpy.emission_regions import Blob
from agnpy.fit import SynchrotronSelfComptonModel, add_systematic_errors_gammapy_flux_points

from gammapy.datasets import FluxPointsDataset, Datasets
from gammapy.estimators import FluxPoints

def load_multiple_gammapy_flux_points(sed_paths, E_min, E_max, systematics_dict=None):
    """Load multiple MWL SEDs from different .ecsv files and merge them into a single dataset.
    
    Parameters
    ----------
    sed_paths : list of str
        list of paths to the .ecsv files to be used for fitting
    E_min : `~astropy.Quantity`
        minimum energy to be used in the fit
    E_max : `~astropy.Quantity`
        maximum energy to be used in the fit
    systematics_dict : dict
        dictionary containing the instrument name and the systematics error
        associated with it. The sys. error should be expressed as a relative
        error on the flux. For example:
        `systematics_dict = {"Fermi" : 0.1, "MAGIC" : 0.30, "Swift-XRT": 0.10}`

    Returns
    -------
    `~gammapy.dataset.Datasets`, list of flux points datasets
    """
    datasets = Datasets()

    # Loop over each file in the provided list of .ecsv files
    for sed_path in sed_paths:
        table = Table.read(sed_path)

        # Group by 'instrument' as in the original function
        table = table.group_by("instrument")

        for group in table.groups:
            name = group["instrument"][0]
            data = FluxPoints.from_table(group, sed_type="e2dnde", format="gadf-sed")

            if systematics_dict is not None:
                for instrument in systematics_dict.keys():
                    if name == instrument:
                        syst_rel_error = systematics_dict[instrument]
                        add_systematic_errors_gammapy_flux_points(data, syst_rel_error)

            # Load the flux points in a dataset
            dataset = FluxPointsDataset(data=data, name=name)

            # Set the minimum energy to be used for the fit
            mask = (dataset.data.energy_ref >= E_min) * (dataset.data.energy_ref <= E_max)
            dataset.mask_fit = mask

            datasets.append(dataset)

    return datasets
    
    
### FIT MODELS: NO LIV
def fit_BL_Lac(datasets, electron_distributions, LogParabola_model, SmoothBrokenPowerLaw_model_HE, absorption):
    """
    ----------------------------------------------------------------------
    Fit funtion to test multiple electron energy distributions (n_e) to given datasets.
    ----------------------------------------------------------------------
    
    Returns:

    fit_results : dict
        Dictionary containing fit results, AIC, chi², and models.
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    """
    
    fit_results = {}

    for i, (name, n_e) in enumerate(electron_distributions.items()):
        print(f"\n=== Fitting using {name} electron distribution ===")

        # -----------------------------------------------------------
        # Blob Physical Parameters
        # -----------------------------------------------------------
        z = 0.069
        Gamma_blob = 30    # Lorentz factor
        delta_blob = 30    # Doppler factor
        Beta_blob = np.sqrt(1 - 1 / np.power(Gamma_blob, 2))
        mu_blob = (1 - 1 / (Gamma_blob * delta_blob)) / Beta_blob
        B_blob = 0.1 * u.G
        R_blob = 3.0e15 * u.cm

        blob = Blob(R_blob, z, delta_blob, Gamma_blob, B_blob, n_e=n_e)
        ssc_blob_model = SynchrotronSelfComptonModel(n_e, backend="gammapy", ssa=True)

        # Set SSC model parameters explicitly
        ssc_blob_model.parameters["z"].value = z
        ssc_blob_model.parameters["delta_D"].value = delta_blob
        ssc_blob_model.parameters["log10_B"].value = np.log10(B_blob.to_value("G"))
        ssc_blob_model.parameters["t_var"].value = blob.t_var.to_value("s")

        # -----------------------------------------------------------
        # Total model definition
        # -----------------------------------------------------------
        Flare_model = LogParabola_model + (SmoothBrokenPowerLaw_model_HE + ssc_blob_model) * absorption

        sky_model = SkyModel(spectral_model=Flare_model, name=f"BL_Lac_{name}")
        datasets.models = [sky_model]

        # -----------------------------------------------------------
        # Free / frozen parameters
        # -----------------------------------------------------------
        # Free electron parameters
        for par_name in ["log10_gamma_min", "log10_gamma_max"]:
            if par_name in Flare_model.parameters.names:
                Flare_model.parameters[par_name].frozen = False

        # Physical parameters
        if "delta_D" in Flare_model.parameters.names:
            Flare_model.parameters["delta_D"].frozen = True
        if "log10_B" in Flare_model.parameters.names:
            Flare_model.parameters["log10_B"].frozen = False
        if "t_var" in Flare_model.parameters.names:
            Flare_model.parameters["t_var"].frozen = False

        # -----------------------------------------------------------
        # Run fit
        # -----------------------------------------------------------
        fitter = Fit()
        results = fitter.run(datasets)

        # Compute AIC
        k = len(results.parameters.free_parameters)
        total_stat = results.total_stat
        AIC = 2 * k + total_stat

        # -----------------------------------------------------------
        # Compute reduced chi-square
        # -----------------------------------------------------------
        chi2_total = 0.0
        n_points = 0

        for dataset in datasets:
            t = dataset.data.to_table()

            E = np.array(t["e_ref"].to("eV").value)
            flux_data = np.array(t["e2dnde"].to("erg cm-2 s-1").value)
            flux_err = np.array(t["e2dnde_errn"].to("erg cm-2 s-1").value)

            flux_model = Flare_model(E * u.eV)
            sed_model = ((E * u.eV)**2 * flux_model).to("erg cm-2 s-1").value

            chi2 = np.sum(((flux_data - sed_model) / flux_err) ** 2)
            chi2_total += chi2
            n_points += len(flux_data)

        dof = n_points - k
        red_chi2 = chi2_total / dof if dof > 0 else np.nan

        # -----------------------------------------------------------
        # Save results
        # -----------------------------------------------------------
        fit_results[name] = {
            "result": results,
            "AIC": AIC,
            "chi2": chi2_total,
            "red_chi2": red_chi2,
            "Flare_model": Flare_model
        }

        print(f"{name}: total_stat = {total_stat:.2f}, AIC = {AIC:.2f}, "
              f"χ² = {chi2_total:.2f}, reduced χ² = {red_chi2:.2f}")

    return fit_results


### SAVE MODELS: NO LIV
def save_all_ssc_models(fit_results, output_dir):
    """
    Save SSC models for:
    PowerLaw
    BrokenPowerLaw
    LogParabola
    ExpCutoffPowerLaw
    """

    os.makedirs(output_dir, exist_ok=True)

    model_keys = [
        "PowerLaw",
        "BrokenPowerLaw",
        "LogParabola",
        "ExpCutoffPowerLaw",
    ]

    for key in model_keys:

        ssc_model = get_ssc_model(fit_results, key)

        data = ssc_model_to_dict(ssc_model)

        filename = os.path.join(output_dir, f"{key}_SSC.json")

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved {filename}")
        
def get_ssc_model(fit_results, model_key):
    """
    Extract the SynchrotronSelfComptonSpectralModel from:
    fit_results[model_key]['Flare_model'].model2.model1.model2
    """
    return (
        fit_results[model_key]
        ['Flare_model']
        .model2
        .model1
        .model2
    )


def ssc_model_to_dict(ssc_model):
    """
    Convert SSC model parameters to dictionary.
    """

    return {
        "model_type": "SynchrotronSelfComptonSpectralModel",
        "parameters": [
            {
                "name": p.name,
                "value": p.value,
                "unit": str(p.unit),
                "error": p.error,
                "min": p.min,
                "max": p.max,
                "frozen": p.frozen,
                "is_norm": getattr(p, "is_norm", False),
            }
            for p in ssc_model.parameters
        ],
    }

def import_ssc_model(BB_i, n_e_name, import_dir):
    """
    Load SSC parameters from JSON and construct a full
    AGNpy SynchrotronSelfComptonModel with chosen electron distribution.
    """

    # --------------------------------------------------
    # 1. Load JSON file
    # --------------------------------------------------
    filename = os.path.join(import_dir, f"{n_e_name}_SSC.json")

    with open(filename) as f:
        data = json.load(f)

    # Convert parameter list into dictionary
    pars = {p["name"]: p for p in data["parameters"]}

    # --------------------------------------------------
    # 2. Extract physical values
    # --------------------------------------------------

    k = 10 ** pars["log10_k"]["value"] * u.Unit("cm-3")

    gamma_min = 10 ** pars["log10_gamma_min"]["value"]
    gamma_max = 10 ** pars["log10_gamma_max"]["value"]

    z = pars["z"]["value"]
    delta_blob = pars["delta_D"]["value"]
    Gamma_blob = delta_blob  # assume small viewing angle

    B_blob = 10 ** pars["log10_B"]["value"] * u.G

    t_var = pars["t_var"]["value"] * u.s

    # --------------------------------------------------
    # 3. Build Electron Distribution
    # --------------------------------------------------

    if n_e_name == "PowerLaw":

        n_e = PowerLaw(
            k=k,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            p=pars["p"]["value"],
            mass=m_e,
        )

    elif n_e_name == "BrokenPowerLaw":

        gamma_b = 10 ** pars["log10_gamma_b"]["value"]

        n_e = BrokenPowerLaw(
            k=k,
            gamma_min=gamma_min,
            gamma_b=gamma_b,
            gamma_max=gamma_max,
            p1=pars["p1"]["value"],
            p2=pars["p2"]["value"],
            mass=m_e,
        )

    elif n_e_name == "LogParabola":

        gamma_0 = 10 ** pars["log10_gamma_0"]["value"]

        n_e = LogParabola(
            k=k,
            p=pars["p"]["value"],
            q=pars["q"]["value"],
            gamma_min=gamma_min,
            gamma_0=gamma_0,
            gamma_max=gamma_max,
            mass=m_e,
        )

    elif n_e_name == "ExpCutoffPowerLaw":

        gamma_c = 10 ** pars["log10_gamma_c"]["value"]

        n_e = ExpCutoffPowerLaw(
            k=k,
            gamma_min=gamma_min,
            gamma_c=gamma_c,
            gamma_max=gamma_max,
            p=pars["p"]["value"],
            mass=m_e,
        )

    else:
        raise ValueError(f"Unknown electron distribution: {n_e_name}")

    # --------------------------------------------------
    # 4. Build Blob
    # --------------------------------------------------

    Beta_blob = np.sqrt(1 - 1 / (Gamma_blob ** 2))
    mu_blob = (1 - 1 / (Gamma_blob * delta_blob)) / Beta_blob

    # Compute R from variability time
    R_blob = (t_var * delta_blob * 3e10 * u.cm / u.s) / (1 + z)

    blob = Blob(R_blob, z, delta_blob, Gamma_blob, B_blob, n_e=n_e)

    # --------------------------------------------------
    # 5. Build SSC Model
    # --------------------------------------------------

    ssc_blob_model = SynchrotronSelfComptonModel(
        n_e,
        backend="gammapy",
        ssa=True,
    )

    # Set SSC model parameters explicitly
    ssc_blob_model.parameters["z"].value = z
    ssc_blob_model.parameters["delta_D"].value = delta_blob
    ssc_blob_model.parameters["log10_B"].value = np.log10(
        B_blob.to_value("G")
    )
    ssc_blob_model.parameters["t_var"].value = blob.t_var.to_value("s")

    # --------------------------------------------------
    # 6. Rename model
    # --------------------------------------------------

    model_name = f"ssc_blob_model_{BB_i}_{n_e_name}"
    ssc_blob_model.name = model_name

    print(f"Loaded model: {model_name}")

    return ssc_blob_model
    

# ---------------------------------------------------------
### FIT MODELS WITH LIV EFFECTS
# ---------------------------------------------------------
def fit_BL_Lac_LIV(
    datasets,
    electron_distributions,
    LogParabola_model,
    SmoothBrokenPowerLaw_model_HE,
    LIV_EBL_absorption,
    BB_i,
    import_dir,
    output_dir,
    xi_min=-1e5,
    xi_max=1e5,
    n_scan=40
):

    import os
    import numpy as np
    import pandas as pd

    
    os.makedirs(output_dir, exist_ok=True)

    fit_results = {}

    # ---------------------------------------------------------
    # Build symmetric log-spacing for xi_n
    # ---------------------------------------------------------
#    n_half = n_scan // 2
#
#    xi_neg = -np.logspace(
#        np.log10(abs(xi_min)),
#        np.log10(abs(-1e-1)),
#        n_half
#    )
#
#    xi_pos = np.logspace(
#        np.log10(abs(1e-1)),
#        np.log10(abs(xi_max)),
#        n_half
#    )
#
#    xi_values = np.concatenate((xi_neg, [0.0], xi_pos))

# ---------------------------------------------------------
    # Build custom density grid for xi_n
# ---------------------------------------------------------
    n_outer = int(n_scan * 0.4)  # 40% for the dense zones in the extremes
    n_inner = int(n_scan * 0.2)  # 20% central zone

    # 1. Negative zobe: (-1e5 a -1e3)
    xi_neg_dense = np.linspace(xi_min, -1e3, n_outer)

    # 2. Central Zone: (-1e3 a 1e3) 
    # Usamos menos puntos para que sea "menos densa"
    xi_central = np.linspace(-1e3, 1e3, n_inner)

    # 3. Positive Zone: (1e3 a 1e5)
    xi_pos_dense = np.linspace(1e3, xi_max, n_outer)

    # Concatenate the custon xi_n grid 
    xi_values = np.unique(np.concatenate((xi_neg_dense, xi_central, xi_pos_dense)))

    # ---------------------------------------------------------
    # Loop over electron distributions
    # ---------------------------------------------------------
    for name in electron_distributions.keys():

        print(f"\n=== LIV Fit using {name} electron distribution ===")

        # --------------------------------------------------
        # Import SSC model (best-fit parameters as initial guesses)
        # --------------------------------------------------
        blob_model_BBx = import_ssc_model(
            BB_i=BB_i,
            n_e_name=name,
            import_dir=import_dir
        )

        # --------------------------------------------------
        # Freeze blob physical constants
        # --------------------------------------------------
        fixed_params = ["z", "delta_D", "log10_B", "t_var"]

        for p in fixed_params:
            if p in blob_model_BBx.parameters.names:
                blob_model_BBx.parameters[p].frozen = True

        # All other parameters remain free
        for par in blob_model_BBx.parameters:
            if par.name not in fixed_params:
                par.frozen = False
                
        # --------------------------------------------------
        # Build intrinsic model
        # --------------------------------------------------
        IntrinsicModel_BBx = (
            LogParabola_model
            + SmoothBrokenPowerLaw_model_HE
            + blob_model_BBx
        )

        LIV_model_BBx = IntrinsicModel_BBx * LIV_EBL_absorption

        # LIV parameter
        xi_param = LIV_model_BBx.parameters["xi_n"]
        xi_param.frozen = False
        xi_param.min = xi_min
        xi_param.max = xi_max

        # --------------------------------------------------
        # Assign model to dataset
        # --------------------------------------------------
        sky_model = SkyModel(
            spectral_model=LIV_model_BBx,
            name=f"BL_Lac_LIV_{name}"
        )

        datasets.models = [sky_model]

        # --------------------------------------------------
        # Initial fit (best-fit starting point)
        # --------------------------------------------------
        fitter = Fit()

        print("Running initial fit...")
        results = fitter.run(datasets)

        xi_best = xi_param.value
        xi_err = xi_param.error
        total_stat = results.total_stat

        k = len(results.parameters.free_parameters)
        AIC = 2 * k + total_stat

        print(f"Best xi_n = {xi_best:.3e} ± {xi_err:.3e}")

        # --------------------------------------------------
        # Likelihood scan using stat_profile
        # --------------------------------------------------
        stat_values = []
        print("Running likelihood scan...")

        for xi in xi_values:

            xi_param.value = xi
            xi_param.frozen = True

            res = fitter.run(datasets)

            stat_values.append(res.total_stat)

            xi_param.frozen = False

        stat_values = np.array(stat_values)

        # Convert to ΔTS
        delta_ts = stat_values - np.min(stat_values)

#        print("Running likelihood profile...")
#        profile = fitter.stat_profile(
#            datasets=datasets,
#            parameter=xi_param,
#            values=xi_values
#        )

#        xi_scan = profile["values"]
#        stat_scan = profile["stat"]

#        delta_ts = stat_scan - np.min(stat_scan)

        # --------------------------------------------------
        # Save likelihood profile
        # --------------------------------------------------
        df_scan = pd.DataFrame({
            "xi_n": xi_values,
            "stat": stat_values,
            "delta_ts": delta_ts
        })

        file_name = f"{output_dir}/LIV_scan_{BB_i}_{name}_v1.csv"
        df_scan.to_csv(file_name, index=False)

        print(f"Likelihood scan saved: {file_name}")

        
        # --------------------------------------------------
        # Save results
        # --------------------------------------------------
        fit_results[name] = {
            "result": results,
            "AIC": AIC,
            "xi_n": xi_best,
            "xi_err": xi_err,
            "scan_xi": xi_values,
            "scan_stat": stat_values,
            "scan_deltaTS": delta_ts,
            "LIV_model": LIV_model_BBx
        }

    return fit_results    
    
# ---------------------------------------------------------
# Build optimized xi_n scan (quadratic LIV)
# ---------------------------------------------------------
def build_xi_scan_quadratic(
    xi_min=-1e11,
    xi_max=1e11,
    n_dense=25,
    n_coarse=5
):
    """
    Optimized xi_n scan for quadratic LIV:
    - dense sampling in sensitive regions
    - sparse sampling near xi ~ 0
    """

    # --- Subluminal (negative) ---
    xi_neg = -np.logspace(
        np.log10(1e8),
        np.log10(abs(xi_min)),
        n_dense
    )

    # --- Superluminal (positive) ---
    xi_pos = np.logspace(
        np.log10(1e8),
        np.log10(xi_max),
        n_dense
    )

    # --- Flat region around 0 ---
    xi_flat = np.linspace(-1e8, 1e8, n_coarse)

    # Combine and sort
    xi_values = np.unique(
        np.concatenate((xi_neg, xi_flat, xi_pos))
    )

    return np.sort(xi_values)




def fit_BL_Lac_LIV_quadratic(
    datasets,
    electron_distributions,
    LogParabola_model,
    SmoothBrokenPowerLaw_model_HE,
    LIV_EBL_absorption,
    BB_i,
    import_dir,
    output_dir,
    xi_min=-1e11,
    xi_max=1e11
):

    import os
    import numpy as np
    import pandas as pd


    os.makedirs(output_dir, exist_ok=True)

    fit_results = {}

    # ---------------------------------------------------------
    # Build symmetric log-spacing for xi_n
    # ---------------------------------------------------------
    xi_values = build_xi_scan_quadratic(
        xi_min=xi_min,
        xi_max=xi_max,
        n_dense=20,   # high resolution in sensitive regions
        n_coarse=5    # minimal points near 0
    )

    # ---------------------------------------------------------
    # Loop over electron distributions
    # ---------------------------------------------------------
    for name in electron_distributions.keys():

        print(f"\n=== LIV Fit using {name} electron distribution ===")

        # --------------------------------------------------
        # Import SSC model (best-fit parameters as initial guesses)
        # --------------------------------------------------
        blob_model_BBx = import_ssc_model(
            BB_i=BB_i,
            n_e_name=name,
            import_dir=import_dir
        )

        # --------------------------------------------------
        # Freeze blob physical constants
        # --------------------------------------------------
        fixed_params = ["z", "delta_D", "log10_B", "t_var"]

        for p in fixed_params:
            if p in blob_model_BBx.parameters.names:
                blob_model_BBx.parameters[p].frozen = True

        # All other parameters remain free
        for par in blob_model_BBx.parameters:
            if par.name not in fixed_params:
                par.frozen = False

        # --------------------------------------------------
        # Build intrinsic model
        # --------------------------------------------------
        IntrinsicModel_BBx = (
            LogParabola_model
            + SmoothBrokenPowerLaw_model_HE
            + blob_model_BBx
        )

        LIV_model_BBx = IntrinsicModel_BBx * LIV_EBL_absorption

        # LIV parameter
        xi_param = LIV_model_BBx.parameters["xi_n"]
        xi_param.frozen = False
        xi_param.min = xi_min
        xi_param.max = xi_max

        # --------------------------------------------------
        # Assign model to dataset
        # --------------------------------------------------
        sky_model = SkyModel(
            spectral_model=LIV_model_BBx,
            name=f"BL_Lac_LIV_{name}"
        )

        datasets.models = [sky_model]

        # --------------------------------------------------
        # Initial fit (best-fit starting point)
        # --------------------------------------------------
        fitter = Fit()

        print("Running initial fit...")
        results = fitter.run(datasets)

        xi_best = xi_param.value
        xi_err = xi_param.error
        total_stat = results.total_stat

        k = len(results.parameters.free_parameters)
        AIC = 2 * k + total_stat

        print(f"Best xi_n = {xi_best:.3e} ± {xi_err:.3e}")

        # --------------------------------------------------
        # Likelihood scan using stat_profile
        # --------------------------------------------------
        stat_values = []
        print("Running likelihood scan...")

        for xi in xi_values:

            xi_param.value = xi
            xi_param.frozen = True

            res = fitter.run(datasets)

            stat_values.append(res.total_stat)

            xi_param.frozen = False

        stat_values = np.array(stat_values)

        # Convert to ΔTS
        delta_ts = stat_values - np.min(stat_values)

#        print("Running likelihood profile...")
#        profile = fitter.stat_profile(
#            datasets=datasets,
#            parameter=xi_param,
#            values=xi_values
#        )

#        xi_scan = profile["values"]
#        stat_scan = profile["stat"]

#        delta_ts = stat_scan - np.min(stat_scan)

        # --------------------------------------------------
        # Save likelihood profile
        # --------------------------------------------------
        df_scan = pd.DataFrame({
            "xi_n": xi_values,
            "stat": stat_values,
            "delta_ts": delta_ts
        })

        file_name = f"{output_dir}/Quadratic_LIV_scan_{BB_i}_{name}_v4.csv"
        df_scan.to_csv(file_name, index=False)

        print(f"Likelihood scan saved: {file_name}")

        
        # --------------------------------------------------
        # Save results
        # --------------------------------------------------
        fit_results[name] = {
            "result": results,
            "AIC": AIC,
            "xi_n": xi_best,
            "xi_err": xi_err,
            "scan_xi": xi_values,
            "scan_stat": stat_values,
            "scan_deltaTS": delta_ts,
            "LIV_model": LIV_model_BBx
        }

    return fit_results
