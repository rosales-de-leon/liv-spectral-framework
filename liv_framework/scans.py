def combine_liv_profiles(scans):

    combined_profiles = {}

    for model, dfs in scans.items():

        if len(dfs) == 0:
            continue

        # assume identical xi grid
        xi_values = dfs[0]["xi_n"].values
        
        stat_total = np.zeros_like(xi_values)

        for df in dfs:
            stat_total += df["stat"].values

        delta_ts = stat_total - np.min(stat_total)

        combined_profiles[model] = {
            "xi": xi_values,
            "stat": stat_total,
            "deltaTS": delta_ts
        }

    return combined_profiles    
    
def compute_dataset_min_profiles(scan_dir, BB_labels, model_names):

    dataset_profiles = {}

    for bb in BB_labels:

        dfs = []

        for model in model_names:

            file = Path(scan_dir) / f"LIV_scan_{bb}_{model}_v1.csv"

            if file.exists():
                dfs.append(pd.read_csv(file))

        if len(dfs) == 0:
            continue

        xi = dfs[0]["xi_n"].values

        deltaTS_models = []

        for df in dfs:
            deltaTS_models.append(df["delta_ts"].values)

        deltaTS_models = np.array(deltaTS_models)

        # minimum ΔTS across models
        deltaTS_min = np.min(deltaTS_models, axis=0)

        dataset_profiles[bb] = {
            "xi": xi,
            "deltaTS": deltaTS_min
        }

    return dataset_profiles


def combine_dataset_profiles(dataset_profiles):

    BB_labels = list(dataset_profiles.keys())

    xi = dataset_profiles[BB_labels[0]]["xi"]

    deltaTS_total = np.zeros_like(xi)

    for bb in BB_labels:
        deltaTS_total += dataset_profiles[bb]["deltaTS"]

    deltaTS_total -= np.min(deltaTS_total)

    combined_profile = {
        "xi": xi,
        "deltaTS": deltaTS_total
    }

    return combined_profile
    
def compute_xi_confidence_interval(xi, delta_ts, delta_ts_level=2.71):
    """
    Compute confidence intervals on xi_n from ΔTS profile.

    Parameters
    ----------
    xi : array
        xi_n scan values
    delta_ts : array
        ΔTS values
    delta_ts_level : float
        threshold (e.g. 2.71 for 90% CL)

    Returns
    -------
    dict with:
        xi_best
        xi_err_low
        xi_err_high
        xi_low_bound
        xi_high_bound
    """

    xi = np.array(xi)
    delta_ts = np.array(delta_ts)

    # --------------------------------------------------
    # Best-fit (minimum ΔTS)
    # --------------------------------------------------
    idx_min = np.argmin(delta_ts)
    xi_best = xi[idx_min]

    # Shift profile (just in case)
    delta_ts = delta_ts - np.min(delta_ts)

    # --------------------------------------------------
    # Split regions
    # --------------------------------------------------
    mask_left = xi < xi_best
    mask_right = xi > xi_best

    xi_left = xi[mask_left]
    ts_left = delta_ts[mask_left]

    xi_right = xi[mask_right]
    ts_right = delta_ts[mask_right]

    # --------------------------------------------------
    # Interpolation
    # --------------------------------------------------
    xi_low_bound = np.nan
    xi_high_bound = np.nan

    try:
        if len(xi_left) > 2:
            f_left = interp1d(ts_left, xi_left, bounds_error=False)
            xi_low_bound = f_left(delta_ts_level)
    except:
        pass

    try:
        if len(xi_right) > 2:
            f_right = interp1d(ts_right, xi_right, bounds_error=False)
            xi_high_bound = f_right(delta_ts_level)
    except:
        pass

    # --------------------------------------------------
    # Errors relative to best-fit
    # --------------------------------------------------
    xi_err_low = xi_best - xi_low_bound if not np.isnan(xi_low_bound) else np.nan
    xi_err_high = xi_high_bound - xi_best if not np.isnan(xi_high_bound) else np.nan

    return {
        "xi_best": xi_best,
        "xi_low": xi_low_bound,
        "xi_high": xi_high_bound,
        "err_low": xi_err_low,
        "err_high": xi_err_high,
    }
    
def compute_one_sided_limit(xi, delta_ts, level=2.71, side="positive"):
    
    xi = np.array(xi)
    delta_ts = delta_ts - np.min(delta_ts)

    if side == "positive":
        mask = xi > 0
    else:
        mask = xi < 0

    xi_side = xi[mask]
    ts_side = delta_ts[mask]

    try:
        f = interp1d(ts_side, xi_side, bounds_error=False)
        return f(level)
    except:
        return np.nan
        `
