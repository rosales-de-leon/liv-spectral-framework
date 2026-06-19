import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_combined_liv_profiles(scans, combined_profiles):

    model_names = list(combined_profiles.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(
        n_models, 1,
        figsize=(7, 4 * n_models),
        sharex=True
    )

    if n_models == 1:
        axes = [axes]
        
    # color map for nights
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for ax, model in zip(axes, model_names):

        dfs = scans[model]

        # --------------------------------------------------
        # Plot individual night profiles
        # --------------------------------------------------
        for i, df in enumerate(dfs):

            xi = df["xi_n"].values
            deltaTS = df["delta_ts"].values

            ax.plot(
                xi,
                deltaTS,
                color=colors[i],
                alpha=0.3,
                lw=1,
                label=BB_labels[i]
            )

        # --------------------------------------------------
        # Plot combined profile
        # --------------------------------------------------
        prof = combined_profiles[model]

        ax.plot(
            prof["xi"],
            prof["deltaTS"],
            color="red",
            lw=2.5,
            label="Combined likelihood"
        )

        # 95% CL line
        ax.axhline(
            2.71,
            color="black",
            ls="--",
            label="95% CL"
        )

        ax.set_xscale("symlog")

        ax.set_ylabel(r"$\Delta TS$")
        ax.set_title(f"LIV Likelihood — {model}")

        ax.grid(alpha=0.3)
        ax.legend(ncol=2, fontsize=8, loc=2)

    axes[-1].set_xlabel(r"$\xi_1$")

    plt.tight_layout()
    plt.show()

def plot_final_stacked_profile(dataset_profiles, combined_profile):

    plt.figure(figsize=(7,5))

    # individual nights
    for bb, prof in dataset_profiles.items():

        plt.plot(
            prof["xi"],
            prof["deltaTS"],
            alpha=0.6,
            lw=1,
            label=bb
        )

    # combined
    plt.plot(
        combined_profile["xi"],
        combined_profile["deltaTS"],
        color="black",
        lw=2.5,
        label="Combined"
    )

    plt.axhline(2.71, ls="--", color="black", label="95% CL")

    plt.axvline(0, color="gray")

    #plt.xscale("symlog")

    plt.xlabel(r"$\xi_1$")
    plt.ylabel(r"$\Delta TS$")

    #plt.title("Stacked LIV Likelihood Profile")

    plt.legend(ncol=2, fontsize=8)

    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

# ============================================================
# MAIN PLOTTING FUNCTION
# ============================================================
def plot_constraints_updated(
    constraints,
    your_sub,
    your_sup,
    order=1,
    confidence_label="95% CL"
):

    # ============================================================
    # PLANCK SCALE
    # ============================================================
    E_PLANCK = 1.22e28  # eV


    # ============================================================
    # UPDATED LITERATURE TABLE
    #
    # Structure:
    # (effect, source, type, distance, method, instrument,
    #  E_QG1 [GeV], E_QG2 [GeV])
    # ============================================================

    literature_constraints = [

        # --------------------------------------------------------
        # Time-delay / ToF
        # --------------------------------------------------------

        ("Time delay", "Mrk 421", "AGN", "z=0.031",
         "band comparison", "Whipple",
         0.4e17, None),

    #    ("Time delay", "Mrk 501", "AGN", "z=0.034",
    #     "ECF", "MAGIC",
    #     2.1e17, 2.6e10),

        ("Time delay", "Mrk 501", "AGN", "z=0.034",
         "ML", "MAGIC",
         3.0e17, 5.7e10),

    #    ("Time delay", "PKS 2155-304", "AGN", "z=0.116",
    #     "MCCF/CWT", "H.E.S.S.",
    #     7.2e17, 1.4e9),

        ("Time delay", "PKS 2155-304", "AGN", "z=0.116",
         "ML", "H.E.S.S.",
         2.1e18, 6.4e10),

        ("Time delay", "GRB 090510", "GRB", "z=0.9",
         "PV/SMM/ML", "Fermi-LAT",
         2.2e19, 4.0e10),
        
        ("Time delay", "Multiple Sources (8)", "GRB", "z=0.9", 
         "PV/SMM/ML", "Fermi-LAT",
         1.0e17, None),

        ("Time delay", "Crab", "Pulsar", "d=2 kpc",
         "PC", "VERITAS",
         3.0e17, 7.0e9),

        ("Time delay", "PG 1553+113", "AGN", "z=0.49",
         "ML", "H.E.S.S.",
         4.1e17, 2.1e10),

        ("Time delay", "Vela", "Pulsar", "d=0.3 kpc",
         "ML", "H.E.S.S.",
         4.0e15, None),

        ("Time delay", "Crab", "Pulsar", "d=2 kpc",
         "ML", "MAGIC",
         5.5e17, 5.9e10),

        ("Time delay", "Mrk 501", "AGN", "z=0.034",
         "ML", "H.E.S.S.",
         3.6e17, 8.5e10),

        ("Time delay", "GRB 190114C", "GRB", "z=0.4245",
         "ML", "MAGIC",
         5.8e18, 6.3e10),

        # --------------------------------------------------------
        # Universe transparency
        # --------------------------------------------------------

        ("Universe transparency", "Multiple Sources (30)", "AGN",
         "z=0.019-0.287", "TS", "Multiple Inst",
         8.6e18, None),

        ("Universe transparency", "Mrk 501", "AGN",
         "z=0.034", "TS", "H.E.S.S.",
         2.6e19, 7.8e11),

    #    ("Universe transparency", "Multiple Sources (6)", "AGN",
    #     "z=0.031-0.188", "TS", "Multiple Inst",
    #     6.9e19, 1.6e12),

        # --------------------------------------------------------
        # Bethe-Heitler
        # --------------------------------------------------------

    #    ("Bethe-Heitler", "Crab", "Nebula",
    #     "d=2 kpc", "ML", "HEGRA/H.E.S.S.",
    #     None, 2.1e11),

        # --------------------------------------------------------
        # Photon decay / splitting
        # --------------------------------------------------------

    #    ("Photon decay", "Multiple (4)", "Galactic",
    #     "d=1.55-2.37 kpc", "TS", "HAWC",
    #     2.2e22, 0.8e14),

    #    ("Photon splitting", "J2032+4102", "Cluster",
    #     "d=1.4 kpc", "TS", "LHAASO",
    #     1.2e24, 1.1e15),
    ]
    
    # ============================================================
    # MARKERS BY EFFECT
    # ============================================================

    effect_markers = {
        "Time delay": "o",
        "Universe transparency": "s",
    #    "Bethe-Heitler": "^",
    #    "Photon decay": "D",
    #    "Photon splitting": "P",
    }


    # ============================================================
    # COLORS BY SOURCE TYPE
    # ============================================================

    type_colors = {
        "AGN": "gray",
        "GRB": "dimgray",
        "Pulsar": "silver",
        "Nebula": "darkgray",
        "Galactic": "black",
        "Cluster": "slategray",
    }

    fig, ax = plt.subplots(figsize=(12, 9))

    # --------------------------------------------------------
    # Select linear or quadratic column
    # --------------------------------------------------------
    idx = 6 if order == 1 else 7

    selected = []

    for c in constraints:
        if c[idx] is not None:
            selected.append(c)

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------
    labels = []

    for c in selected:

        effect, source, typ, dist, method, instr, _, _ = c

        label = (
            f"{source} ({instr})"
            #f"{effect}, {dist}, {method}"
        )

        labels.append(label)

    labels += [
        f"BL Lac (MWL + LST-1; subluminal)",
        f"BL Lac (MWL + LST-1; superluminal)"
    ]

    y = np.arange(len(labels))

    
    # --------------------------------------------------------
    # Convert Sub/Sup Limits:  eV -> GeV
    # --------------------------------------------------------
    EV_TO_GEV = 1e-9

    E_PLANCK_GEV = 1.22e28 * EV_TO_GEV

    your_sub *= EV_TO_GEV
    your_sup *= EV_TO_GEV

    # --------------------------------------------------------
    # Plot literature constraints
    # --------------------------------------------------------
    for i, c in enumerate(selected):

        effect, source, typ, dist, method, instr, E1, E2 = c

        val = E1 if order == 1 else E2

        marker = effect_markers.get(effect, "o")
        color = type_colors.get(typ, "gray")

        # lower-limit line
        ax.hlines(
            y=i,
            xmin=val,
            xmax=1e25,
            color=color,
            lw=2,
            linestyle="--",
            alpha=0.8
        )

        ax.plot(
            val,
            i,
            marker=marker,
            color=color,
            markersize=8
        )

    # --------------------------------------------------------
    # Your results
    # --------------------------------------------------------
    idx_sub = len(selected)
    idx_sup = len(selected) + 1

    # subluminal
    ax.hlines(
        idx_sub,
        your_sub,
        1e25,
        color="tab:blue",
        lw=3
    )

    ax.plot(
        your_sub,
        idx_sub,
        marker=">",
        color="tab:blue",
        markersize=12
    )

    # superluminal
    ax.hlines(
        idx_sup,
        your_sup,
        1e25,
        color="tab:red",
        lw=3
    )

    ax.plot(
        your_sup,
        idx_sup,
        marker=">",
        color="tab:red",
        markersize=12
    )

    # --------------------------------------------------------
    # Planck scale
    # --------------------------------------------------------
    if order == 1:
        ax.axvline(
            E_PLANCK_GEV,
            color="black",
            linestyle=":",
            lw=2
        )

        ax.text(
            E_PLANCK_GEV * 1.1,
            0.02,
            r"$E_{\rm Planck}$",
            rotation=0,
            va="bottom",
            fontsize=12
        )
        
    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------
    ax.set_xscale("log")

    if order == 1:

        ax.set_xlim(1e14, 1e21)
        ax.tick_params(axis='x', labelsize=14)
        ax.set_title(r"Linear LIV Constraints ($n=1$)",fontsize=15)
        ax.set_xlabel(r"Lower limit on $E_{\rm QG, 1}$ [GeV]", fontsize=15)
        
        
    else:
        ax.set_xlim(1e8, 1e14)
        ax.tick_params(axis='x', labelsize=14)
        ax.set_title(r"Quadratic LIV Constraints ($n=2$)", fontsize=15)
        ax.set_xlabel(r"Lower limit on $E_{\rm QG, 2}$ [GeV]", fontsize=15)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=14)
    ax.grid(True, which="both", alpha=0.3)
    
    

    # --------------------------------------------------------
    # Effect legend
    # --------------------------------------------------------
    legend_effects = [

        Line2D(
            [0], [0],
            marker=marker,
            color='gray',
            linestyle='None',
            markersize=8,
            label=effect
        )

        for effect, marker in effect_markers.items()
    ]

    # --------------------------------------------------------
    # This work legend
    # --------------------------------------------------------
    legend_work = [

        Line2D(
            [0], [0],
            color='tab:red',
            lw=3,
            label='This work (superluminal)'
        ),

        Line2D(
            [0], [0],
            color='tab:blue',
            lw=3,
            label='This work (subluminal)'
        ),
    ]

    ax.legend(
        handles=legend_effects + legend_work,
        fontsize=12,
        loc="lower left"
    )

    # confidence label
    ax.text(
        0.98,
        0.02,
        confidence_label,
        transform=ax.transAxes,
        ha="right",
        fontsize=14
    )
    
    plt.tight_layout()
    plt.show()
    
    
# ---------------------------------------------------------
### FRANCESCHINI EBL+LIV PLOT FUNCTION
# ---------------------------------------------------------
def plot_franceschini_results(E_gamma_vals, xi_n_array, results_fra, z, n_order):
    """
    Grafica la opacidad (tau) cargando los datos desde results_fra.
    Las curvas superluminales explotan verticalmente al alcanzar el umbral.
    """
    plt.figure(figsize=(8, 7))
    
    # Energías en TeV para el eje X
    energy_tev = E_gamma_vals.to(u.TeV).value
    
    # Eje X original
    x_raw = E_gamma_vals.to(u.TeV).value
    # Eje X suavizado (1000 puntos para que se vea curvo)
    x_smooth = np.logspace(np.log10(x_raw.min()), np.log10(x_raw.max()), 1000)
    
    # Definición de colores (Azules para subluminal, Rojos para superluminal)
    # Filtramos xi para las paletas
    sub_vals = [x for x in xi_n_array if x < 0]
    sup_vals = [x for x in xi_n_array if x > 0]
    
    colors_sub = plt.cm.Blues(np.linspace(0.5, 0.9, len(sub_vals)))
    colors_sup = plt.cm.Reds(np.linspace(0.5, 0.9, len(sup_vals)))
    
    sub_count = 0
    sup_count = 0

    for i, xi in enumerate(xi_n_array):
        y_raw = np.array(results_fra[i])
        
        # --- CASO ESTÁNDAR O SUBLUMINAL (Suavizado Spline) ---
        if xi <= 0:
            # Interpolación en espacio log-log para mayor estabilidad
            spline = make_interp_spline(np.log10(x_raw), np.log10(np.maximum(y_raw, 1e-10)), k=1)
            y_smooth = 10**spline(np.log10(x_smooth))
            
            color = 'black' if xi == 0 else colors_sub[sub_count]
            ls = '-'
            label = "Standard EBL" if xi == 0 else f"$\\xi_n = {xi:.0e}$"
            if xi < 0: sub_count += 1
            
            plt.plot(x_smooth, y_smooth, color=color, ls=ls, lw=2, label=label)

        # --- CASO SUPERLUMINAL (Suavizado + Explosión) ---
        else:
            color = colors_sup[sup_count]
            sup_count += 1
            
            # Buscamos el punto de explosión
            idx_max = np.argmax(y_raw)
            # Solo suavizamos hasta antes de la explosión
            x_to_smooth = x_raw[:idx_max+1]
            y_to_smooth = y_raw[:idx_max+1]
            
            if len(x_to_smooth) > 3: # Necesitamos puntos suficientes para el spline (k=3)
                x_smooth_lim = np.logspace(np.log10(x_to_smooth.min()), np.log10(x_to_smooth.max()), 1000)
                spline = make_interp_spline(np.log10(x_to_smooth), np.log10(np.maximum(y_to_smooth, 1e-10)), k=1)
                y_smooth_lim = 10**spline(np.log10(x_smooth_lim))
                
                # Graficamos la parte curva
                plt.plot(x_smooth_lim, y_smooth_lim, color=color, ls='-', lw=2)
                # Graficamos la explosión vertical al final
                plt.vlines(x_to_smooth[-1], y_to_smooth[-1], 1e8, color=color, ls='-', lw=2, label=f"$\\xi_n = {xi:.0e}$")
            else:
                # Si hay muy pocos puntos, graficar normal para evitar errores de spline
                plt.plot(x_raw, y_raw, color=color, ls='--', lw=1.8, label=f"$\\xi_n = {xi:.0e}$ (Sup)")
    
    
    # Configuración de ejes y estética
    plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 20,
    "legend.fontsize": 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    })
    
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlim(1e-2, 1e2)
    plt.ylim(1e-3, 1e4) # Límite superior para visualizar la explosión
    
    plt.xlabel(r'$E_{\gamma}$ [TeV]', fontsize=18)
    plt.ylabel(r'$\tau(E_{\gamma})$', fontsize=18)
    plt.title(f'Franceschini Model Opacity (z={z}, n={n_order})', fontsize=18)
    
    plt.grid(True, which="both", ls="--", alpha=0.35)
    
    # Leyenda en dos columnas para mejor visibilidad
    plt.legend(ncol=1, fontsize=10, loc='upper left')
    
    plt.tight_layout()
    plt.show()
