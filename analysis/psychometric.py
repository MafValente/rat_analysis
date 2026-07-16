import numpy as np
import matplotlib.pyplot as plt
from psychofit import mle_fit_psycho, weibull, weibull50, erf_psycho, erf_psycho_2gammas
from scipy.optimize import curve_fit
from scipy.special import erf

MIN_ILD_POINTS_FOR_FIT = 4

def erf_4par(x, alpha, beta, gamma, lambda_):
    """4-parameter cumulative Gaussian psychometric function."""
    return gamma + (1 - gamma - lambda_) * 0.5 * (1 + erf((x - alpha) / (beta * np.sqrt(2))))

# ------------------------
# Base wrapper

def my_psycho_model(x, a, b, c, d):
    """
    Custom psychometric function.

    Example: 4-parameter logistic
    b = midpoint (bias)
    a = slope
    c = lower asymptote (gamma)
    d = upper asymptote (lambda)
    """
    return c + (d-c) / (1 + np.exp(-(2*a) * (x-b)))


def fit_and_plot_psychometric(xData, yData, model,
                              n_trials,
                              parstart=None, parmin=None, parmax=None,
                              show_plot=True, ax=None, save_path=None):
    """
    Wrapper around psychofit.mle_fit_psycho that takes xData = ILDs
    and yData = proportion of leftward trials at each ILD.
    
    Parameters
    ----------
    xData : array-like
        Unique stimulus values (ILDs).
    yData : array-like
        Proportion of leftward choices (0..1) at each ILD.
    model : str
        Psychometric function to fit: "weibull", "weibull50", 
        "erf_psycho", or "erf_psycho_2gammas".
    n_trials : array-like or int, optional
        Number of trials at each ILD. If None, assume 50 per condition.
    parstart, parmin, parmax : list or None
        Fit initialisation and bounds. If None, defaults are used.
    show_plot : bool
        If True, plot the fit.
    ax : matplotlib axis, optional
        Axis to plot into (if None, creates new figure).
    save_path : str, optional
        If provided, save the plot instead of showing.
    
    Returns
    -------
    pars : np.ndarray
        Best-fit parameters.
    L : float
        Likelihood of the best fit.
    """
    
    xData = np.asarray(xData)
    yData = np.asarray(yData)

    # set number of trials if not provided
    if n_trials is None:
        n_trials = np.full_like(xData, 50, dtype=int)  # assume 50 per ILD
    else:
        n_trials = np.asarray(n_trials)
        

    # construct data matrix: 3 x N
    data = np.vstack((xData, n_trials, yData))

    #print("ILDs:", xData)
    #print("PropLeft:", yData)
    #print("n_trials:", n_trials)

    if parstart is None:
        if model.startswith("weibull"):
            parstart = [np.median(xData), 2.0, 0.05]   # [threshold, slope, lapse]
            parmin   = [min(xData), 0.1, 0]
            parmax   = [max(xData), 10.0, 0.5]
        elif model == "erf_psycho":
            parstart = [0.0, 5.0, 0.05]                # [bias, slope, lapse]
            parmin   = [min(xData), 0.1, 0]
            parmax   = [max(xData), 10.0, 0.5]
        elif model == "erf_psycho_2gammas":
            parstart = [0.0, 5.0, 0.05, 0.05]
            parmin   = [min(xData), 0.1, 0, 0]
            parmax   = [max(xData), 10.0, 0.5, 0.5]
        elif model == "erf_4par":
            parstart = [0.0, 5.0, 0.05, 0.05]  # [bias, slope, γ, λ]
            parmin   = [min(xData), 0.1, 0, 0]
            parmax   = [max(xData), 10.0, 0.5, 0.5]
        elif model=="my_psycho":
            parstart = [1, 0, 0.05, 0.95]  # [slope, bias, c, d]
            parmin   = [0.1, min(xData), 0, 0.5]
            parmax   = [10.0, max(xData), 0.5, 1]

    # fit
    """ 
    if model == "erf_4par":
        p0 = [0.0, 5.0, 0.05, 0.05]
        bounds = ([min(xData), 0.1, 0, 0],
                [max(xData), 10.0, 0.5, 0.5])
        pars, _ = curve_fit(erf_4par, xData, yData, p0=p0, bounds=bounds)
        L = np.nan
    elif model == "my_psycho":
        p0 = parstart
        bounds = (parmin,
                parmax)
        pars, _ = curve_fit(my_psycho_model, xData, yData, p0=p0, bounds=bounds)
        L = np.nan
    else:
        pars, L = mle_fit_psycho(
            np.vstack((xData, n_trials, yData)),
            P_model=model,
            parstart=np.array(parstart),
            parmin=np.array(parmin),
            parmax=np.array(parmax))
    """
    xx = np.linspace(min(xData), max(xData), 200)

    dispatcher = {
        "weibull": weibull,
        "weibull50": weibull50,
        "erf_psycho": erf_psycho,
        "erf_psycho_2gammas": erf_psycho_2gammas,
        "erf_4par": erf_4par,
        "my_psycho": my_psycho_model,   # 👈 add your custom model
    }
    psyfun = dispatcher[model]

    if model in ["erf_4par", "my_psycho"]:
        p0 = [1.0, 0.0, 0.05, 0.95]  # adjust to your model’s parameter meaning
        bounds = ([0.1, min(xData), 0, 0.5],
                [10.0, max(xData), 0.5, 1])
        psyfun = dispatcher[model]
        pars, _ = curve_fit(psyfun, xData, yData, p0=p0, bounds=bounds)
        L = np.nan
    else:
        pars, L = mle_fit_psycho(
            np.vstack((xData, n_trials, yData)),
            P_model=model,
            parstart=np.array(parstart),
            parmin=np.array(parmin),
            parmax=np.array(parmax))


    if model in ["erf_4par", "my_psycho"]:
        yy = psyfun(xx, *pars)
    else:
        yy = psyfun(pars, xx)


    # optional plot
    if show_plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(xData, yData, color="k", label="Data")
        ax.plot(xx, yy, "r-", label=f"Fit ({model})")
        ax.set_xlabel("ILD")
        ax.set_ylabel("Proportion Right")
        ax.set_ylim(0, 1)
        ax.legend()

        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()

    return pars, L, xx, yy

# ------------------------
# Higher-level function: psychometrics by ABL

def compute_psychometrics_by_ABL(df_last, model="my_psycho", min_ilds_for_fit=4):
    """
    Compute psychometric fits separated by ABL.

    Parameters
    ----------
    df_last : pd.DataFrame
        Data for the last session (must contain columns ILD, ABL, success).
    model : str
        Psychometric model to fit (default: "erf_psycho").

    Returns
    -------
    results : dict
        Dictionary keyed by ABL value. Each entry contains:
        {
          "ILDs": array of stimulus levels,
          "PropLeft": array of proportions left,
          "n_trials": array of trial counts,
          "pars": fit parameters,
          "L": likelihood
        }
    """
    results = {}

    for abl in sorted(df_last["ABL"].unique()):
        df_sub = df_last[df_last["ABL"] == abl]
        # print(f"\nABL={abl}, n_trials={len(df_sub)}")

        ILDs = np.sort(df_sub["ILD"].unique())

        # Always compute POINTS if there are any ILDs at all
        if len(ILDs) == 0:
            print("  ⚠️ Skipping ABL because no ILDs found")
            continue


        PropLeft = np.array([
            (((df_sub["ILD"] == ild) & (df_sub["success"] == 1)).sum()) /
            (((df_sub["ILD"] == ild) & (df_sub["success"] != 0)).sum())
            for ild in ILDs
        ], dtype=float)

        # Flip for negative ILDs (keep your behavior)
        PropLeft = np.where(ILDs < 0, 1 - PropLeft, PropLeft)

        n_trials = np.array([(df_sub["ILD"] == ild).sum() for ild in ILDs], dtype=int)

        # Default: no fit
        pars, L, xx, yy = None, np.nan, None, None

# Fit ONLY if enough ILDs (your rule)
        if len(ILDs) >= min_ilds_for_fit and np.isfinite(PropLeft).all():
            try:
                pars, L, xx, yy = fit_and_plot_psychometric(
                    ILDs, PropLeft,
                    model=model,
                    n_trials=n_trials,
                    show_plot=False,
                )
            except Exception:
                # keep points; just no fit
                pars, L, xx, yy = None, np.nan, None, None

        results[abl] = {
            "ILDs": ILDs,
            "PropLeft": PropLeft,
            "n_trials": n_trials,
            "pars": pars,
            "L": L,
            "xx": xx,
            "yy": yy,
        }

    return results
