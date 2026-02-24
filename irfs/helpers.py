"""
Helper functions for the IRF hands-on notebook.

This module keeps the notebook lightweight by hosting the longer utilities that:
- convert CSV tables to GADF/Gammapy-style IRFs (AEFF / EDISP / PSF)
- build background IRFs from CSV / from other IRFs

Place this file in the same folder as the notebook (or add it to PYTHONPATH).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter

import numpy as np
# helpers.py
import numpy as np
import astropy.units as u
from gammapy.irf import Background2D


# -----------------------------
# Generic utilities
# -----------------------------
def seconds_per_year() -> float:
    return 365.0 * 24.0 * 3600.0


def get_energy_axis_from_bkg2d(bkg2d: Background2D):
    """
    Return energy edges [Quantity], centers [Quantity], centers in TeV [ndarray], and dE in GeV [ndarray].
    """
    energy_axis = bkg2d.axes["energy"]
    e_edges = energy_axis.edges  # Quantity
    e_centers = np.sqrt(e_edges[:-1] * e_edges[1:])  # Quantity
    E_reco_TeV = e_centers.to_value(u.TeV)
    dE_GeV = np.diff(e_edges).to_value(u.GeV)
    return e_edges, e_centers, E_reco_TeV, dE_GeV


def get_theta_geometry_from_bkg2d(bkg2d: Background2D, theta_cut_deg: float = 80.0):
    """
    From bkg2d 'offset' axis compute:
      - theta_edges (Quantity)
      - theta_centers (Quantity)
      - mask: theta_centers > theta_cut_deg
      - dOmega per theta bin (ndarray), using dΩ = 2π (cos θ_lo - cos θ_hi)
    """
    theta_axis = bkg2d.axes["offset"]
    theta_edges = theta_axis.edges  # Quantity
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    mask = theta_centers.to_value(u.deg) > theta_cut_deg

    theta_edges_rad = theta_edges.to_value(u.rad)
    dOmega = 2.0 * np.pi * (np.cos(theta_edges_rad[:-1]) - np.cos(theta_edges_rad[1:]))

    return theta_edges, theta_centers, mask, dOmega


# -----------------------------
# Neutrinos
# -----------------------------
def integrate_over_theta_only(rate_2d, dOmega, mask):
    """
    Integrate a (nE, nTheta) differential rate over selected theta bins,
    keeping the energy dependence.

    Parameters
    ----------
    rate_2d : array-like
        Shape (nE, nTheta) OR (nE, 1)
    dOmega : ndarray
        Shape (nTheta_geom,)
    mask : ndarray[bool]
        Shape (nTheta_geom,)

    Returns
    -------
    rate_1d : ndarray
        Shape (nE,) -- integrated over solid angle selection
    """
    rate_2d = np.asarray(rate_2d)
    if rate_2d.ndim != 2:
        raise ValueError(f"rate_2d must be 2D, got shape {rate_2d.shape}")

    nE, nTheta_rate = rate_2d.shape
    Omega_sel = dOmega[mask].sum()

    if nTheta_rate == 1:
        # treat as per-sr with no theta binning
        return rate_2d[:, 0] * Omega_sel

    if nTheta_rate != dOmega.shape[0]:
        raise ValueError(
            f"Theta mismatch: rate has nTheta={nTheta_rate}, but dOmega has {dOmega.shape[0]}"
        )

    return (rate_2d[:, mask] * dOmega[mask][None, :]).sum(axis=1)


def neutrino_events_per_year_per_bin(atm_rate_2d, dE_GeV, dOmega, mask, t_year_s=None):
    """
    Convert differential atm rate (per GeV) into events/year per energy bin,
    integrating over theta selection.

    Returns
    -------
    events_year : ndarray (nE,)
    """
    if t_year_s is None:
        t_year_s = seconds_per_year()

    rate_E = integrate_over_theta_only(atm_rate_2d, dOmega, mask)  # (nE,)
    return rate_E * np.asarray(dE_GeV) * t_year_s

def read_muon_horizon_from_bkg_fits(
    bkg_fits_path: str,
    horizon_index: int | None = None,
    smooth_sigma_decades: float | None = 0.20,
    smooth_in_logE_func=None,
):
    """
    Read muon Background2D FITS and extract the horizon declination/offset band.

    Assumptions (consistent with the sin(dec)-based builder):
      - Background2D has axes: "energy" and one angular axis (named e.g. "offset").
      - Angular axis stores declination-like edges in degrees (derived from sin(dec)).
      - Data are in units convertible to "s-1 MeV-1 sr-1" or already a plain ndarray
        in those units.

    Returns
    -------
    E_edges_GeV : ndarray (nE+1,)
    E_centers_TeV : ndarray (nE,)
    mu_rate_bin_s : ndarray (nE,)   # per energy bin, in s^-1, for the chosen band
    mu_rate_bin_s_smooth : ndarray (nE,) or None
    offset_edges_deg : ndarray (nOffset+1,)
    chosen_index : int
    """
    import numpy as np
    from gammapy.irf import Background2D

    # -------------------------------------------------------------------------
    # Read Background2D
    # -------------------------------------------------------------------------
    bkg2 = Background2D.read(bkg_fits_path)

    # Energy axis
    if "energy" not in bkg2.axes.names:
        raise KeyError(f"Background2D axes do not contain 'energy': {bkg2.axes}")

    E_edges_GeV = bkg2.axes["energy"].edges.to_value("GeV")
    nE = len(E_edges_GeV) - 1

    # Angular axis: take the first non-energy axis
    ang_axes = [ax for ax in bkg2.axes if ax.name != "energy"]
    if len(ang_axes) != 1:
        raise ValueError(
            f"Expected exactly one angular axis, found {len(ang_axes)}: {bkg2.axes}"
        )
    offset_axis = ang_axes[0]

    # Interpret this axis as declination-like in degrees
    offset_edges_deg = offset_axis.edges.to_value("deg")
    nOffset = len(offset_edges_deg) - 1

    # Pick horizon index: by default, middle band
    if horizon_index is None:
        horizon_index = nOffset // 2
    if not (0 <= horizon_index < nOffset):
        raise IndexError(
            f"horizon_index={horizon_index} outside [0, {nOffset-1}]"
        )

    # -------------------------------------------------------------------------
    # Solid angle for declination bands: dΩ = 2π (sin_dec_hi - sin_dec_lo)
    # -------------------------------------------------------------------------
    offset_edges_rad = np.deg2rad(offset_edges_deg)
    sin_edges = np.sin(offset_edges_rad)
    dOmega = 2.0 * np.pi * np.diff(sin_edges)  # (nOffset,)

    # -------------------------------------------------------------------------
    # Data: differential rate [s^-1 MeV^-1 sr^-1]
    # -------------------------------------------------------------------------
    data_obj = bkg2.data

    # Convert to plain ndarray with correct units if possible
    try:
        mu_rate_diff = data_obj.to_value("s-1 MeV-1 sr-1")
    except AttributeError:
        mu_rate_diff = np.asarray(data_obj, dtype=float)

    mu_rate_diff = np.nan_to_num(mu_rate_diff, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure shape is (nE, nOffset)
    if mu_rate_diff.shape == (nOffset, nE):
        # Many IRFs are stored as (nOffset, nE); transpose if needed
        mu_rate_diff = mu_rate_diff.T

    if mu_rate_diff.shape != (nE, nOffset):
        raise ValueError(
            f"Unexpected mu_rate_diff shape {mu_rate_diff.shape}, "
            f"expected (nE={nE}, nOffset={nOffset}) or (nOffset, nE)."
        )

    # If everything is exactly zero, fail fast
    if not np.any(mu_rate_diff):
        raise ValueError(
            "Muon differential background is all zeros after reading. "
            "This usually means the FITS was written with zero data or wrong units."
        )

    # -------------------------------------------------------------------------
    # From differential to per-bin rate: [s^-1]
    # -------------------------------------------------------------------------
    dE_GeV = np.diff(E_edges_GeV)
    dE_MeV = dE_GeV * 1e3

    mu_rate_bin = mu_rate_diff * dE_MeV[:, None] * dOmega[None, :]  # (nE, nOffset)

    mu_horizon_raw = mu_rate_bin[:, horizon_index]

    # -------------------------------------------------------------------------
    # Optional smoothing in log10(E)
    # -------------------------------------------------------------------------
    mu_horizon_smooth = None
    if smooth_sigma_decades is not None:
        if smooth_in_logE_func is None:
            raise ValueError(
                "smooth_in_logE_func must be provided if smoothing is requested."
            )
        mu_horizon_smooth = smooth_in_logE_func(
            mu_horizon_raw,
            E_edges_GeV,
            sigma_decades=smooth_sigma_decades,
        )

    # Energy bin centers in TeV for plotting
    E_centers_TeV = np.sqrt(E_edges_GeV[:-1] * E_edges_GeV[1:]) / 1e3

    return (
        E_edges_GeV,
        E_centers_TeV,
        mu_horizon_raw,
        mu_horizon_smooth,
        offset_edges_deg,
        horizon_index,
    )
def to_year(rate_s, t_year_s=None):
    if t_year_s is None:
        t_year_s = seconds_per_year()
    return np.asarray(rate_s) * t_year_s


# -----------------------------
# Plot
# -----------------------------
def plot_background_comparison(
    E_TeV,
    conv_year,
    prompt_year,
    mu_raw_year=None,
    mu_smooth_year=None,
    title=None,
    figsize=(3.5, 2.75),
    dpi=300,
):
    import matplotlib.pyplot as plt
    E_TeV = np.asarray(E_TeV)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.plot(E_TeV, conv_year, label="Conventional neutrinos")
    ax.plot(E_TeV, prompt_year, label="Prompt neutrinos", ls="--")

    if mu_raw_year is not None:
        ax.plot(E_TeV, mu_raw_year, label="Muons (raw)", drawstyle="steps-mid")
    if mu_smooth_year is not None:
        ax.plot(E_TeV, mu_smooth_year, label="Muons (smoothed)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$E_\mathrm{reco}$ [TeV]")
    ax.set_ylabel(r"Background rate [yr$^{-1}$]")
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()
    return fig, ax

def atmospheric_conv_flux(E_GeV, cos_theta, flavor="nu"):
    E_TeV = np.asarray(E_GeV, dtype=float) / 1e3

    # Normalisation: nu slightly larger than nubar
    norm = 1.0e-8 if flavor == "nu" else 0.8e-8

    # Broken power-law + soft knee
    gamma1 = 3.7
    gamma2 = 4.0
    E_knee = 100.0  # TeV

    flux = norm * E_TeV**(-gamma1) * (1.0 + (E_TeV / E_knee) ** (gamma2 - gamma1)) ** -1

    # More flux near horizon (toy): cos_theta=1 vertical, 0 horizon
    ct = np.clip(cos_theta, -1.0, 1.0)
    zenith_factor = np.clip(0.35 + 0.65 * (1.0 - np.abs(ct)), 0.3, 1.0)
    flux *= zenith_factor

    return flux


def atmospheric_prompt_flux(E_GeV, cos_theta, flavor="nu", E_cross_TeV=100.0):
    """
    Prompt flux with harder spectrum. Normalisation set so that, for vertical-ish
    directions, prompt crosses conventional around E_cross_TeV (order-of-magnitude).
    """
    E_TeV = np.asarray(E_GeV, dtype=float) / 1e3
    gamma = 2.7

    # Choose a reference angle for matching (roughly mid-zenith)
    ct_ref = 0.5

    # Compute conv at crossover energy to set prompt norm (per flavor)
    conv_at_cross = atmospheric_conv_flux(E_cross_TeV * 1e3, ct_ref, flavor=flavor)

    # For prompt: flux = norm * E^-gamma, so norm = flux(Ec) * Ec^gamma
    norm = conv_at_cross * (E_cross_TeV ** gamma)

    # Prompt is more isotropic: very weak zenith dependence (optional)
    ct = np.clip(cos_theta, -1.0, 1.0)
    ang = 0.95 + 0.05 * np.abs(ct)

    flux = norm * E_TeV**(-gamma) * ang
    return flux


from astropy.io import fits
import numpy as np
import pandas as pd

def build_aeff_from_table_no_zenith(
    aeff_table_path: str,
    file_name: str,
    theta_edges=None,           # rad
    zenith_reduce: str = "mean",
    upgoing_only: bool = False,
):
    df = pd.read_csv(aeff_table_path)

    # map columns (adatta se i nomi differiscono)
    col_map = {
        "log10(nu_E [GeV]) low": "log10E_lo",
        "log10(nu_E [GeV]) high": "log10E_hi",
        "cos(zen) center": "cosz_c",
        "aeff [m^2]": "aeff_m2",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    if upgoing_only and "cosz_c" in df.columns:
        df = df[df["cosz_c"] < 0].copy()

    # energy edges
    e_edges_log = np.unique(np.concatenate([df["log10E_lo"].to_numpy(), df["log10E_hi"].to_numpy()]))
    e_edges_log = np.sort(e_edges_log)
    e_edges = 10 ** e_edges_log  # GeV
    nE = len(e_edges) - 1

    # reduce zenith
    g = df.groupby(["log10E_lo", "log10E_hi"], sort=True)["aeff_m2"]
    if zenith_reduce == "mean":
        aeff_1d = g.mean().to_numpy()
    elif zenith_reduce == "median":
        aeff_1d = g.median().to_numpy()
    else:
        raise ValueError("zenith_reduce must be 'mean' or 'median'.")

    if aeff_1d.size != nE:
        raise ValueError(f"Reduced aeff size mismatch: got {aeff_1d.size}, expected {nE}")

    # theta/offset edges
    if theta_edges is None:
        theta_edges = np.linspace(0.0, np.pi, 13)  # 12 bins like WORKING
    theta_edges = np.asarray(theta_edges, dtype=float)
    nT = len(theta_edges) - 1

    # replicate same aeff across theta bins (zenith-independent)
    aeff = np.repeat(aeff_1d.reshape(nE, 1), nT, axis=1)  # (nE, nT)

    hdu0 = fits.PrimaryHDU()
    col_e_lo = fits.Column(name="ENERG_LO", format=f"{nE}E", unit="GeV", array=[e_edges[:-1]])
    col_e_hi = fits.Column(name="ENERG_HI", format=f"{nE}E", unit="GeV", array=[e_edges[1:]])
    col_t_lo = fits.Column(name="THETA_LO", format=f"{nT}E", unit="rad", array=[theta_edges[:-1]])
    col_t_hi = fits.Column(name="THETA_HI", format=f"{nT}E", unit="rad", array=[theta_edges[1:]])

    # IMPORTANT: store aeff.T and dim=(nE,nT) to match your WORKING convention
    col_data = fits.Column(
        name="EFFAREA",
        format=f"{nE*nT}D",
        dim=f"({nE},{nT})",
        unit="m2",
        array=[aeff.T],
    )

    hdu1 = fits.BinTableHDU.from_columns([col_e_lo, col_e_hi, col_t_lo, col_t_hi, col_data])
    hdu1.header["EXTNAME"]  = "EFFECTIVE AREA"
    hdu1.header["HDUDOC"]   = "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats"
    hdu1.header["HDUVERS"]  = "0.2"
    hdu1.header["HDUCLASS"] = "GADF"
    hdu1.header["HDUCLAS1"] = "RESPONSE"
    hdu1.header["HDUCLAS2"] = "EFF_AREA"
    hdu1.header["HDUCLAS3"] = "FULL-ENCLOSURE"
    hdu1.header["HDUCLAS4"] = "AEFF_2D"

    fits.HDUList([hdu0, hdu1]).writeto(file_name, overwrite=True)
    print("File written:", file_name)
    return aeff, e_edges, theta_edges

def build_edisp_from_csv(
    csv_path: str,
    file_name: str,
    max_off_deg: float = 180.0,
    n_off_bins: int = 6,
    migra_min: float | None = None,   # optional: override migra range (recommended!)
    migra_max: float | None = None,
    check_with_gammapy: bool = True,  # kept for API compatibility
):
    """
    Create a GADF EDISP_2D FITS from a CSV (true-logE, reco-logE) table.

    Output MATRIX is a PDF density in migra such that for each (E_true, offset):
        sum_j PDF(migra_j) * d(migra)_j = 1

    IMPORTANT: writes MATRIX to FITS as (offset, migra, energy)
    so that Gammapy internal transpose() yields (energy, migra, offset).

    Expected CSV columns:
      - "log10(nu_E [GeV]) low", "center", "high"
      - "log10(reco_E [GeV]) low", "center", "high"
      - "dP/dlog10(nu_E [GeV])"
    """
    import numpy as np
    import pandas as pd

    df = pd.read_csv(csv_path, skipinitialspace=True)

    # -----------------------
    # True-energy edges (log10 GeV) -> GeV
    # -----------------------
    e_log_lo = np.sort(df["log10(nu_E [GeV]) low"].unique())
    e_log_hi = np.sort(df["log10(nu_E [GeV]) high"].unique())
    e_true_log_edges = np.unique(np.concatenate([[e_log_lo.min()], np.sort(e_log_hi)]))
    if e_true_log_edges.size < 2:
        raise ValueError("Could not infer valid true-energy edges from CSV.")
    e_true_edges = 10 ** e_true_log_edges  # GeV
    e_true_centers = np.sqrt(e_true_edges[:-1] * e_true_edges[1:])
    n_e = e_true_centers.size

    # -----------------------
    # Reco-energy edges (log10 GeV) -> GeV (used only to set MIGRA range/bins count)
    # -----------------------
    r_log_lo = np.sort(df["log10(reco_E [GeV]) low"].unique())
    r_log_hi = np.sort(df["log10(reco_E [GeV]) high"].unique())
    reco_log_edges = np.unique(np.concatenate([[r_log_lo.min()], np.sort(r_log_hi)]))
    if reco_log_edges.size < 2:
        raise ValueError("Could not infer valid reco-energy edges from CSV.")
    reco_edges = 10 ** reco_log_edges  # GeV
    n_reco = reco_edges.size - 1
    if n_reco < 1:
        raise ValueError("Invalid reco-energy binning inferred from CSV.")

    # -----------------------
    # MIGRA axis
    # -----------------------
    auto_migra_min = float(reco_edges.min() / e_true_edges.max())
    auto_migra_max = float(reco_edges.max() / e_true_edges.min())

    if migra_min is None:
        migra_min = max(auto_migra_min, 1e-6)
    if migra_max is None:
        migra_max = max(auto_migra_max, migra_min * 1.01)

    migra_edges = np.logspace(np.log10(migra_min), np.log10(migra_max), n_reco + 1)
    migra_edges = np.unique(migra_edges)
    if migra_edges.size < 2:
        raise ValueError("MIGRA edges are not valid/unique.")
    n_m = migra_edges.size - 1
    dm = np.diff(migra_edges)

    # -----------------------
    # Offset axis (deg -> rad)
    # -----------------------
    if n_off_bins < 1:
        raise ValueError("n_off_bins must be >= 1.")
    if max_off_deg <= 0:
        raise ValueError("max_off_deg must be > 0.")
    off_edges_deg = np.linspace(0.0, float(max_off_deg), int(n_off_bins) + 1)
    theta_edges_rad = np.deg2rad(off_edges_deg).astype(float)
    n_t = int(n_off_bins)

    # -----------------------
    # Helper: distribute probability mass into global migra bins (in log-space)
    # -----------------------
    def add_mass_to_global_bins(m_lo, m_hi, mass, global_edges):
        if mass <= 0:
            return np.zeros(len(global_edges) - 1, dtype=float)
        lo = np.log10(max(m_lo, 1e-12))
        hi = np.log10(max(m_hi, 1e-12))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros(len(global_edges) - 1, dtype=float)

        g = np.log10(global_edges)
        out = np.zeros(len(global_edges) - 1, dtype=float)
        for j in range(len(out)):
            olo = max(lo, g[j])
            ohi = min(hi, g[j + 1])
            if ohi > olo:
                frac = (ohi - olo) / (hi - lo)
                out[j] += mass * frac
        return out

    # -----------------------
    # Build edisp for ONE offset bin: PDF density in migra
    # -----------------------
    edisp_one_pdf = np.zeros((n_e, n_m), dtype=float)

    for ie in range(n_e):
        e_c = float(e_true_centers[ie])
        e_center = np.log10(e_c)

        sel = np.isclose(df["log10(nu_E [GeV]) center"], e_center, rtol=0, atol=1e-6)
        sub = df.loc[sel].copy()
        if sub.empty:
            continue

        # density per log10(reco_E) -> convert to probability mass per reco bin
        dlog10 = (
            sub["log10(reco_E [GeV]) high"].values
            - sub["log10(reco_E [GeV]) low"].values
        ).astype(float)
        dens = sub["dP/dlog10(nu_E [GeV])"].values.astype(float)

        mass = dens * dlog10
        mass = np.where(np.isfinite(mass), mass, 0.0)

        tot = mass.sum()
        if tot > 0:
            mass /= tot

        # rebin to migra mass
        slice_mass = np.zeros(n_m, dtype=float)
        for m_i, rlo_log, rhi_log in zip(
            mass,
            sub["log10(reco_E [GeV]) low"].values,
            sub["log10(reco_E [GeV]) high"].values,
        ):
            if m_i <= 0:
                continue
            rlo = 10 ** float(rlo_log)
            rhi = 10 ** float(rhi_log)
            m_lo = rlo / e_c
            m_hi = rhi / e_c
            slice_mass += add_mass_to_global_bins(m_lo, m_hi, float(m_i), migra_edges)

        # normalize mass
        s = slice_mass.sum()
        if s > 0:
            slice_mass /= s

        # mass -> density
        pdf = np.zeros_like(slice_mass)
        good = dm > 0
        pdf[good] = slice_mass[good] / dm[good]

        # safety: enforce ∑ pdf*dm = 1
        norm = float((pdf * dm).sum())
        if norm > 0:
            pdf /= norm

        edisp_one_pdf[ie, :] = pdf

    # Replicate across offsets
    edisp_pdf = np.repeat(edisp_one_pdf[:, :, None], n_t, axis=2)  # (E, M, Off)

    _write_edisp_fits_autolayout(
        out_fits=file_name,
        e_true_edges=e_true_edges,
        migra_edges=migra_edges,
        theta_edges_rad=theta_edges_rad,
        edisp_pdf_EMT=edisp_pdf,   # (E,M,T)
        overwrite=True,
    )

    print(f"File written: {file_name}")

    return edisp_pdf, e_true_edges, migra_edges, theta_edges_rad

import numpy as np
from astropy.io import fits
from gammapy.irf import EnergyDispersion2D

def _write_edisp_fits_autolayout(
    out_fits: str,
    e_true_edges: np.ndarray,
    migra_edges: np.ndarray,
    theta_edges_rad: np.ndarray,
    edisp_pdf_EMT: np.ndarray,  # shape (nE, nM, nT) density in migra
    overwrite: bool = True,
):
    """
    Write EDISP_2D trying a few MATRIX layouts until Gammapy can read it.

    edisp_pdf_EMT must be (E, M, T) and already normalized so sum(pdf*dm)=1.
    """
    e_true_edges = np.asarray(e_true_edges, float).ravel()
    migra_edges = np.asarray(migra_edges, float).ravel()
    theta_edges_rad = np.asarray(theta_edges_rad, float).ravel()

    nE = len(e_true_edges) - 1
    nM = len(migra_edges) - 1
    nT = len(theta_edges_rad) - 1

    assert edisp_pdf_EMT.shape == (nE, nM, nT), (
        f"Expected (nE,nM,nT)={(nE,nM,nT)}, got {edisp_pdf_EMT.shape}"
    )

    # Common columns
    hdu0 = fits.PrimaryHDU()
    col_e_lo = fits.Column(
        name="ENERG_LO", format=f"{nE}E", unit="GeV", array=[e_true_edges[:-1]]
    )
    col_e_hi = fits.Column(
        name="ENERG_HI", format=f"{nE}E", unit="GeV", array=[e_true_edges[1:]]
    )
    col_m_lo = fits.Column(
        name="MIGRA_LO", format=f"{nM}E", unit="", array=[migra_edges[:-1]]
    )
    col_m_hi = fits.Column(
        name="MIGRA_HI", format=f"{nM}E", unit="", array=[migra_edges[1:]]
    )
    col_t_lo = fits.Column(
        name="THETA_LO", format=f"{nT}E", unit="rad", array=[theta_edges_rad[:-1]]
    )
    col_t_hi = fits.Column(
        name="THETA_HI", format=f"{nT}E", unit="rad", array=[theta_edges_rad[1:]]
    )

    # Candidate layouts:
    # Gammapy does: data = MATRIX[0].transpose()
    # It expects data.shape == (nE, nM, nT)
    # => MATRIX[0].shape must be (nT, nM, nE) or something compatible.
    candidates = []

    # A) store as (T, M, E) with dim matching that shape
    data_TME = edisp_pdf_EMT.transpose(2, 1, 0)  # (T,M,E)
    candidates.append(("TME_dim_TME", data_TME, f"({nT},{nM},{nE})"))

    # B) store as (T, M, E) but dim swapped (some FITS readers reshape differently)
    candidates.append(("TME_dim_EMT", data_TME, f"({nE},{nM},{nT})"))

    # C) legacy GADF-like: store as (M, E, T) with dim=(M,E,T)
    data_MET = edisp_pdf_EMT.transpose(1, 0, 2)  # (M,E,T)
    candidates.append(("MET_dim_MET", data_MET, f"({nM},{nE},{nT})"))

    last_err = None

    for tag, data3d, dim_str in candidates:
        col_mat = fits.Column(
            name="MATRIX",
            format=f"{nE * nM * nT}D",
            dim=dim_str,
            unit="",
            array=[data3d],
        )
        hdu1 = fits.BinTableHDU.from_columns(
            [col_e_lo, col_e_hi, col_m_lo, col_m_hi, col_t_lo, col_t_hi, col_mat]
        )
        hdu1.header.update(
            {
                "EXTNAME":  "ENERGY DISPERSION",
                "HDUDOC":   "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats",
                "HDUVERS":  "0.2",
                "HDUCLASS": "GADF",
                "HDUCLAS1": "RESPONSE",
                "HDUCLAS2": "EDISP",
                "HDUCLAS3": "FULL-ENCLOSURE",
                "HDUCLAS4": "EDISP_2D",
            }
        )

        # Build HDUList in memory
        hdul = fits.HDUList([hdu0, hdu1])

        try:
            # Let Gammapy interpret this layout directly from memory
            ed = EnergyDispersion2D.from_hdulist(hdul)
            arr = np.asarray(ed.data)
            if arr.shape != (nE, nM, nT):
                raise ValueError(
                    f"Gammapy read shape {arr.shape} != expected {(nE,nM,nT)}"
                )

            # success: write to out_fits once
            hdul.writeto(out_fits, overwrite=overwrite)
            print(f"[OK] wrote {out_fits} using layout {tag} with dim={dim_str}")

            # quick normalization check
            dm = np.diff(np.asarray(ed.axes["migra"].edges))
            integ = (arr * dm[None, :, None]).sum(axis=1)
            return

        except Exception as e:
            last_err = e
            # try next layout
            continue

    raise RuntimeError(
        f"Could not write an EDISP FITS readable by Gammapy. Last error: {last_err}"
    )

def build_psf_from_csv(
    psf_csv: str,
    file_name: str,
    max_off_deg: float = 180.0,
    n_off_bins: int = 6,
    clip_negative: bool = True,
    renormalize: bool = True,
):

    """
    Build a GADF PSF_2D_TABLE readable by gammapy.irf.PSF3D from a CSV.

    This version matches the convention that your WORKING file used:
    - rpsf array is built as (n_rad, n_theta, n_energy)
    - TDIM is declared as (n_energy, n_theta, n_rad)
    - Gammapy then does transpose() internally and obtains (n_energy, n_theta, n_rad)

    Expected CSV columns:
      - "log10(nu_E [GeV]) low"
      - "log10(nu_E [GeV]) high"
      - "log10(psi [degrees])"   (psi bin centers in log10 space)
      - "dP/dOmega"              (units: sr^-1)
    """
    import numpy as np
    import pandas as pd
    from astropy.io import fits

    df = pd.read_csv(psf_csv, skipinitialspace=True)

    required = [
        "log10(nu_E [GeV]) low",
        "log10(nu_E [GeV]) high",
        "log10(psi [degrees])",
        "dP/dOmega",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in CSV: {missing}")

    # -----------------------
    # ENERGY edges (true energy), in GeV
    # -----------------------
    elog_edges = np.unique(
        np.concatenate(
            [
                df["log10(nu_E [GeV]) low"].unique(),
                df["log10(nu_E [GeV]) high"].unique(),
            ]
        )
    )
    elog_edges = np.sort(elog_edges)
    if elog_edges.size < 2:
        raise ValueError("Need at least 2 unique energy edges.")
    e_edges = 10 ** elog_edges
    n_energy = e_edges.size - 1
    if n_energy <= 0:
        raise ValueError("Invalid energy edges.")

    e_low_vals = np.sort(df["log10(nu_E [GeV]) low"].unique())
    if e_low_vals.size != n_energy:
        print(
            f"Warning: found {e_low_vals.size} unique energy-low values, "
            f"but n_energy from edges is {n_energy}. Using min()."
        )
        n_energy = min(n_energy, e_low_vals.size)
        e_low_vals = e_low_vals[:n_energy]
        e_edges = e_edges[: n_energy + 1]

    # -----------------------
    # RAD edges (psi), in degrees
    # -----------------------
    logpsi_centers = np.unique(df["log10(psi [degrees])"].values)
    logpsi_centers = np.sort(logpsi_centers)
    if logpsi_centers.size < 2:
        raise ValueError("Need at least 2 psi centers to build edges.")

    mid = 0.5 * (logpsi_centers[1:] + logpsi_centers[:-1])
    first_edge = logpsi_centers[0] - (mid[0] - logpsi_centers[0])
    last_edge  = logpsi_centers[-1] + (logpsi_centers[-1] - mid[-1])
    logpsi_edges = np.concatenate([[first_edge], mid, [last_edge]])

    rad_edges_deg = 10 ** logpsi_edges
    n_rad = rad_edges_deg.size - 1
    if n_rad <= 0:
        raise ValueError("Invalid rad edges.")

    # -----------------------
    # OFFSET (THETA) edges
    # -----------------------
    if n_off_bins < 1:
        raise ValueError("n_off_bins must be >= 1.")
    if max_off_deg <= 0:
        raise ValueError("max_off_deg must be > 0.")

    off_edges_deg = np.linspace(0.0, float(max_off_deg), int(n_off_bins) + 1)
    theta_edges = np.deg2rad(off_edges_deg).astype(float)
    n_theta = int(n_off_bins)

    # -----------------------
    # Build rpsf: (n_rad, n_theta, n_energy)
    # Fill theta slice from CSV then replicate across theta bins
    # -----------------------
    rpsf_one = np.zeros((n_rad, n_energy), dtype=float)

    for ie, elog_lo in enumerate(e_low_vals):
        sub = df[df["log10(nu_E [GeV]) low"] == elog_lo].copy()
        sub = sub.sort_values("log10(psi [degrees])")
        vals = sub["dP/dOmega"].to_numpy(dtype=float)

        if vals.size != n_rad:
            raise ValueError(
                f"Energy bin log10E_low={elog_lo}: expected {n_rad} rad bins, got {vals.size}"
            )

        rpsf_one[:, ie] = vals

    # replicate across theta bins (toy: no offset dependence)
    rpsf = np.repeat(rpsf_one[:, None, :], n_theta, axis=1)  # (rad, theta, energy)

    # -----------------------
    # Sanity: clip negatives + renormalize per (E, theta) over dOmega
    # -----------------------
    if clip_negative:
        rpsf = np.clip(rpsf, 0.0, None)

    if renormalize:
        # dOmega per radial bin (deg -> rad)
        rad_lo = np.deg2rad(rad_edges_deg[:-1])
        rad_hi = np.deg2rad(rad_edges_deg[1:])
        dOmega = 2.0 * np.pi * (np.cos(rad_lo) - np.cos(rad_hi))  # (n_rad,)

        # Integral over dOmega for each (theta, energy):
        # rpsf is (rad, theta, energy)
        integ = (rpsf * dOmega[:, None, None]).sum(axis=0, keepdims=True)  # (1, theta, energy)

        good = integ > 0
        rpsf = np.where(good, rpsf / integ, 0.0)

    # -----------------------
    # Write FITS (GADF PSF_2D_TABLE)
    # IMPORTANT: dim = (n_energy, n_theta, n_rad) with array=[rpsf]
    # -----------------------
    hdu0 = fits.PrimaryHDU()

    col_e_lo = fits.Column(name="ENERG_LO", format=f"{n_energy}E", unit="GeV", array=[e_edges[:-1]])
    col_e_hi = fits.Column(name="ENERG_HI", format=f"{n_energy}E", unit="GeV", array=[e_edges[1:]])
    col_t_lo = fits.Column(name="THETA_LO", format=f"{n_theta}E", unit="rad", array=[theta_edges[:-1]])
    col_t_hi = fits.Column(name="THETA_HI", format=f"{n_theta}E", unit="rad", array=[theta_edges[1:]])
    col_r_lo = fits.Column(name="RAD_LO",   format=f"{n_rad}E",   unit="deg", array=[rad_edges_deg[:-1]])
    col_r_hi = fits.Column(name="RAD_HI",   format=f"{n_rad}E",   unit="deg", array=[rad_edges_deg[1:]])

    col_rpsf = fits.Column(
        name="RPSF",
        format=f"{n_rad * n_theta * n_energy}D",
        dim=f"({n_energy},{n_theta},{n_rad})",
        unit="sr-1",
        array=[rpsf],
    )

    hdu1 = fits.BinTableHDU.from_columns([col_e_lo, col_e_hi, col_t_lo, col_t_hi, col_r_lo, col_r_hi, col_rpsf])

    hdu1.header["EXTNAME"]  = "PSF_2D_TABLE"
    hdu1.header["HDUDOC"]   = "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats"
    hdu1.header["HDUVERS"]  = "0.2"
    hdu1.header["HDUCLASS"] = "GADF"
    hdu1.header["HDUCLAS1"] = "RESPONSE"
    hdu1.header["HDUCLAS2"] = "RPSF"
    hdu1.header["HDUCLAS3"] = "FULL-ENCLOSURE"
    hdu1.header["HDUCLAS4"] = "PSF_TABLE"

    fits.HDUList([hdu0, hdu1]).writeto(file_name, overwrite=True)
    print(f"File written: {file_name}")

    return rpsf, e_edges, rad_edges_deg, theta_edges


def build_background2d_from_csv_v2(
    bkg_csv,
    out_fits="bkg_muons.fits",
):
    """
    Build a GADF BACKGROUND (BKG_2D) for atmospheric muons from a CSV containing
    per-bin rates [s^-1].

    Expected CSV columns:
      - log10(reco_E [GeV]) low / high
      - sin(dec) low / high
      - rate [s^-1] (events per (E, sin(dec)) bin per second)

    The output is written as a Gammapy Background2D with:
      - "energy" axis in GeV
      - "offset" axis in degrees, representing declination band edges derived from sin(dec)

    The final BKG units are: [s^-1 MeV^-1 sr^-1].
    """
    import numpy as np
    import pandas as pd
    import astropy.units as u
    from gammapy.irf import Background2D
    from gammapy.maps import MapAxis, MapAxes

    # --- read CSV
    df = pd.read_csv(bkg_csv, skipinitialspace=True)

    # -------------------------------------------------------------------------
    # Energy edges (GeV)
    # -------------------------------------------------------------------------
    e_edges_log = np.unique(
        np.concatenate([
            df["log10(reco_E [GeV]) low"].to_numpy(),
            df["log10(reco_E [GeV]) high"].to_numpy(),
        ])
    )
    e_edges_log = np.sort(e_edges_log)
    if e_edges_log.size < 2:
        raise ValueError("Invalid energy binning inferred from CSV.")

    e_edges = 10 ** e_edges_log
    nE = len(e_edges) - 1

    # -------------------------------------------------------------------------
    # sin(dec) edges → dec edges in degrees
    # -------------------------------------------------------------------------
    s_edges = np.unique(
        np.concatenate([
            df["sin(dec) low"].to_numpy(),
            df["sin(dec) high"].to_numpy(),
        ])
    )
    s_edges = np.sort(s_edges)
    if s_edges.size < 2:
        raise ValueError("Invalid sin(dec) binning inferred from CSV.")

    # Convert sin(dec) → dec in radians → degrees
    dec_edges_rad = np.arcsin(np.clip(s_edges, -1.0, 1.0))
    dec_edges_deg = np.degrees(dec_edges_rad)
    nD = len(dec_edges_deg) - 1

    # -------------------------------------------------------------------------
    # Fill per-bin rates (s^-1) in a 2D grid (E, dec)
    # The binning is done using the "low" edges from the CSV.
    # -------------------------------------------------------------------------
    e_lo = 10 ** df["log10(reco_E [GeV]) low"].to_numpy()
    s_lo = df["sin(dec) low"].to_numpy()
    rates = df["rate [s^-1]"].to_numpy(dtype=float)

    e_idx = np.searchsorted(e_edges, e_lo, side="right") - 1
    d_idx = np.searchsorted(s_edges, s_lo, side="right") - 1

    rate_bin = np.zeros((nE, nD), dtype=float)
    for i in range(len(df)):
        ie = int(e_idx[i])
        idc = int(d_idx[i])
        if 0 <= ie < nE and 0 <= idc < nD:
            rate_bin[ie, idc] += rates[i]

    # -------------------------------------------------------------------------
    # Convert raw bin rates → differential rate [s^-1 MeV^-1 sr^-1]
    # -------------------------------------------------------------------------

    # Energy bin width in MeV
    dE_GeV = np.diff(e_edges)
    dE_MeV = dE_GeV * 1e3

    # Solid angle for sin(dec) bands:
    # dΩ = 2π (sin_dec_hi - sin_dec_lo)
    s_hi = s_edges[1:]
    s_lo = s_edges[:-1]
    dOmega = 2.0 * np.pi * (s_hi - s_lo)

    # Differential rate
    rate_diff = rate_bin / dE_MeV[:, None] / dOmega[None, :]
    rate_diff = np.nan_to_num(rate_diff, nan=0.0, posinf=0.0, neginf=0.0)

    # -------------------------------------------------------------------------
    # Build a Background2D object (Gammapy ensures correct GADF formatting)
    # -------------------------------------------------------------------------
    energy_axis = MapAxis.from_edges(e_edges, unit="GeV", name="energy", interp="log")
    offset_axis = MapAxis.from_edges(dec_edges_deg, unit="deg", name="offset", interp="lin")
    axes = MapAxes([energy_axis, offset_axis])

    bkg = Background2D(
        axes=axes,
        data=rate_diff,
        unit=u.Unit("s-1 MeV-1 sr-1"),
        meta={
            "BKGTYPE": "RATE_PER_BIN_FROM_CSV",
            "COMMENT": "Offset axis stores declination edges [deg] derived from sin(dec).",
        },
    )

    # Write FITS in GADF-DL3 format
    hdu = bkg.to_table_hdu(format="gadf-dl3")
    hdu.name = "BACKGROUND"
    hdu.writeto(out_fits, overwrite=True)

    print(f"[OK] wrote muon background: {out_fits}")
    return bkg, e_edges, dec_edges_deg

def theta_geometry_from_edges(theta_edges, theta_unit="rad", theta_cut_deg=80.0):
    """
    theta_edges: array of edges
      - if theta_unit="rad": edges in rad
      - if theta_unit="deg": edges in deg
    returns: centers(rad), mask, dOmega (sr)
    """
    th = np.asarray(theta_edges, dtype=float).ravel()
    if theta_unit == "deg":
        th_rad = np.deg2rad(th)
        th_cent_deg = 0.5 * (th[:-1] + th[1:])
    else:
        th_rad = th
        th_cent_deg = np.rad2deg(0.5 * (th[:-1] + th[1:]))

    mask = th_cent_deg > float(theta_cut_deg)
    dOmega = 2.0 * np.pi * (np.cos(th_rad[:-1]) - np.cos(th_rad[1:]))
    th_cent_rad = 0.5 * (th_rad[:-1] + th_rad[1:])
    return th_cent_rad, mask, dOmega

def build_background2d_from_neutrinos_v2(
    atm_conv_rate,            # (nE, nT)  [s^-1 GeV^-1 sr^-1]
    atm_prompt_rate,          # (nE, nT)  [s^-1 GeV^-1 sr^-1]
    e_edges_GeV,              # (nE+1,)
    theta_edges_rad=None,     # (nT+1,) in rad; default -> 0..pi
    n_theta_bins=6,
    out_fits="bkg_nu_total.fits",
    split="total",            # "total"|"conv"|"prompt"
    overwrite=True,
):
    import numpy as np
    from astropy.io import fits

    e_edges_GeV = np.asarray(e_edges_GeV, dtype=float).ravel()
    nE = e_edges_GeV.size - 1
    if nE <= 0:
        raise ValueError("Invalid energy edges.")

    conv = np.asarray(atm_conv_rate, dtype=float)
    prompt = np.asarray(atm_prompt_rate, dtype=float)

    if conv.shape[0] != nE or prompt.shape[0] != nE:
        raise ValueError("Energy dimension mismatch between rates and e_edges_GeV.")

    if split in (None, "total"):
        rate_GeV = conv + prompt
    elif split == "conv":
        rate_GeV = conv
    elif split == "prompt":
        rate_GeV = prompt
    else:
        raise ValueError("split must be 'total', 'conv', or 'prompt'.")

    if theta_edges_rad is None:
        theta_edges_rad = np.linspace(0.0, np.pi, int(n_theta_bins) + 1)
    theta_edges_rad = np.asarray(theta_edges_rad, dtype=float).ravel()
    nT = theta_edges_rad.size - 1
    if nT <= 0:
        raise ValueError("Invalid theta edges.")

    # Ensure shape (nE, nT)
    if rate_GeV.ndim == 1:
        rate_GeV = rate_GeV[:, None]
    if rate_GeV.shape[1] == 1 and nT > 1:
        rate_GeV = np.repeat(rate_GeV, nT, axis=1)
    if rate_GeV.shape[1] != nT:
        raise ValueError(f"Rate theta bins mismatch: rate has {rate_GeV.shape[1]} but theta_edges imply {nT}.")

    # Convert GeV^-1 -> MeV^-1
    rate_MeV = rate_GeV * 1e-3

    # Write FITS (same convention as your working writer)
    hdu0 = fits.PrimaryHDU()
    col1 = fits.Column(name="ENERG_LO", format=f"{nE}E", unit="GeV", array=[e_edges_GeV[:-1]])
    col2 = fits.Column(name="ENERG_HI", format=f"{nE}E", unit="GeV", array=[e_edges_GeV[1:]])
    col3 = fits.Column(name="THETA_LO", format=f"{nT}E", unit="rad", array=[theta_edges_rad[:-1]])
    col4 = fits.Column(name="THETA_HI", format=f"{nT}E", unit="rad", array=[theta_edges_rad[1:]])
    col5 = fits.Column(
        name="BKG",
        format=f"{rate_MeV.size}D",
        dim=f"({nT},{nE})",
        unit="MeV-1 s-1 sr-1",
        array=[rate_MeV],
    )

    hdu1 = fits.BinTableHDU.from_columns([col1, col2, col3, col4, col5])
    hdu1.header.update({
        "EXTNAME": "BACKGROUND",
        "HDUDOC": "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats",
        "HDUVERS": "0.2",
        "HDUCLASS": "GADF",
        "HDUCLAS1": "RESPONSE",
        "HDUCLAS2": "BKG",
        "HDUCLAS3": "FULL-ENCLOSURE",
        "HDUCLAS4": "BKG_2D",
        "BKGTYPE": f"NU_{split.upper()}",
        "COMMENT": "THETA axis is zenith/offset (rad).",
    })

    fits.HDUList([hdu0, hdu1]).writeto(out_fits, overwrite=overwrite)
    print(f"[OK] wrote: {out_fits}")

    return e_edges_GeV, theta_edges_rad, rate_MeV



def build_atmospheric_background_from_irfs(
    aeff_irf,             # gammapy EffectiveAreaTable2D
    edisp_q_all,          # ndarray or (ndarray, ...)
    e_true_edges_coarse,  # GeV edges
    migra_edges,          # migra edges (dimensionless)
    theta_edges_coarse,   # rad edges
    e_reco_edges=None,    # optional GeV edges; if None -> uses e_true_edges_coarse
):
    """
    Build a toy atmospheric background (conventional + prompt) from AEFF + EDISP.

    Output rates are differential in reconstructed energy:
      atm_*_rate[Ereco_bin, theta_bin]  ~  (arbitrary units) / GeV / sr

    Notes
    -----
    - Robust to EDISP being either probability per migra-bin OR a density in migra.
    - Folding is done explicitly using migra = E_reco / E_true.
    """
    # -----------------------
    # 0) sanitize axes
    # -----------------------
    e_true_edges = np.asarray(e_true_edges_coarse, dtype=float).ravel()
    migra_edges  = np.asarray(migra_edges, dtype=float).ravel()
    th_edges     = np.asarray(theta_edges_coarse, dtype=float).ravel()

    if e_true_edges.size < 2 or migra_edges.size < 2 or th_edges.size < 2:
        raise ValueError("Need at least 2 edges for each axis.")

    e_true_centers = np.sqrt(e_true_edges[:-1] * e_true_edges[1:])
    th_centers     = 0.5 * (th_edges[:-1] + th_edges[1:])
    nE = e_true_centers.size
    nT = th_centers.size
    nM = migra_edges.size - 1

    if e_reco_edges is None:
        e_reco_edges = e_true_edges.copy()
    else:
        e_reco_edges = np.asarray(e_reco_edges, dtype=float).ravel()

    e_reco_centers = np.sqrt(e_reco_edges[:-1] * e_reco_edges[1:])
    nR = e_reco_edges.size - 1

    dE_true = np.diff(e_true_edges)          # GeV
    dE_reco = np.diff(e_reco_edges)          # GeV
    dM      = np.diff(migra_edges)           # dimensionless

    # -----------------------
    # 1) get AEFF array in m^2 and map it onto (Etrue, theta_coarse)
    #    (simple interpolation, but NO zeroing outside range)
    # -----------------------
    data_obj = aeff_irf.data
    if hasattr(data_obj, "quantity"):
        aeff = np.asarray(data_obj.quantity.to_value("m2"))
    elif hasattr(data_obj, "to_value"):
        aeff = np.asarray(data_obj.to_value("m2"))
    else:
        aeff = np.asarray(data_obj, dtype=float)

    aeff = np.squeeze(aeff)
    if aeff.ndim == 1:
        aeff = aeff[:, None]
    if aeff.ndim != 2:
        raise ValueError(f"Unsupported AEFF shape: {aeff.shape}")

    axes = aeff_irf.axes
    # energy_true
    if "energy_true" in axes.names:
        e_aeff = axes["energy_true"].center.to_value("GeV")
    else:
        e_aeff = axes[0].center.to_value("GeV")

    # theta/offset
    if "offset" in axes.names:
        th_aeff = np.deg2rad(axes["offset"].center.to_value("deg"))
    else:
        th_aeff = (axes[1].center * (axes[1].unit or 1.0)).to_value("rad")

    if aeff.shape[0] != e_aeff.size:
        raise ValueError(f"AEFF energy axis mismatch: {aeff.shape[0]} vs {e_aeff.size}")

    # interpolate in logE
    logE_src = np.log10(e_aeff)
    logE_tgt = np.log10(e_true_centers)

    aeff_E = np.zeros((nE, aeff.shape[1]), dtype=float)
    for j in range(aeff.shape[1]):
        aeff_E[:, j] = np.interp(logE_tgt, logE_src, aeff[:, j],
                                 left=aeff[0, j], right=aeff[-1, j])

    # interpolate in theta (or replicate if only 1 theta bin in aeff)
    if aeff_E.shape[1] == 1:
        aeff_coarse = np.repeat(aeff_E, nT, axis=1)
    else:
        aeff_coarse = np.zeros((nE, nT), dtype=float)
        for i in range(nE):
            aeff_coarse[i] = np.interp(th_centers, th_aeff, aeff_E[i],
                                       left=aeff_E[i, 0], right=aeff_E[i, -1])

    # -----------------------
    # 2) extract/reshape EDISP -> (theta, migra, Etrue)
    # -----------------------
    if isinstance(edisp_q_all, (tuple, list)):
        edisp = np.asarray(edisp_q_all[0], dtype=float)
    else:
        edisp = np.asarray(edisp_q_all, dtype=float)

    if edisp.ndim != 3:
        raise ValueError(f"EDISP must be 3D, got {edisp.shape}")

    sh = edisp.shape
    if sh == (nT, nM, nE):
        edisp_tme = edisp
    elif sh == (nE, nM, nT):
        edisp_tme = np.transpose(edisp, (2, 1, 0))
    elif sh == (nM, nE, nT):
        edisp_tme = np.transpose(edisp, (2, 0, 1))
    else:
        raise ValueError(
            f"EDISP shape {sh} not compatible with expected nT={nT}, nM={nM}, nE={nE}"
        )

    # -----------------------
    # 3) normalize EDISP correctly
    #    - if it's already "probability per migra-bin": sum_m edisp ~= 1
    #    - if it's "density in migra": sum_m edisp * dM ~= 1  -> convert to per-bin prob
    # -----------------------
    sum_plain = edisp_tme.sum(axis=1)              # (theta, E)
    sum_w     = (edisp_tme * dM[None, :, None]).sum(axis=1)

    # decide which looks like "1"
    plain_ok = np.nanmedian(sum_plain) > 0.2 and np.nanmedian(sum_plain) < 5.0
    w_ok     = np.nanmedian(sum_w)     > 0.2 and np.nanmedian(sum_w)     < 5.0

    if w_ok and not plain_ok:
        # edisp is a density -> convert to per-bin probability
        edisp_prob = edisp_tme * dM[None, :, None]
    else:
        # assume already per-bin probability
        edisp_prob = edisp_tme.copy()

    # final renormalization so that sum_m P(migra|E,theta)=1 (avoid numerical drift)
    norm = edisp_prob.sum(axis=1, keepdims=True)  # (theta,1,E)
    norm = np.where(norm <= 0, 1.0, norm)
    edisp_prob /= norm

    # -----------------------
    # 4) true-energy differential rates (per GeV per sr) on (Etrue, theta)
    # -----------------------
    atm_conv_true   = np.zeros((nE, nT), dtype=float)
    atm_prompt_true = np.zeros((nE, nT), dtype=float)

    for j, th in enumerate(th_centers):
        ct = np.cos(th)
        conv = atmospheric_conv_flux(e_true_centers, ct)
        prom = atmospheric_prompt_flux(e_true_centers, ct)

        # include nu + nubar simply as factor 2 (toy)
        atm_conv_true[:, j]   = 2.0 * conv * aeff_coarse[:, j]
        atm_prompt_true[:, j] = 2.0 * prom * aeff_coarse[:, j]

    # -----------------------
    # 5) fold Etrue -> Ereco using migra bins
    #    migra bin m corresponds to Ereco in [m_lo*Etrue, m_hi*Etrue]
    #    distribute probability by overlap with Ereco bins (uniform within migra bin)
    # -----------------------
    atm_conv_rate   = np.zeros((nR, nT), dtype=float)
    atm_prompt_rate = np.zeros_like(atm_conv_rate)

    for j in range(nT):
        for k in range(nE):
            # events in true-energy bin k (per sr)
            ev_conv_true   = atm_conv_true[k, j]   * dE_true[k]
            ev_prompt_true = atm_prompt_true[k, j] * dE_true[k]

            Etrue = e_true_centers[k]

            # for each migra bin, map to an Ereco interval
            for m in range(nM):
                p = edisp_prob[j, m, k]
                if p <= 0:
                    continue

                Ereco_lo = migra_edges[m]     * Etrue
                Ereco_hi = migra_edges[m + 1] * Etrue
                if Ereco_hi <= e_reco_edges[0] or Ereco_lo >= e_reco_edges[-1]:
                    continue

                # find overlapping reco bins
                i0 = np.searchsorted(e_reco_edges, Ereco_lo, side="right") - 1
                i1 = np.searchsorted(e_reco_edges, Ereco_hi, side="left")

                i0 = max(i0, 0)
                i1 = min(i1, nR)

                width = Ereco_hi - Ereco_lo
                if width <= 0:
                    continue

                for i in range(i0, i1):
                    lo = max(Ereco_lo, e_reco_edges[i])
                    hi = min(Ereco_hi, e_reco_edges[i + 1])
                    if hi > lo:
                        frac = (hi - lo) / width  # uniform within migra bin
                        atm_conv_rate[i, j]   += ev_conv_true   * p * frac
                        atm_prompt_rate[i, j] += ev_prompt_true * p * frac

    # convert events/bin back to differential rate per GeV
    atm_conv_rate   /= dE_reco[:, None]
    atm_prompt_rate /= dE_reco[:, None]

    return atm_conv_rate, atm_prompt_rate, e_reco_edges, th_edges, "rad"

def smooth_in_logE(y, e_centers_GeV, sigma_decades=0.15, eps=1e-40):
    """
    Smooth y(logE) assuming e_centers are logarithmically spaced or near-log spaced.
    sigma_decades = width in log10(E) (e.g. 0.10–0.25 decades).
    """
    y = np.asarray(y, float)
    e = np.asarray(e_centers_GeV, float)

    logE = np.log10(e)
    dlogE = np.median(np.diff(logE))
    if dlogE <= 0:
        raise ValueError("Energy centers must be strictly increasing")

    sigma_bins = sigma_decades / dlogE

    # smooth in log-space of y to preserve power-laws
    y_clip = np.clip(y, eps, None)
    logy = np.log10(y_clip)
    logy_sm = gaussian_filter1d(logy, sigma=sigma_bins, mode="nearest")
    return 10**logy_sm
