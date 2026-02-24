"""Helper utilities for the KM3NeT hadronic-fraction pseudo-experiment notebook.

This module is intentionally verbose and heavily documented so it can be used in a
hands-on session. Functions are small and composable.

Assumptions:
- Relative paths are the same as in the original notebook.
- Gammapy/Naima/KM3NeT custom utilities (PriorDatasets, Fit_wp, PionDecayKelner06) are installed.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import os
import warnings
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord

import yaml
from naima.radiative import BaseProton
from gammapy.modeling import Covariance, Fit
from gammapy.datasets import MapDataset, Datasets

from gammapy.modeling.models import DiskSpatialModel, SkyModel, NaimaSpectralModel
from naima.models import ExponentialCutoffPowerLaw

from astropy.constants import c
from naima.utils import trapz_loglog

from naima.extern.validator import (
    validate_array,
    validate_physical_type,
    validate_scalar,
)

def _validate_ene(ene):
    from astropy.table import Table

    if isinstance(ene, dict) or isinstance(ene, Table):
        try:
            ene = validate_array(
                "energy", u.Quantity(ene["energy"]), physical_type="energy"
            )
        except KeyError:
            raise TypeError("Table or dict does not have 'energy' column")
    else:
        if not isinstance(ene, u.Quantity):
            ene = u.Quantity(ene)
        validate_physical_type("energy", ene, physical_type="energy")

    return ene

class PionDecayKelner06(BaseProton):
    def __init__(
        self,
        particle_distribution,
        nh=1.0 / u.cm ** 3,
        particle_type='muon_neutrino',
        oscillation_factor=1,
        **kwargs
    ):
        super().__init__(particle_distribution)
        self.nh = validate_scalar("nh", nh, physical_type="number density")
        self.Epmin = 100 * u.GeV  # Model only valid for energies down to 100 GeV
        self.Epmax = 100 * u.PeV  # Model only valid for energies up to 100 PeV
        self.nEpd = 100
        self.pt=particle_type
        self.of=oscillation_factor
        self.param_names += ["nh"]
        self.__dict__.update(**kwargs)

    def _sigma_inel(self, Tp):
        L = np.log((Tp.to('TeV').value))
        sigma = 34.3 + 1.88 * L + 0.25 * L ** 2
        return sigma * 1e-27  # convert from mbarn to cm-2

    def _Fnu1(self, Ep, x):
        y=x/0.427
        L= np.log(Ep.to('TeV').value)
        B_=1.75 + 0.204*L + 0.010*L**2
        beta_=1/(1.67 + 0.111*L + 0.0038*L**2)
        k_=1.07 - 0.086*L + 0.002*L**2
        yb=y**beta_
        F1=B_ * np.log(y)/y * ( (1-yb)/(1+ k_*yb *(1-yb)) )**4 * \
        (1/np.log(y)-( (4*beta_*yb) / (1-yb) ) - (4*k_*beta_*yb*(1-2*yb))/(1+k_*yb*(1-yb)))
        F1[np.where(x>0.427)]=0
        return F1

    def _Fnu2(self, Ep, x):
        L= np.log(Ep.to('TeV').value)
        B=1/(69.5 + 2.65*L + 0.3*L**2)
        beta=(0.201 + 0.062*L + 0.00042*L**2)**(-0.25)
        k=(0.279 + 0.141*L + 0.0172*L**2)/(0.3 + (2.3+L)**2)
        F2= B * (1+ k*np.log(x)**2)**3 /(x*(1+0.3/x**beta)) *(-np.log(x))**5
        F2[x>=1]=0
        return F2

    def _Fgamma(self, x, Ep):
        L = np.log(Ep.to('TeV').value)
        B = 1.30 + 0.14 * L + 0.011 * L ** 2
        beta = (1.79 + 0.11 * L + 0.008 * L ** 2) ** -1
        k = (0.801 + 0.049 * L + 0.014 * L ** 2) ** -1
        xb = x ** beta

        F1 = B * (np.log(x) / x) * ((1 - xb) / (1 + k * xb * (1 - xb))) ** 4
        F2 = (
            1.0 / np.log(x)
            - (4 * beta * xb) / (1 - xb)
            - (4 * k * beta * xb * (1 - 2 * xb)) / (1 + k * xb * (1 - xb))
        )

        F= F1 * F2
        F[x>=1]=0
        return np.nan_to_num(F)

    def _particle_distribution(self, E):
        return self.particle_distribution(E * u.TeV).to("1/TeV").value

    def _spectrum(self, neutrino_energy):
        Enu = _validate_ene(neutrino_energy).to("GeV")
        if Enu.min()<50*u.GeV:
            warnings.warn("Model only valid for energies > 100 GeV")
        Ep = self._Ep * u.GeV
        J = self._J * u.Unit("1/GeV")
        sigma_inel = self._sigma_inel(Ep) * u.cm**2

        if self.pt == 'muon_neutrino':
            mg = np.meshgrid(Enu,Ep)
            x=(mg[0]/Ep[:,None]).value
            Ep_=mg[1]
            F = self._Fnu1(Ep_,x) + self._Fnu2(Ep_,x)
            specpp = trapz_loglog(F*J[:,None]*sigma_inel[:,None]/Ep_, Ep, axis=0)*self.of
        elif self.pt == 'gamma':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mg = np.meshgrid(Enu,Ep)
                x=(mg[0]/Ep[:,None]).value
                Ep_=mg[1]
                F = self._Fgamma(x,Ep_)
                specpp = trapz_loglog(F*J[:,None]*sigma_inel[:,None]/Ep_, Ep, axis=0)
        elif self.pt == 'electron':
            mg = np.meshgrid(Enu,Ep)
            x=(mg[0]/Ep[:,None]).value
            Ep_=mg[1]
            F = self._Fnu2(Ep_,x)
            specpp = trapz_loglog(F*J[:,None]*sigma_inel[:,None]/Ep_, Ep, axis=0)*self.of
        else:
            raise KeyError('Only "muon_neutrino", "gamma" and "electron" are supported particle_types')

        self.specpp = u.Quantity(specpp)
        self.specpp *= self.nh * c.cgs
        return self.specpp.to("1/(s eV)")

@contextmanager
def pushd(path: str | Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield Path.cwd()
    finally:
        os.chdir(old)

@dataclass(frozen=True)
class SourceInfo:
    name: str
    position: SkyCoord
    radius_deg: float
    distance_kpc: float

@dataclass(frozen=True)
class PDParams:
    amplitude_1_over_eV: float
    e0_TeV: float
    alpha: float
    ecut_TeV: float
    beta: float

class Fit_wp(Fit):
    def __init__(self, datasets, store_trace=False):
        super().__init__(store_trace=store_trace)
        self.datasets = datasets

class PriorDatasets(Datasets):
    def __init__(self,datasets=None, nuisance=None, **kwargs):
        super().__init__(datasets, **kwargs)
        self._nuisance = nuisance

    @property
    def nuisance(self):
        return self._nuisance

    @nuisance.setter
    def nuisance(self, nuisance):
        self._nuisance = nuisance

    def stat_sum(self):
        wstat = super().stat_sum()
        liketotal = wstat
        if self.nuisance:
            liketotal += self.nuisance_alpha() + self.nuisance_beta() + self.nuisance_ecut() + self.nuisance_amp()
        return liketotal

    def stat_sum_no_prior(self):
        return super().stat_sum()

    def nuisance_alpha(self):
        alpha = self.models['nu_PD'].parameters['alpha'].value
        alpha0 = self.nuisance['alpha']
        error = self.nuisance['alpha_err']
        return (alpha-alpha0)**2 / error**2

    def nuisance_beta(self):
        beta = self.models['nu_PD'].parameters['beta'].value
        beta0 = self.nuisance['beta']
        error = self.nuisance['beta_err']
        return (beta-beta0)**2 / error**2

    def nuisance_ecut(self):
        unit = self.nuisance['ecut_unit']
        ecut = self.models['nu_PD'].parameters['e_cutoff'].quantity.to_value(unit)
        ecut0 = self.nuisance['ecut']
        error = self.nuisance['ecut_err']
        return (ecut-ecut0)**2 / error**2

    def nuisance_amp(self):
        unit = self.nuisance['amp_unit']
        amp = self.models['nu_PD'].parameters['amplitude'].quantity.to_value(unit)
        amp0 = self.nuisance['amp']
        error = self.nuisance['amp_err']
        return (amp-amp0)**2 / error**2

def load_source_list(source_name: str, source_list_path: str | Path = "../models/source_list.txt") -> SourceInfo:
    df = pd.read_csv(source_list_path, delim_whitespace=True, index_col=0, comment="#")
    row = df.loc[source_name]
    pos = SkyCoord(row["RA"], row["Dec"], unit="deg", frame="icrs")
    return SourceInfo(
        name=source_name,
        position=pos,
        radius_deg=float(row["Radius"]),
        distance_kpc=float(row["Dist"]),
    )

def load_pd_params_from_yaml(model_yaml: str | Path = "../models/model.yaml") -> PDParams:
    with Path(model_yaml).open("r") as f:
        y = yaml.safe_load(f)

    amplitude = float(y["amplitude"]) / u.eV
    e0 = float(y["e_0"]) * u.TeV
    alpha = float(y["alpha"])
    ecut = float(y["e_cutoff"]) * u.TeV
    beta = float(y["beta"])

    return PDParams(
        amplitude_1_over_eV=amplitude.to(1 / u.eV).value,
        e0_TeV=e0.to(u.TeV).value,
        alpha=alpha,
        ecut_TeV=ecut.to(u.TeV).value,
        beta=beta,
    )

def load_km3net_datasets(dataset_dir: str | Path, datasets_yaml: str = "datasets.yaml"):
    with pushd(dataset_dir):
        return PriorDatasets.read(filename=datasets_yaml)

def build_hadronic_pd_model(source: SourceInfo, pars: PDParams, datasets_names):
    spatial_model = DiskSpatialModel(
        lon_0=source.position.ra,
        lat_0=source.position.dec,
        r_0=source.radius_deg * u.deg,
        frame="icrs",
    )

    ecpl = ExponentialCutoffPowerLaw(
        amplitude=pars.amplitude_1_over_eV / u.eV,
        e_0=pars.e0_TeV * u.TeV,
        alpha=pars.alpha,
        e_cutoff=pars.ecut_TeV * u.TeV,
        beta=pars.beta,
    )

    nu_ecpl = PionDecayKelner06(ecpl, particle_type="muon_neutrino", oscillation_factor=0.5)
    nu_spec = NaimaSpectralModel(nu_ecpl, distance=source.distance_kpc * u.kpc)

    model = SkyModel(
        spectral_model=nu_spec,
        spatial_model=spatial_model,
        name="nu_PD",
        datasets_names=datasets_names,
    )

    model.spectral_model.beta.min = 0.1
    model.spectral_model.beta.max = 2.0
    model.spectral_model.e_cutoff.min = 0.0
    model.spectral_model.e_0.frozen = True
    model.spatial_model.parameters.freeze_all()
    return model

def snapshot_pd_parameters(model) -> dict:
    return dict(
        amplitude=model.spectral_model.amplitude.value,
        alpha=model.spectral_model.alpha.value,
        beta=model.spectral_model.beta.value,
        ecut=model.spectral_model.e_cutoff.value,
    )

def restore_pd_parameters(model, snap: dict):
    model.spectral_model.amplitude.value = snap["amplitude"]
    model.spectral_model.alpha.value = snap["alpha"]
    model.spectral_model.beta.value = snap["beta"]
    model.spectral_model.e_cutoff.value = snap["ecut"]

def build_nuisance_prior(pars: PDParams, rel_err: float = 0.3) -> dict:
    return dict(
        amp=pars.amplitude_1_over_eV,
        amp_err=rel_err * abs(pars.amplitude_1_over_eV),
        amp_unit=u.Unit("1/eV"),
        alpha=pars.alpha,
        alpha_err=rel_err * abs(pars.alpha),
        beta=pars.beta,
        beta_err=rel_err * abs(pars.beta),
        ecut=pars.ecut_TeV,
        ecut_err=rel_err * abs(pars.ecut_TeV),
        ecut_unit=u.TeV,
    )

def fit_model_to_fake_data(datasets, model, f_injected: float, seed: int):
    snap = snapshot_pd_parameters(model)
    model.spectral_model.amplitude.value *= f_injected

    for ds in datasets:
        if ds.models is None:
            ds.models = []
        ds.models = [m for m in ds.models if getattr(m, "name", None) != model.name] + [model]

    for ds in datasets:
        ds.fake(random_state=seed)

    restore_pd_parameters(model, snap)

def run_likelihood_scan(
    datasets,
    model,
    nuisance: dict,
    scan_values,
    e_int_min,
    e_int_max,
    rel_err: float = 0.3,
    *,
    warm_start: bool = True,
    restore_before_first_fit: bool = True,
):
    scan_values = np.asarray(scan_values, dtype=float)
    snap = snapshot_pd_parameters(model)

    datasets.nuisance = nuisance
    fit = Fit_wp(datasets)

    if restore_before_first_fit:
        restore_pd_parameters(model, snap)
    int_had = model.spectral_model.integral(e_int_min, e_int_max)

    stats = np.empty(len(scan_values), dtype=float)
    stats_with_prior = np.empty(len(scan_values), dtype=float)
    integral_pd = np.empty(len(scan_values), dtype=float)

    for i, f_val in enumerate(scan_values):
        if (i == 0 and restore_before_first_fit) or (not warm_start):
            restore_pd_parameters(model, snap)

        nuisance["amp"] = snap["amplitude"] * float(f_val)
        nuisance["amp_err"] = rel_err * abs(nuisance["amp"]) + 1e20
        model.spectral_model.amplitude.value = nuisance["amp"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fit.optimize(datasets=datasets)
            stats_with_prior[i] = float(result.total_stat)
            stats[i] = float(datasets.stat_sum_no_prior())
            integral_pd[i] = model.spectral_model.integral(e_int_min, e_int_max).to_value("cm-2 s-1")

    return dict(
        values=scan_values,
        stat=stats,
        stat_with_prior=stats_with_prior,
        int_PD=integral_pd,
        int_had=int_had.to_value("cm-2 s-1"),
    )

def _fmt_f(f: float) -> str:
    s = f"{float(f):.3f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")

def save_scan_results(res: dict, out_folder, f_input: float, seed: int):
    out_folder = Path(out_folder)
    f_tag = _fmt_f(f_input)

    subdir = out_folder / f"results_f{f_tag}"
    subdir.mkdir(parents=True, exist_ok=True)

    fname = subdir / f"results_f{f_tag}_s{seed}.npy"
    np.save(fname, res, allow_pickle=True)

    print(f"[INFO] Saved: {subdir.name}/{fname.name}")
    return fname

def load_scan_files(results_folder: str | Path, f_input: float | str | None = None, *, verbose: bool = False):
    """
    Load cached scan results from a folder.

    This function is tolerant to different filename conventions
    (e.g. f=1.0 vs f=1p0) and different np.save formats.
    """

    results_folder = Path(results_folder)

    if f_input is None:
        patterns = ["results_f*_s*.npy"]
    else:
        f_str = str(f_input)
        f_str_p = f_str.replace(".", "p")
        patterns = [
            f"results_f{f_str}_s*.npy",
            f"results_f{f_str_p}_s*.npy",
        ]

    files = []
    for pat in patterns:
        files.extend(results_folder.glob(pat))
    files = sorted(set(files))

    data = []
    first_err = None
    first_bad = None

    for fn in files:
        try:
            obj = np.load(fn, allow_pickle=True)

            # Case 1: array(dict, dtype=object)
            if isinstance(obj, np.ndarray) and obj.shape == ():
                obj = obj.item()

            # Case 2: dict directly
            if isinstance(obj, dict):
                data.append(obj)
            elif hasattr(obj, "item"):
                maybe = obj.item()
                if isinstance(maybe, dict):
                    data.append(maybe)
                else:
                    raise TypeError(f"Loaded object is not a dict (type={type(maybe)})")
            else:
                raise TypeError(f"Loaded object is not supported (type={type(obj)})")

        except Exception as e:
            if first_err is None:
                first_err = e
                first_bad = fn

    if verbose and first_err is not None:
        print(f"[WARN] First load failure: {first_bad.name} -> {type(first_err).__name__}: {first_err}")

    return files, data

def delta_ts_from_stat_with_prior(stat_with_prior: np.ndarray) -> np.ndarray:
    ts_min = np.min(stat_with_prior)
    return stat_with_prior - ts_min

def median_and_credible_band(all_delta_ts: np.ndarray, confidence_level: float):
    median = np.median(all_delta_ts, axis=0)
    lo = 100 * (0.5 - confidence_level / 2.0)
    hi = 100 * (0.5 + confidence_level / 2.0)
    band_lo = np.percentile(all_delta_ts, lo, axis=0)
    band_hi = np.percentile(all_delta_ts, hi, axis=0)
    return median, band_lo, band_hi

def run_likelihood_scan_no_opt(datasets, model, nuisance, scan_values, e_int_min, e_int_max, rel_err=0.3):
    scan_values = np.asarray(scan_values, dtype=float)
    snap = snapshot_pd_parameters(model)

    int_had = model.spectral_model.integral(e_int_min, e_int_max)

    stats = np.empty(len(scan_values), float)
    integral_pd = np.empty(len(scan_values), float)

    for i, f_val in enumerate(scan_values):
        amp0 = snap["amplitude"] * float(f_val)
        model.spectral_model.amplitude.value = amp0
        stats[i] = float(datasets.stat_sum_no_prior())
        integral_pd[i] = model.spectral_model.integral(e_int_min, e_int_max).to_value("cm-2 s-1")

    return dict(
        values=scan_values,
        stat=stats,
        stat_with_prior=np.full_like(stats, np.nan),
        int_PD=integral_pd,
        int_had=int_had.to_value("cm-2 s-1"),
    )
