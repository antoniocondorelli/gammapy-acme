from naima.radiative import BaseProton
import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.constants import c
from naima.utils import trapz_loglog
from pathlib import Path
import pandas as pd
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from gammapy.maps import WcsGeom, MapAxis, WcsNDMap

from astropy.time import Time
from gammapy.data import Observation
from gammapy.datasets import MapDataset, Datasets
from gammapy.makers import MapDatasetMaker, SafeMaskMaker
from naima.models import ExponentialCutoffPowerLaw
from gammapy.modeling.models import (
    SkyModel,
    Models,
    DiskSpatialModel,
    GaussianSpatialModel,
    TemplateNPredModel,
    NaimaSpectralModel,
    DatasetModels,
)
from gammapy.maps import WcsGeom, MapAxis, WcsNDMap
from gammapy.irf import (
    PSF3D,
    EnergyDispersion2D,
    Background2D,
    EffectiveAreaTable2D,
)
from naima.extern.validator import (
    validate_array,
    validate_physical_type,
    validate_scalar,
)
import yaml
import os

edge_coord_pixel_pos = [
    (299, 0), (299, 150), (299, 299),
    (150, 0), (150, 150), (150, 299),
    (0, 0), (0, 150), (0, 299)
]
def calc_exposure(offset, aeff, geom):
    energy = geom.axes['energy_true'].center
    exposure = aeff.evaluate(offset=offset, energy_true=energy[:, np.newaxis, np.newaxis])
    return exposure

def make_map_exposure_true_energy_patched(offset, livetime, aeff, geom):
    exposure = calc_exposure(offset, aeff, geom)
    exposure = (exposure * livetime).to('m2 s')
    return WcsNDMap(geom, exposure.value.reshape(geom.data_shape), unit=exposure.unit)

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
        """
        Inelastic cross-section for p-p interaction. Kelner Eq. 73

        Parameters
        ----------
        Tp : quantity
            Kinetic energy of proton (i.e. Ep - m_p*c**2) [GeV]

        Returns
        -------
        sigma_inel : float
            Inelastic cross-section for p-p interaction [1/cm2].

        """
        L = np.log((Tp.to('TeV').value))
        sigma = 34.3 + 1.88 * L + 0.25 * L ** 2 #* (1-Eth**4)**2

        return sigma * 1e-27  # convert from mbarn to cm-2

    def _Fnu1(self, Ep, x):
        """
        Energy distribution of mu neutrinos per pp-interaction from pi --> mu nu
        Kelner Eq. 66

        Ep: quantity [eV]
        x: Enu/Ep
        """
        y=x/0.427 # cutoff due momentum-energy-conservation
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
        """
        Energy distibution of mu neutrinos per pp-interaction from nu --> e + nu_e + nu_mu
        Kelner Eq. 62

        Ep: quantity [eV]
        x: Enu/Ep
            """
        L= np.log(Ep.to('TeV').value)
        B=1/(69.5 + 2.65*L + 0.3*L**2)
        beta=(0.201 + 0.062*L + 0.00042*L**2)**(-0.25)  # can become Nan for negativ L (Ep < ~100 GeV)
        k=(0.279 + 0.141*L + 0.0172*L**2)/(0.3 + (2.3+L)**2)
        F2= B * (1+ k*np.log(x)**2)**3 /(x*(1+0.3/x**beta)) *(-np.log(x))**5
        F2[x>=1]=0
        return F2

    def _Fgamma(self, x, Ep):
        """
        KAB06 Eq.58

        Note: Quantities are not used in this function

        Parameters
        ----------
        x : float
            Egamma/Eprot
        Ep : float
            Eprot [TeV]
        """
        L = np.log(Ep.to('TeV').value)
        B = 1.30 + 0.14 * L + 0.011 * L ** 2  # Eq59
        beta = (1.79 + 0.11 * L + 0.008 * L ** 2) ** -1  # Eq60
        k = (0.801 + 0.049 * L + 0.014 * L ** 2) ** -1  # Eq61
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
        """
        Compute differential spectrum from pp interactions using the
        parametrization of Kelner et al. 2008
        'Accuracy better than 10 percent over the range of parent protons 0,1 - 10⁵ TeV and x=E_nu/E_p >= 1e-3'

        Parameters
        ----------
        neutrino_energy : :class:`~astropy.units.Quantity` instance
            neutrino energy array.
        """



        Enu = _validate_ene(neutrino_energy).to("GeV")
        if Enu.min()<50*u.GeV:
            import warnings
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

            import warnings
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


def load_source_table(source_dir: Path) -> pd.DataFrame:
    return pd.read_csv(source_dir / "source_list.txt",
                       delim_whitespace=True,
                       index_col=0)


def get_source_position(source_name: str,
                        table: pd.DataFrame) -> SkyCoord:
    if source_name not in table.index:
        raise ValueError(f"Source '{source_name}' not in source_list.txt")
    row = table.loc[source_name]
    return SkyCoord(row["RA"], row["Dec"], unit="deg", frame="icrs")

def stack_maps(maps):

    if len(maps) == 0:
        raise ValueError("Cannot stack an empty list of maps.")

    result = maps[0].copy()
    for m in maps[1:]:
        result += m
    return result

def compute_npred_per_dataset(datasets):

    print("==> Computing predicted counts (npred) per dataset...")

    npred_sum = []
    npred_src = []

    for i, ds in enumerate(datasets):
        npred_total = ds.npred()
        npred_sig = ds.npred_signal()

        npred_src.append(npred_sig)
        npred_sum.append(npred_total)

        print(
            f"Dataset {i}: npred_total={npred_total.data.sum():.2f}, "
            f"npred_signal={npred_sig.data.sum():.2f}"
        )

    return npred_src, npred_sum

import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation

def make_observer_and_times(
    time_step: float,
    location: EarthLocation | None = None,
    start_time: str = "2023-01-01T00:00:00",
    end_time: str = "2024-01-01T00:00:00",
) -> tuple[EarthLocation, Time]:
    # Default: ARCA-like position if no location is provided
    if location is None:
        location = EarthLocation.from_geodetic(
            lat="36d16m",   # 36° 16'
            lon="16d06m",   # 16° 06'
            height=-3500 * u.m,
        )

    t_start = Time(start_time, format="isot", scale="utc")
    t_end   = Time(end_time,   format="isot", scale="utc")

    # Build time grid in UNIX seconds
    dt_sec = t_end.unix - t_start.unix
    n_steps = int(np.floor(dt_sec / time_step))
    times_unix = t_start.unix + np.arange(n_steps) * time_step

    obstimes = Time(times_unix, format="unix", scale="utc")
    return location, obstimes

def compute_visibility(src_pos: SkyCoord,
                       pos_arca: EarthLocation,
                       obstimes: Time,
                       cos_zen_bins: np.ndarray,
                       source_name: str):
    frames = AltAz(obstime=obstimes, location=pos_arca)
    local_coords = src_pos.transform_to(frames)

    zen_angles_std = local_coords.zen.value
    zen_angles = 180.0 - zen_angles_std

    cos_zen_binc = 0.5 * (cos_zen_bins[:-1] + cos_zen_bins[1:])
    zen_bins = np.degrees(np.arccos(cos_zen_bins))
    zen_binc = np.degrees(np.arccos(cos_zen_binc))
    cos_zen_values = np.cos(np.radians(zen_angles))

    vis_hist, _ = np.histogram(cos_zen_values, bins=cos_zen_bins)
    vis_times = vis_hist / vis_hist.sum() * u.yr
    bin_mask = (zen_binc < 100) & (vis_hist > 0)

    print("===> Included zenith bins (< 100° and visible):")
    for i, valid in enumerate(bin_mask):
        if valid:
            print(f"  Bin {i}: {zen_bins[i]:.1f}° – {zen_bins[i+1]:.1f}°")

    zen_min = np.min(zen_angles)
    zenith_threshold = 33.6  # deg
    print(f"===> Minimum zenith angle reached by {source_name}: {zen_min:.2f}°")
    if zen_min < zenith_threshold:
        print(f"===> {source_name} reaches zenith.")
    else:
        print(f"===> {source_name} does NOT reach zenith.")

    return zen_angles, vis_hist, vis_times, bin_mask, zen_bins
def setup_geoms(src_pos: SkyCoord,
                bin_mask,
                zen_bins,
                vis_times,
                time_step: float):
    print("===> Setting up map geometry and energy axes...")
    BIN_SIZE = 0.1 * u.deg
    MAP_WIDTH = 6 * u.deg

    energy_axis = MapAxis.from_edges(
        np.logspace(2, 7, 26),
        unit="GeV",
        name="energy"
    )
    energy_axis_true = MapAxis.from_edges(
        np.logspace(2, 7.5, 31),
        unit="GeV",
        name="energy_true"
    )
    migra_axis = MapAxis.from_edges(
        np.logspace(-5, 2, 57),
        name="migra"
    )
    rad_axis = MapAxis.from_edges(
        np.linspace(0, 8, 101),
        name="rad",
        unit="deg"
    )

    geom = WcsGeom.create(
        binsz=BIN_SIZE,
        width=MAP_WIDTH,
        skydir=src_pos,
        frame="icrs",
        axes=[energy_axis]
    )
    geom_true = WcsGeom.create(
        binsz=BIN_SIZE,
        width=MAP_WIDTH,
        skydir=src_pos,
        frame="icrs",
        axes=[energy_axis_true]
    )

    # Livetime per pointing:
    livetime_pointings = (vis_times[bin_mask] * 10).to("s")

    zen_bins_selected = np.vstack(
        (zen_bins[:-1][bin_mask], zen_bins[1:][bin_mask])
    ).T

    return (
        geom, geom_true,
        energy_axis, energy_axis_true,
        migra_axis, rad_axis,
        livetime_pointings, zen_bins_selected
    )


def load_irfs(irf_dir: Path):
    print("===> Loading IRFs...")
    aeff = EffectiveAreaTable2D.read(irf_dir / "aeff_no_zenith.fits")
    edisp = EnergyDispersion2D.read(irf_dir / "edisp_no_zenith.fits")
    psf = PSF3D.read(irf_dir / "psf_no_zenith.fits")
    bkg_nu = Background2D.read(irf_dir / "bkg_nu.fits")
    bkg_mu = Background2D.read(irf_dir / "bkg_mu.fits")
    irfs = dict(aeff=aeff, edisp=edisp, psf=psf)
    return irfs, bkg_nu, bkg_mu


def patch_background_evaluate():

    original_evaluate = Background2D.evaluate

    def bkg_2d_eval_patched(self, offset, energy, method="linear", **kwargs):
        kwargs.pop("fov_lat", None)
        kwargs.pop("fov_lon", None)

        return original_evaluate(
            self,
            offset=offset,
            energy=energy,
            method=method,
            **kwargs,
        )

    # Monkey-patch
    Background2D.evaluate = bkg_2d_eval_patched

    print("===> Patched Background2D.evaluate")


def make_datasets(pointings, livetime_pointings, irfs,
                  geom, geom_true, energy_axis_true,
                  migra_axis, rad_axis):
    print("===> Creating Observations and MapDatasets...")
    obs_list = [
        Observation.create(
            pointing=pointings[i],
            livetime=livetime_pointings[i],
            irfs=irfs
        )
        for i in range(len(pointings))
    ]
    empty = MapDataset.create(
        geom=geom,
        energy_axis_true=energy_axis_true,
        migra_axis=migra_axis,
        rad_axis=rad_axis,
        binsz_irf=1,
    )

    dataset_maker = MapDatasetMaker(selection=["exposure", "psf", "edisp"])
    datasets = []
    for i, obs in enumerate(obs_list):
        ds = dataset_maker.run(empty.copy(name=f"nu{i+1:02d}"), obs)
        datasets.append(ds)
    return datasets

def fill_background_and_exposure_maps(
    datasets,
    obstimes,
    dataset_idx,
    geom,
    geom_true,
    irfs,
    bkg_nu,
    bkg_mu,
    time_step,
    pos_arca,
):

    aeff = irfs["aeff"]
    n_datasets = len(datasets)

    # -----------------------------------------
    # Zenith angle map per time bin
    # -----------------------------------------
    frames = AltAz(obstime=obstimes, location=pos_arca)

    map_coord = geom_true.to_image().get_coord()
    sky_coord = map_coord.skycoord

    # shape: (n_times, ny, nx)
    map_coord_zeniths = np.array([
        sky_coord.transform_to(frames[i]).zen.value
        for i in range(len(obstimes))
    ])

    # Pixel solid angle (sr), shape: (ny, nx)
    d_omega = geom_true.to_image().solid_angle()

    # -----------------------------------------------
    # CONTENITORI FINALI
    # -----------------------------------------------
    nu_background_maps = np.zeros((n_datasets, *geom.data_shape))
    mu_background_maps = np.zeros((n_datasets, *geom.data_shape))
    exposure_maps = np.zeros((n_datasets, *geom_true.data_shape))

    #-------------------------------------------------------
    # LOOP OVER TIME BINS TO FILL BACKGROUND & EXPOSURE MAPS
    #-------------------------------------------------------
    for i in range(len(obstimes)):
        # Print progress every 100 time bins
        if (i + 1) % 100 == 0:
            print(f"===> Processed {i + 1} / {len(obstimes)}")

        idx = dataset_idx[i]

        # Skip se indice fuori range
        if idx < 0 or idx >= n_datasets:
            continue

        zen_vals = map_coord_zeniths[i]  # shape (ny, nx)

        # ---- neutrino background ----
        bkg_nu_de = bkg_nu.integrate_log_log(
            axis_name="energy",
            offset=zen_vals * u.deg,
            energy=geom.axes["energy"].edges[:, np.newaxis, np.newaxis],
        )
        # aggiungi fattore dOmega -> 1/s per pixel e bin di energia
        bkg_nu_de = (bkg_nu_de * d_omega).to_value("s-1")
        nu_background_maps[idx] += bkg_nu_de

        # ---- muon background ----
        bkg_mu_de = bkg_mu.integrate_log_log(
            axis_name="energy",
            offset=zen_vals * u.deg,
            energy=geom.axes["energy"].edges[:, np.newaxis, np.newaxis],
        )
        bkg_mu_de = (bkg_mu_de * d_omega).to_value("s-1")
        mu_background_maps[idx] += bkg_mu_de

        # ---- exposure ----
        exp = (
            calc_exposure(zen_vals * u.deg, aeff, geom_true)
            * time_step
            * u.s
        ).to_value("m2 s")
        exposure_maps[idx] += exp

    # Scala a 10 anni totali (se il resto del codice lo assume)
    exposure_maps *= 10

    return nu_background_maps, mu_background_maps, exposure_maps


def build_source_model(source_name, src_pos, source_table, source_dir, spatial_model):

    print("==> Loading hadronic (parent) model parameters from YAML...")

    model_dir = Path(source_dir)
    model_yaml = model_dir / f"{source_name}_model.yaml"

    with model_yaml.open("r") as f:
        p = yaml.safe_load(f)

    # Parent proton ECPL (unità 'differential energy', es. 1/eV)
    proton_pl = ExponentialCutoffPowerLaw(
        amplitude=float(p["amplitude"]) * u.Unit(p.get("amplitude_unit", "1/eV")),
        e_0=float(p["e_0"]) * u.Unit(p.get("e_0_unit", "TeV")),
        alpha=float(p["alpha"]),
        e_cutoff=float(p["e_cutoff"]) * u.Unit(p.get("e_cutoff_unit", "TeV")),
        beta=float(p["beta"]),
    )

    # Pion-decay → neutrino flux
    nu_ECPL_PD = PionDecayKelner06(
        proton_pl,
        particle_type="muon_neutrino",
        oscillation_factor=0.5,
    )

    dist_kpc = source_table.loc[source_name, "Dist"] * u.kpc
    radius_deg = source_table.loc[source_name, "Radius"] * u.deg

    spectral_model = NaimaSpectralModel(
        nu_ECPL_PD,
        distance=dist_kpc,
    )

    model = SkyModel(
        spectral_model=spectral_model,
        spatial_model=spatial_model,
        name=source_name,
    )

    print(f"==> Built source SkyModel for '{source_name}'")
    return model

def compute_and_save_background_npred(
    bkg_models,
    bkg_models_mu,
    source_name,
    outdir="npred_bkg_km3net",
    save=True,
):

    n_datasets = len(bkg_models)
    os.makedirs(outdir, exist_ok=True)

    #-------------------------------------------------------
    # COMPUTE NPRED VALUES (EVENT EXPECTATION PER PIXEL)
    #-------------------------------------------------------
    npred_bkg_nu = []
    npred_bkg_mu = []

    for i in range(n_datasets):
        npred_nu = bkg_models[i]
        npred_mu = bkg_models_mu[i]

        if npred_nu is None:
            print(f"Dataset {i} - Neutrino npred is None!")
        else:
            npred_bkg_nu.append(npred_nu)
            print(f"Dataset {i} - Neutrino npred sum: {npred_nu.data.sum()}")

        if npred_mu is None:
            print(f"Dataset {i} - Muon npred is None!")
        else:
            npred_bkg_mu.append(npred_mu)
            print(f"Dataset {i} - Muon npred sum: {npred_mu.data.sum()}")

    if save:
        for i in range(n_datasets):
            npred_bkg_nu[i].write(
                f"{outdir}/{source_name}_npred_nu_{i+1:02d}.fits",
                overwrite=True,
            )
            npred_bkg_mu[i].write(
                f"{outdir}/{source_name}_npred_mu_{i+1:02d}.fits",
                overwrite=True,
            )

    #-------------------------------------------------------
    # COMBINE NPRED TOTAL
    #-------------------------------------------------------
    print("==> Summing neutrino and muon background predictions per dataset...")
    npred_bkg_total = []
    for i in range(n_datasets):
        npred_total = npred_bkg_nu[i] + npred_bkg_mu[i]
        npred_bkg_total.append(npred_total)
        print(f"Dataset {i} - total background npred sum: {npred_total.data.sum()}")

    return npred_bkg_nu, npred_bkg_mu, npred_bkg_total

def prepare_background_models(
    datasets,
    geom,
    geom_true,
    nu_background_maps,
    mu_background_maps,
    exposure_maps,
    dataset_idx,
    livetime_pointings,
):

    n_datasets = len(datasets)

    #-------------------------------------------------------
    # ASSIGN EXPOSURE MAPS TO EACH DATASET
    #-------------------------------------------------------
    for i in range(n_datasets):
        exp_map = WcsNDMap(geom_true.copy(), data=exposure_maps[i], unit="m2 s")
        datasets[i].exposure = exp_map

    #-------------------------------------------------------
    # NORMALIZE BACKGROUND MAPS AND APPLY LIVETIME
    #-------------------------------------------------------
    for i in range(n_datasets):
        n_times = (dataset_idx == i).sum()

        if n_times == 0 or np.isnan(n_times):
            n_times = max(n_times, 1e-10)

        # rate medio per time bin
        nu_background_maps[i] /= n_times
        mu_background_maps[i] /= n_times

        # applica il livetime del dataset → aspettativa di conteggi
        nu_background_maps[i] *= livetime_pointings[i].to_value("s")
        mu_background_maps[i] *= livetime_pointings[i].to_value("s")

    #-------------------------------------------------------
    # CREATE BACKGROUND MAP MODELS (NEUTRINOS + MUONS)
    # COMBINE BACKGROUND MAPS AND ASSIGN TO DATASETS
    #-------------------------------------------------------
    bkg_models = []
    bkg_models_mu = []
    bkg_combined_models = []

    for i in range(n_datasets):
        bkg_model_map = WcsNDMap(geom.copy(), data=nu_background_maps[i], unit="s-1")
        bkg_models.append(bkg_model_map)

        bkg_model_mu_map = WcsNDMap(geom.copy(), data=mu_background_maps[i], unit="s-1")
        bkg_models_mu.append(bkg_model_mu_map)

        bkg_combined_map = WcsNDMap(
            geom.copy(),
            data=bkg_models[i].data + bkg_models_mu[i].data,
        )
        bkg_combined_models.append(bkg_combined_map)

        datasets[i].background = bkg_combined_map

        # Display background info for verification
        print(f"Dataset {i}")
        #print(f"  Background: {datasets[i].background}")
        #print(f"  Background type: {type(datasets[i].background)}")
        #print(f"  Background geometry: {datasets[i].background.geom}")

    return bkg_models, bkg_models_mu, bkg_combined_models

def make_energy_counts_plot(
    energy_axis,
    npred_bkg_nu_all,
    npred_bkg_mu_all,
    npred_src_all,
    npred_sum_all,
    counts_all,
    src_reg_mask,
    source_name,
):

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 7,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 1,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "legend.frameon": False,
    })

    def feldman_cousins_errors(n_obs):
        n_obs = np.asarray(n_obs)
        if (not (n_obs % 1 == 0).all()) or (n_obs < 0).any():
            raise ValueError("Input must consist of positive integer values!")

        lookup = {
            0: (0.00, 1.29),
            1: (0.63, 1.75),
            2: (1.26, 2.25),
            3: (1.90, 2.30),
            4: (1.66, 2.78),
            5: (2.25, 2.81),
            6: (2.18, 3.28),
            7: (2.75, 3.30),
            8: (2.70, 3.32),
            9: (2.67, 3.79),
            10: (3.22, 3.81),
            11: (3.19, 3.82),
            12: (3.17, 4.29),
            13: (3.72, 4.30),
            14: (3.70, 4.32),
            15: (3.68, 4.32),
            16: (3.67, 4.80),
            17: (4.21, 4.81),
            18: (4.19, 4.82),
            19: (4.18, 4.82),
            20: (4.17, 5.30),
        }

        return np.array(
            list(
                map(
                    lambda i: lookup.get(i, (np.sqrt(i), np.sqrt(i))),
                    n_obs,
                )
            )
        )

    # ---------- Energy vs Counts ------------
    print("===> Creating energy vs counts plot")

    e = energy_axis.center.value / 1e3

    fig, ax = plt.subplots(figsize=(3.5, 2.2), dpi=200)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$E\,[\mathrm{TeV}]$")
    ax.set_ylabel("Counts in source region")
    ax.set_title("Pseudo dataset")

    ax.plot(
        e,
        npred_bkg_nu_all.data[:, src_reg_mask].sum(axis=1),
        color="tab:cyan",
        label=r"$\nu$ background",
    )
    ax.plot(
        e,
        npred_bkg_mu_all.data[:, src_reg_mask].sum(axis=1),
        color="tab:cyan",
        ls="--",
        label=r"$\mu$ background",
    )
    ax.plot(
        e,
        npred_src_all.data[:, src_reg_mask].sum(axis=1),
        color="tab:red",
        label="Signal",
    )
    ax.plot(
        e,
        npred_sum_all.data[:, src_reg_mask].sum(axis=1),
        color="tab:green",
        label="Sum",
    )

    cts = counts_all.data[:, src_reg_mask].sum(axis=1)
    e_plot = e[cts > 0]
    cts_plot = cts[cts > 0]

    ax.errorbar(
        e_plot,
        cts_plot,
        xerr=None,
        yerr=feldman_cousins_errors(cts_plot).T,
        linestyle="None",
        marker="o",
        markersize=3,
        color="k",
        label="flux points",
        zorder=8,
    )

    ax.text(
        0.97,
        0.95,
        source_name,
        ha="right",
        va="top",
        transform=ax.transAxes,
    )
    ax.legend(loc="lower left", ncol=2, frameon=False)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    fig.savefig(
        f"plots/counts_{source_name}.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        f"plots/counts_{source_name}.png",
        format="png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()



def add_models_and_fake_data(
    datasets,
    src_model,
    random_state=314,
):

    n = len(datasets)

    print("==> Assigning source model to datasets...")
    for i in range(n):
        datasets[i].models = DatasetModels([src_model])

    print("==> Computing predicted counts (npred) per dataset...")
    npred_sum = []
    npred_src = []
    for i in range(n):
        npred_sum.append(datasets[i].npred())
        npred_src.append(datasets[i].npred_signal())

    print("==> Generating simulated counts (pseudo-dataset)...")
    rs = np.random.RandomState(random_state) if isinstance(random_state, int) else random_state
    for i in range(n):
        datasets[i].fake(rs)

    counts_all = datasets[0].counts.copy()
    for ds in datasets[1:]:
        counts_all += ds.counts

    return datasets, npred_src, npred_sum, counts_all


def quick_visibility_diagnostics(
    obstimes=None,
    zen_angles=None,
    bin_mask=None,
    vis_times=None,
    vis_hist=True,
    src_pos=None,
    location=None,
    show=True,
):
    """
    Build a consistent (time, zenith) pair and produce quick diagnostic plots.

    Parameters
    ----------
    obstimes : astropy.time.Time or None
        Full time grid used in visibility computation.
    zen_angles : array-like or astropy.units.Quantity or None
        Zenith angles corresponding to either the full time grid or visible subset.
    bin_mask : array-like of bool or None
        Boolean mask on the full grid selecting "visible" time steps.
    vis_times : astropy.time.Time or astropy.units.Quantity or None
        Either a Time array of visible times or time offsets (e.g. days/years).
    vis_hist : array-like or None
        Histogram of visibility per zenith bin (for optional second plot).
    src_pos : astropy.coordinates.SkyCoord or None
        Source coordinate used to recompute zenith if needed.
    location : astropy.coordinates.EarthLocation or None
        Observatory location used together with src_pos.
    show : bool
        If True, call plt.show() at the end.

    Notes
    -----
    The function is robust to several internal representations:

      - Masked full time grid (obstimes + bin_mask)
      - vis_times as Time (subset)
      - vis_times as Quantity (offsets from an implied anchor time)
      - Fallback: recompute zenith from obstimes + (src_pos, location)

    It will:
      - build a Time array `t`
      - compute a 1D zenith-angle array `y_deg` in degrees
      - plot zenith vs time
      - optionally plot vis_hist if provided
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import astropy.units as u
    from astropy.time import Time, TimeDelta
    from astropy.coordinates import AltAz

    t = None
    y_deg = None

    # Case 1: masked full grid (obstimes + bin_mask)
    if (obstimes is not None) and (bin_mask is not None):
        try:
            if isinstance(obstimes, Time) and len(bin_mask) == len(obstimes):
                t = obstimes[bin_mask]

                if zen_angles is not None:
                    za = u.Quantity(zen_angles)
                    if len(za) == len(obstimes):
                        y_deg = za[bin_mask].to_value("deg")
                    else:
                        # assume already subset
                        y_deg = za.to_value("deg")
        except Exception:
            t = None
            y_deg = None

    # Case 2: vis_times is already a Time array (visible subset)
    if t is None and vis_times is not None and isinstance(vis_times, Time):
        t = vis_times
        if zen_angles is not None:
            y_deg = u.Quantity(zen_angles).to_value("deg")

    # Case 3: vis_times is a Quantity of offsets (e.g. days/years)
    if t is None and vis_times is not None and isinstance(vis_times, u.Quantity):
        # Anchor time: prefer obstimes[0] if available, otherwise "now" (for plotting only)
        if (obstimes is not None) and isinstance(obstimes, Time) and len(obstimes) > 0:
            t0 = obstimes[0]
        else:
            t0 = Time.now()

        t = t0 + TimeDelta(vis_times.to(u.day))

        # Try to align zen_angles, otherwise recompute from src_pos+location
        if zen_angles is not None and len(zen_angles) == len(vis_times):
            y_deg = u.Quantity(zen_angles).to_value("deg")
        elif (src_pos is not None) and (location is not None):
            altaz_vis = src_pos.transform_to(AltAz(obstime=t, location=location))
            y_deg = (90 * u.deg - altaz_vis.alt).to_value("deg")

    # Final fallback: recompute from obstimes if possible
    if t is None and (obstimes is not None) and isinstance(obstimes, Time):
        t = obstimes
        if (src_pos is not None) and (location is not None):
            altaz_all = src_pos.transform_to(AltAz(obstime=t, location=location))
            y_deg = (90 * u.deg - altaz_all.alt).to_value("deg")

    # --- Plot time vs zenith if we succeeded
    if t is None or y_deg is None:
        print("[quick_visibility_diagnostics] Could not build a consistent (time, zenith) pair.")
        print("  obstimes:", isinstance(obstimes, Time), "| vis_times is Time:", isinstance(vis_times, Time))
        print("  vis_times is Quantity:", isinstance(vis_times, u.Quantity), "| bin_mask provided:", bin_mask is not None)
    else:
        x_days = (t - t[0]).to_value("day")
        n = min(len(x_days), len(y_deg))

        plt.figure()
        plt.plot(x_days[:n], y_deg[:n])
        plt.xlabel("Time since start [days]")
        plt.ylabel("Zenith angle [deg]")
        plt.title("Zenith angle vs time")
        plt.grid(alpha=0.3)

        if show:
            plt.show()

    # --- Optional: histogram per zenith bin
    if vis_hist is not None:
        plt.figure()
        plt.step(range(len(vis_hist)), vis_hist, where="mid")
        plt.xlabel("Zenith bin index")
        plt.ylabel("Counts (time steps)")
        plt.title("Visibility histogram per zenith bin")
        plt.grid(alpha=0.3)
        if show:
            plt.show()
