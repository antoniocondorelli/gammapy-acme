from naima.radiative import BaseProton
import numpy as np
from astropy import units as u
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
