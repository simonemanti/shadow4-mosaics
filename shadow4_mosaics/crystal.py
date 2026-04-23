from crystalpy.diffraction.GeometryType import BraggDiffraction
from crystalpy.diffraction.DiffractionSetupDabax import DiffractionSetupDabax
from crystalpy.util.Photon import Photon

from dataclasses import dataclass, field

import numpy as np

import scipy.constants as codata


@dataclass
class MosaicCrystal:
    material: str = "Graphite"
    hkl: tuple[int, int, int] = (0, 0, 2)
    thickness_cm: float = 0.1
    mosaicity_fwhm_deg: float = 0.4
    mosaic_model: str = "gaussian"

    _setup: DiffractionSetupDabax = field(init=False, repr=False)

    def __post_init__(self):

        h, k, l = self.hkl
        thickness_m = self.thickness_cm * 1e-2

        self._setup = DiffractionSetupDabax(
            geometry_type=BraggDiffraction(),
            crystal_name=self.material,
            thickness=thickness_m,
            miller_h=h,
            miller_k=k,
            miller_l=l,
            asymmetry_angle=0.0,
            azimuthal_angle=0.0,
            dabax=None,   # let crystalpy manage dabax
        )

    def asymmetry_factor(self, energy_eV):
        return self._setup.asymmetryFactor(energy_eV)

    def bragg_angle(self, energy_eV):
        return self._setup.angleBragg(
            np.asarray(energy_eV, dtype=float)
        )
    
    @property
    def d_spacing_A(self):
        return self._setup.dSpacing()
    
    def deviation_of_incoming_photon(self, energy_eV, direction_vector=None):
        energy = np.asarray(energy_eV, dtype=float)

        if direction_vector is None:
            direction_vector = self.vectorK0(energy)

        photon = Photon(
            energy_in_ev=energy,
            direction_vector=direction_vector
        )

        return self._setup.deviationOfIncomingPhoton(photon)
    
    def F0(self, energy_eV):
        energy = np.asarray(energy_eV, dtype=float)
        return self._setup.F0(energy)

    def FH(self, energy_eV):
        energy = np.asarray(energy_eV, dtype=float)
        return self._setup.FH(energy)

    def FH_bar(self, energy_eV):
        energy = np.asarray(energy_eV, dtype=float)
        return self._setup.FH_bar(energy)
    
    def mu_cm_inv(self, energy_eV):
        energy = np.asarray(energy_eV, dtype=float)
        lam_cm = self.wavelength_cm(energy)
        psi0 = self.psi0(energy)
        return -2 * np.pi / lam_cm * psi0.imag

    def psi0(self, energy_eV):
        energy = np.asarray(energy_eV, dtype=float)
        return self._setup.psi0(energy)

    def psiH(self, energy_eV):
        energy = np.asarray(energy_eV, dtype=float)
        return self._setup.psiH(energy)

    def psiH_bar(self, energy_eV):
        energy = np.asarray(energy_eV, dtype=float)
        return self._setup.psiH_bar(energy)    

    def Q_s_cm_inv(self, energy_eV):
        energy = np.asarray(energy_eV, dtype=float)
        lam_cm = self.wavelength_cm(energy)
        psiH = self.psiH(energy)
        psiHb = self.psiH_bar(energy)
        theta = self.bragg_angle(energy)

        return (
            np.pi**2
            * np.abs(psiH * psiHb)
            / (lam_cm * np.sin(2 * theta))
        )

    def Q_p_cm_inv(self, energy_eV):
        energy = np.asarray(energy_eV, dtype=float)
        theta = self.bragg_angle(energy)
        return self.Q_s_cm_inv(energy) * np.cos(2 * theta) ** 2

    def Q_cm_inv(self, energy_eV):
        return self.Q_s_cm_inv(energy_eV)

    @property
    def unitCellVolume_A3(self):
        return self._setup.unitcellVolume()
    
    def vectorK0direction(self, energy_eV):
        energy = np.asarray(energy_eV, dtype=float)
        return self._setup.vectorK0direction(energy)

    def vectorHdirection(self):
        return self._setup.vectorHdirection()

    def vectorH(self):
        return self._setup.vectorH()

    def vectorK0(self, energy_eV):
        energy = np.asarray(energy_eV, dtype=float)
        return self._setup.vectorK0(energy)

    def vectorKh(self, energy_eV):
        energy = np.asarray(energy_eV, dtype=float)
        return self._setup.vectorKh(energy)

    def vectorKhdirection(self, energy_eV):
        energy = np.asarray(energy_eV, dtype=float)
        return self._setup.vectorKhdirection(energy)
    
    def wavelength_cm(self, energy_eV):
        energy = np.asarray(energy_eV, dtype=float)
        lam_m = codata.h * codata.c / codata.e / energy
        return lam_m * 100.0
