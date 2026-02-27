from dataclasses import dataclass, field
import numpy as np

from crystalpy.diffraction.GeometryType import BraggDiffraction
from crystalpy.diffraction.DiffractionSetupDabax import DiffractionSetupDabax


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
    
    @property
    def unitCellVolume_A3(self):
        return self._setup.unitcellVolume()