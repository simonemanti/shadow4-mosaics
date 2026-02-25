from syned.beamline.optical_elements.crystals.crystal import Crystal, DiffractionGeometry
from syned.beamline.shape import SurfaceShape

# this will go to the syned package
class MosaicCrystal(Crystal):
    """
    Constructor.

    Parameters
    ----------
    name : str, optional
        The name of the optical element.
    surface_shape : instance of SurfaceShape, optional
        The geometry of the crystal surface. if None, it is initialized to SurfaceShape().
    boundary_shape : instance of BoundaryShape, optional
        The geometry of the slit aperture. if None, it is initialized to BoundaryShape().
    material : str, optional
        The material name.
    diffraction_geometry : int (as defined in DiffractionGeometry, optional
        BRAGG = 0, LAUE = 1.
    miller_index_h : int, optional
        The Miller index H.
    miller_index_k : int, optional
        The Miller index K.
    miller_index_l : int, optional
        The Miller index L.
    asymmetry_angle : float, optional
        The asymmetry angle in rad.
    thickness : float, optional
        The crystal thickness in m.
    mosaicity_fwhm_deg : float, optional
        The crystal mosaicity (FWHM) in degrees.
    mosaicity_profile_flag : int, optional
        The distribution function of the crystallites:
        0: Gaussian,
        1: Lorentzian.

    """
    def __init__(self,
                 name="Undefined",
                 surface_shape=None,
                 boundary_shape=None,
                 material="Graphite",
                 diffraction_geometry=DiffractionGeometry.BRAGG,
                 miller_index_h=1,
                 miller_index_k=1,
                 miller_index_l=1,
                 asymmetry_angle=0.0,
                 thickness=0.0,
                 mosaicity_fwhm_deg=0.4,
                 mosaicity_profile_flag=0, # 0=Gaussian, 1=Lorentzian
                 ):
        if surface_shape is None: surface_shape = SurfaceShape()
        super().__init__(
            name=name,
            surface_shape=surface_shape,
            boundary_shape=boundary_shape,
            material=material,
            diffraction_geometry=DiffractionGeometry.BRAGG,
            miller_index_h=miller_index_h,
            miller_index_k=miller_index_k,
            miller_index_l=miller_index_l,
            asymmetry_angle=asymmetry_angle,
            thickness=thickness,
            )


        self._mosaicity_fwhm_deg = mosaicity_fwhm_deg
        self._mosaicity_profile_flag = mosaicity_profile_flag

        # support text containg name of variable, help text and unit. Will be stored in self._support_dictionary
        self._add_support_text([
                    ("mosaicity_fwhm_deg",     "Mosaicity fwhm" ,                                      "deg" ),
                    ("mosaicity_profile_flag", "Mosaic distribution profile 0=Gaussian, 1=Lorentzian" , "" ),
            ] )

if __name__ == "__main__":
    a = MosaicCrystal()
    print(a.info())
    print("distribution: 0=Gaussian, 1=Lorentzian: ", a._mosaicity_profile_flag)
