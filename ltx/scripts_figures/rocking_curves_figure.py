"""
Figure for section 7.2 ("Rocking curves: effect of the source spectrum") of the
paper. Copied from rocking_curves.py (same beamline and sweep), with the final
plotting call changed to save the figure to
ltx/figures/rocking_curves_spectra.pdf instead of showing it interactively.

At the crystal's nominal mosaicity (0.2 deg FWHM) the three source spectra are
essentially indistinguishable (see the paper text): the 4 eV bandwidth maps to
an angular smearing about two orders of magnitude smaller than the mosaic
width. To make the effect of the source spectrum more visible, the mosaicity
here is reduced to 0.05 deg FWHM (4x smaller than section 7.1's HOPG example),
narrower than 0.2 deg but still well above the crystal's Darwin width
(~0.05 deg = 873 urad vs a Darwin width of order 50 urad for a strong
reflection): mosaicities close to the Darwin width would violate the model's
own validity condition (mosaicity >> perfect-crystallite rocking width,
section 4), so a smaller mosaicity than this was not used even though it
would make the effect more visible still (0.005 deg = 87 urad was tried
first and is of the same order as the Darwin width, i.e. outside the
mosaic-crystal regime).
"""
import numpy


def run_beamline(angle=0.0, spectrum='monochromatic', nrays=10000, mosaicity_fwhm_deg=0.05):
    footprint = None
    import numpy as np
    from dabax.dabax_xraylib import DabaxXraylib
    from shadow4.beamline.s4_beamline import S4Beamline

    beamline = S4Beamline()

    #
    # source
    #
    from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
    light_source = SourceGeometrical(name='Geometrical Source', nrays=nrays, seed=5676561)
    light_source.set_spatial_type_point()
    light_source.set_depth_distribution_off()
    light_source.set_angular_distribution_uniform(hdiv1=-0.0005, hdiv2=0.0005, vdiv1=-0.0001, vdiv2=0.0001)

    if spectrum == 'monochromatic':
        light_source.set_energy_distribution_singleline(value=8000.0, unit='eV')
    elif spectrum == 'box':
        light_source.set_energy_distribution_uniform(value_min=7998.0, value_max=8002.0, unit='eV')
    elif spectrum == 'lorentzian':
        from source_lorentzian_energy import set_energy_distribution_lorentzian
        set_energy_distribution_lorentzian(light_source, center=8000.0, fwhm=4.0)
    else:
        raise ValueError("spectrum must be 'monochromatic', 'box' or 'lorentzian'")

    light_source.set_polarization(polarization_degree=1, phase_diff=0, coherent_beam=0)
    beam = light_source.get_beam()

    beamline.set_light_source(light_source)

    # optical element number XX
    boundary_shape = None

    from shadow4.beamline.optical_elements.mosaic_crystals.s4_plane_mosaic_crystal import S4PlaneMosaicCrystal
    optical_element = S4PlaneMosaicCrystal(name='Mosaic Crystal PLANE (1)',
                                           boundary_shape=boundary_shape, material='Graphite',
                                           miller_index_h=0, miller_index_k=0, miller_index_l=2,
                                           thickness=0.001,
                                           f_central=1, f_phot_cent=0, phot_cent=8000.0,
                                           file_refl='HOPG2_20_v2.002',
                                           material_constants_library_flag=1,
                                           # 0=xraylib,1=dabax,2=preprocessor v1,3=preprocessor v2
                                           dabax=DabaxXraylib(file_f0="f0_InterTables.dat", file_f1f2="f1f2_Windt.dat"),
                                           # used when material_constants_library_flag=1,
                                           mosaicity_fwhm_deg=mosaicity_fwhm_deg,
                                           mosaicity_profile_flag=0,  # 0=Gaussian, 1=Lorentzian
                                           )
    from syned.beamline.element_coordinates import ElementCoordinates
    coordinates = ElementCoordinates(p=10, q=10, angle_radial=1.337226398, angle_azimuthal=0,
                                     angle_radial_out=1.337226398)
    from shadow4.beamline.s4_beamline_element_movements import S4BeamlineElementMovements

    movements = S4BeamlineElementMovements(f_move=1, offset_x=0, offset_y=0, offset_z=0, rotation_x=numpy.radians(angle), rotation_y=0, rotation_z=0)
    from shadow4.beamline.optical_elements.mosaic_crystals.s4_plane_mosaic_crystal import S4PlaneMosaicCrystalElement
    beamline_element = S4PlaneMosaicCrystalElement(optical_element=optical_element,coordinates=coordinates, movements=movements, input_beam=beam)

    beam, footprint = beamline_element.trace_beam()

    beamline.append_beamline_element(beamline_element)

    return beam

import numpy as np

angle = np.linspace(-0.1, 0.1, 150)
spectra = ['monochromatic', 'box', 'lorentzian']
inten = {spectrum: np.zeros_like(angle) for spectrum in spectra}

for spectrum in spectra:
    for i, angle_i in enumerate(angle):
        beam = run_beamline(angle=angle_i, spectrum=spectrum)
        inten[spectrum][i] = beam.get_intensity(nolost=1)

from srxraylib.plot.gol import plot

angle_arcmin = angle * 60.0

fig, ax = plot(angle_arcmin, inten['monochromatic'],
               angle_arcmin, inten['box'],
               angle_arcmin, inten['lorentzian'],
               xtitle="crystal tilt angle [arcmin]", ytitle="reflected intensity",
               legend=["monochromatic (8000 eV)", "box (4 eV wide)", "Lorentzian (4 eV FWHM)"],
               title="Rocking curves for different source spectra\n(mosaicity 0.05$^\\circ$ FWHM)",
               show=0)

outfile = "../figures/rocking_curves_spectra.pdf"
fig.savefig(outfile)

for spectrum in spectra:
    imax = np.argmax(inten[spectrum])
    half = inten[spectrum].max() / 2.0
    above = inten[spectrum] >= half
    idx = np.where(above)[0]
    fwhm_deg = angle[idx[-1]] - angle[idx[0]]
    print(f"{spectrum}: peak={inten[spectrum].max():.1f} at angle={angle[imax]:.4f} deg, "
          f"FWHM={fwhm_deg*60:.3f} arcmin")

print("saved:", outfile)
