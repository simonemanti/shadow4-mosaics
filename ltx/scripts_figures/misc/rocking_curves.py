"""
Rocking curves (reflected intensity vs crystal tilt angle) of a plane mosaic
HOPG (002) crystal, for three different source photon-energy spectra centred
at 8000 eV: monochromatic, a 4 eV wide box, and a 4 eV FWHM Lorentzian. The
crystal is tilted rigidly (S4BeamlineElementMovements rotation_x) around its
nominal Bragg-angle setting, and the same beamline is retraced at each tilt.

The box and Lorentzian widths (4 eV) are chosen at the top of the 2-4 eV range
suggested for this comparison, to make the effect of the source bandwidth on
the rocking curve as visible as possible.
"""
import numpy


def run_beamline(angle=0.0, spectrum='monochromatic', nrays=5000, mosaicity_fwhm_deg=0.2):
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


    # test plot
    if 0:
       from srxraylib.plot.gol import plot_scatter
       plot_scatter(beam.get_photon_energy_eV(nolost=1), beam.get_column(23, nolost=1), title='(Intensity,Photon Energy)', plot_histograms=0)
       plot_scatter(1e6 * beam.get_column(1, nolost=1), 1e6 * beam.get_column(3, nolost=1), title='(X,Z) in microns')

    return beam

import numpy as np

mosaicity_fwhm_deg=0.2
angle = np.linspace(-0.4, 0.4, 10)
spectra = ['monochromatic', 'box', 'lorentzian']
inten = {spectrum: np.zeros_like(angle) for spectrum in spectra}

for spectrum in spectra:
    for i, angle_i in enumerate(angle):
        beam = run_beamline(angle=angle_i, spectrum=spectrum, mosaicity_fwhm_deg=mosaicity_fwhm_deg)
        inten[spectrum][i] = beam.get_intensity(nolost=1)

from srxraylib.plot.gol import plot

plot(angle, inten['monochromatic'],
     angle, inten['box'],
     angle, inten['lorentzian'],
     xtitle="crystal tilt angle [deg]", ytitle="reflected intensity",
     legend=["monochromatic (8000 eV)", "box (4 eV wide)", "Lorentzian (4 eV FWHM)"],
     title="Rocking curves for different source spectra",
     show=1)
