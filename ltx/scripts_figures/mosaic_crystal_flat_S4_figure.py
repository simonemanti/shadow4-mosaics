"""
Figure for section 7.1 ("Plane mosaic crystals") of the paper.

Copied from mosaic_crystal_flat_S4.py (same beamline, unchanged): a plane HOPG
(002) mosaic crystal, 0.2 deg FWHM mosaicity, 1 mm thick, in the symmetric
1:1 parafocusing geometry (p = q = 10 m) of Fig. 1 (bottom), illuminated by a
point source with three closely spaced energy lines (7996, 8000, 8004 eV).
Only the final plotting step is changed: the figure is saved to
ltx/figures/mosaic_crystal_flat_S4_dispersion.pdf instead of being shown
interactively.
"""
import numpy as np


def run_beamline():
    footprint = None
    import numpy as np
    from dabax.dabax_xraylib import DabaxXraylib
    from shadow4.beamline.s4_beamline import S4Beamline

    beamline = S4Beamline()

    #
    #
    #
    from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
    light_source = SourceGeometrical(name='Geometrical Source', nrays=5000, seed=5676561)
    light_source.set_spatial_type_point()
    light_source.set_depth_distribution_off()
    light_source.set_angular_distribution_uniform(hdiv1=-0.0005, hdiv2=0.0005, vdiv1=-0.0001, vdiv2=0.0001)
    light_source.set_energy_distribution_severallines(values=[7996.000000, 8000.000000, 8004.000000, ], unit='eV')
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
                                           mosaicity_fwhm_deg=0.2,
                                           mosaicity_profile_flag=0,  # 0=Gaussian, 1=Lorentzian
                                           )
    from syned.beamline.element_coordinates import ElementCoordinates
    coordinates = ElementCoordinates(p=10, q=10, angle_radial=1.337226398, angle_azimuthal=0,
                                     angle_radial_out=1.337226398)
    movements = None
    from shadow4.beamline.optical_elements.mosaic_crystals.s4_plane_mosaic_crystal import S4PlaneMosaicCrystalElement
    beamline_element = S4PlaneMosaicCrystalElement(optical_element=optical_element, coordinates=coordinates,
                                                   movements=movements, input_beam=beam)

    beam, footprint = beamline_element.trace_beam()

    beamline.append_beamline_element(beamline_element)
    return beam, footprint


#
# main
#
from srxraylib.plot.gol import plot_image_with_histograms

beam, footprint = run_beamline()

ticket = beam.histo2(1, 3, nbins_h=100, nbins_v=100, xrange=[-0.05, 0.05], yrange=[-0.004, 0.004], nolost=1, ref=23)

title = "I: %.1f " % ticket['intensity']
if ticket['fwhm_h'] is not None: title += "FWHM H: %f " % ticket['fwhm_h']
if ticket['fwhm_v'] is not None: title += "FWHM V: %f " % ticket['fwhm_v']

print(title)
print("nrays (good, lost):", beam.get_number_of_rays(nolost=1), beam.get_number_of_rays(nolost=2))

fig, axScatter, axHistx, axHisty = plot_image_with_histograms(
    ticket['histogram'], ticket['bin_h_center'], ticket['bin_v_center'],
    title=title, xtitle=r"$X$ [m] (perpendicular plane)",
    ytitle=r"$Z$ [m] (dispersion / diffraction plane)",
    cmap='jet', add_colorbar=True, figsize=(8, 8), histo_path_flag=1, show=0)

outfile = "../figures/mosaic_crystal_flat_S4_dispersion.pdf"
fig.savefig(outfile)
print("saved:", outfile)
