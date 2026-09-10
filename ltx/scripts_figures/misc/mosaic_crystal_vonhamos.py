import numpy as np


def run_beamline(mosaicity_fwhm_deg=0.2):
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
    light_source.set_energy_distribution_severallines(values=[8000.000000, ], unit='eV')
    light_source.set_polarization(polarization_degree=1, phase_diff=0, coherent_beam=0)
    beam = light_source.get_beam()

    beamline.set_light_source(light_source)

    # optical element number XX
    from syned.beamline.shape import Rectangle
    boundary_shape = Rectangle(x_left=-0.1, x_right=0.1, y_bottom=-0.1, y_top=0.1)

    from shadow4.beamline.optical_elements.mosaic_crystals.s4_sphere_mosaic_crystal import S4SphereMosaicCrystal
    optical_element = S4SphereMosaicCrystal(name='Mosaic Crystal  CURVED',
                                            boundary_shape=boundary_shape, material='Si',
                                            miller_index_h=1, miller_index_k=1, miller_index_l=1,
                                            thickness=1e-05,
                                            f_central=1, f_phot_cent=0, phot_cent=8000.0,
                                            file_refl='HOPG2_20_v2.002',
                                            material_constants_library_flag=3,
                                            # 0=xraylib,1=dabax,2=preprocessor v1,3=preprocessor v2
                                            dabax=None,  # used when material_constants_library_flag=1,
                                            radius=2.471446267033425, is_cylinder=1,
                                            cylinder_direction=1, convexity=1,
                                            mosaicity_fwhm_deg=mosaicity_fwhm_deg,
                                            mosaicity_profile_flag=0,  # 0=Gaussian, 1=External
                                            )
    from syned.beamline.element_coordinates import ElementCoordinates
    coordinates = ElementCoordinates(p=10, q=10, angle_radial=1.321063972, angle_azimuthal=0,
                                     angle_radial_out=1.321063972)
    movements = None
    from shadow4.beamline.optical_elements.mosaic_crystals.s4_sphere_mosaic_crystal import S4SphereMosaicCrystalElement
    beamline_element = S4SphereMosaicCrystalElement(optical_element=optical_element, coordinates=coordinates,
                                                    movements=movements, input_beam=beam)

    beam, footprint = beamline_element.trace_beam()

    beamline.append_beamline_element(beamline_element)
    return beam, footprint


#
# main
#
from srxraylib.plot.gol import plot, plot_image, plot_image_with_histograms, plot_show

# WARNING: NO incremental result allowed!!"
beam, footprint = run_beamline(mosaicity_fwhm_deg=0.2)

ticket = beam.histo2(1, 3, nbins_h=100, nbins_v=100, xrange=[-0.05, 0.05], yrange=[-0.004, 0.004], nolost=1, ref=23)

title = "I: %.1f " % ticket['intensity']
if ticket['fwhm_h'] is not None: title += "FWHM H: %f " % ticket['fwhm_h']
if ticket['fwhm_v'] is not None: title += "FWHM V: %f " % ticket['fwhm_v']

plot_image_with_histograms(ticket['histogram'], ticket['bin_h_center'], ticket['bin_v_center'],
                           title=title, xtitle="column 1", ytitle="column 3",
                           cmap='jet', add_colorbar=True, figsize=(8, 8), histo_path_flag=1, show=1)

#
# study the focal position (caustic)
#


#
# main
#
import numpy
from srxraylib.plot.gol import plot, plot_image, plot_show

beam_to_analyze = beam


y_min, y_max, npositions = -10, 10, 300
x_min, x_max, npoints_x = -0.2, 0.2, 300
nolost = 1

positions = numpy.linspace(y_min, y_max, npositions)
out_x = numpy.zeros((npoints_x, npositions))
fwhm = numpy.zeros(npositions)
center = numpy.zeros(npositions)
col = 1
ref = 23
col_title = "X (col 1)"

for i in range(npositions):
    beami = beam_to_analyze.duplicate()
    beami.retrace(positions[i], resetY=True)
    tkt_x = beami.histo1(col, xrange=[x_min, x_max], nbins=npoints_x, nolost=nolost, ref=ref)
    out_x[:, i] = tkt_x['histogram']
    fwhm[i] = tkt_x['fwhm']
    if ref == 23:
        center[i] = numpy.average(beami.get_column(col, nolost=nolost), weights=beami.get_column(23, nolost=nolost))
    else:
        center[i] = numpy.average(beami.get_column(col, nolost=nolost))
#
# plots
#
print('Result arrays X,Y (shapes): ', out_x.shape, tkt_x['bin_center'].shape, positions.shape)
x = tkt_x['bin_center']
y = positions

# 2D
plot_image(out_x.T, y, 1e6 * x,
           title="", ytitle="%s [um] (%d pixels)" % (col_title, x.size),
           xtitle="Y [m] (%d pixels)" % (y.size), aspect="auto" )
# # FWHM
# fwhm[fwhm == 0] = "nan"
# plot(y, 1e6 * fwhm, title="FWHM",
#      xtitle="y [m]", ytitle="FHWH [um]", marker=".")
# I0
nx, ny = out_x.shape
I0 = out_x.T[:, nx // 2]
plot(y, I0, title="I at central profile", xtitle="y [m]", ytitle="I0", marker=".")
# # center
# plot(y, 1e6 * center, title="CENTER",
#      xtitle="y [m]", ytitle="CENTER [um]", marker=".", yrange=[1e6 * x_min, 1e6 * x_max])

print("Focal position found at %2.3f m from the expected position (10 m downstream from the crystal" % (y[numpy.argmax(I0)]))