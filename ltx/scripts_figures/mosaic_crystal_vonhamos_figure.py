"""
Figure for section 7.3 ("Von Hamos spectrometer") of the paper.

Based on mosaic_crystal_vonhamos.py (same beamline: a sagittally-bent Si (111)
mosaic crystal, p = q = 10 m, radius chosen from the standard sagittal-focusing
formula R = 2 f sin(theta_B), f = (1/p+1/q)^-1 = 5 m, theta_B = 14.31 deg at
8000 eV, so R = 2.4714 m for a point-to-point focus).

Three panels:
(a) the image at the nominal q = 10 m image plane, as in the original script
    (0.2 deg FWHM mosaicity) -- reproduces the "focus not found" observation;
(b) a diagnostic: the image-plane spread in the sagittal (focusing) direction
    X, as the crystal's mosaicity is reduced at fixed geometry, showing that
    the blur scales linearly with the mosaicity and is not a geometry/coding
    defect -- see the printed diagnostics for the underlying check (the
    footprint's outgoing-direction slope dvx/dx already matches -1/q to 2%);
(c) the shift of the true focal position (the caustic waist) with respect to
    the nominal q = 10 m plane, as a function of mosaicity: the focus moves
    closer to the crystal as the mosaicity grows, and the shift vanishes as
    the mosaicity tends to zero, confirming the reading of panels (a,b).

The caustic scan of panel (c) follows the same idea as the one added in
misc/mosaic_crystal_vonhamos.py (retrace the beam to a grid of propagation
distances and locate the tightest cross section), but scans the sagittal
beam width (its intensity-weighted standard deviation, obtained by analytic
forward-projection of each ray rather than by retracing and rebinning) instead
of the peak of the central histogram bin: minimizing the width is far less
sensitive to Monte Carlo/binning noise than maximizing a single bin count,
which is essential once the shift becomes small, at low mosaicity.
"""
import numpy as np
from dabax.dabax_xraylib import DabaxXraylib
from shadow4.beamline.s4_beamline import S4Beamline
from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
from shadow4.beamline.optical_elements.mosaic_crystals.s4_sphere_mosaic_crystal import (
    S4SphereMosaicCrystal, S4SphereMosaicCrystalElement)
from syned.beamline.shape import Rectangle
from syned.beamline.element_coordinates import ElementCoordinates


def run_beamline(mosaicity_fwhm_deg=0.2, nrays=20000):
    light_source = SourceGeometrical(name='Geometrical Source', nrays=nrays, seed=5676561)
    light_source.set_spatial_type_point()
    light_source.set_depth_distribution_off()
    light_source.set_angular_distribution_uniform(hdiv1=-0.0005, hdiv2=0.0005, vdiv1=-0.0001, vdiv2=0.0001)
    light_source.set_energy_distribution_severallines(values=[8000.000000, ], unit='eV')
    light_source.set_polarization(polarization_degree=1, phase_diff=0, coherent_beam=0)
    beam = light_source.get_beam()

    boundary_shape = Rectangle(x_left=-0.1, x_right=0.1, y_bottom=-0.1, y_top=0.1)

    optical_element = S4SphereMosaicCrystal(name='Mosaic Crystal CURVED',
                                            boundary_shape=boundary_shape, material='Si',
                                            miller_index_h=1, miller_index_k=1, miller_index_l=1,
                                            thickness=1e-5,
                                            f_central=1, f_phot_cent=0, phot_cent=8000.0,
                                            material_constants_library_flag=1,
                                            dabax=None,
                                            radius=2.471446267033425, is_cylinder=1,
                                            cylinder_direction=1, convexity=1,
                                            mosaicity_fwhm_deg=mosaicity_fwhm_deg,
                                            mosaicity_profile_flag=0,
                                            )
    coordinates = ElementCoordinates(p=10, q=10, angle_radial=1.321063972, angle_azimuthal=0,
                                     angle_radial_out=1.321063972)
    beamline_element = S4SphereMosaicCrystalElement(optical_element=optical_element, coordinates=coordinates,
                                                    movements=None, input_beam=beam)

    beam, footprint = beamline_element.trace_beam()
    return beam, footprint


def find_focal_shift(beam, y_min=-10.0, y_max=10.0, npositions=400):
    """
    Caustic scan along the sagittal (focusing) direction X: analytically
    forward-projects each ray, x(y) = x0 + [(y - y0)/vy] * vx, to a grid of
    propagation distances y (measured from the nominal image plane already
    applied by trace_beam, i.e. y=0 is the nominal q), computes the
    intensity-weighted standard deviation of x at each y, and returns the y
    of minimum width (the true focal position shift), refined to sub-grid
    accuracy by a parabolic fit around the minimum.
    """
    positions = np.linspace(y_min, y_max, npositions)
    x0 = beam.get_column(1, nolost=1)
    y0 = beam.get_column(2, nolost=1)
    vx = beam.get_column(4, nolost=1)
    vy = beam.get_column(5, nolost=1)
    I = beam.get_column(23, nolost=1)

    widths = np.zeros(npositions)
    for i, yprop in enumerate(positions):
        t = (yprop - y0) / vy
        xt = x0 + t * vx
        mean = np.average(xt, weights=I)
        widths[i] = np.sqrt(np.average((xt - mean) ** 2, weights=I))

    imin = np.argmin(widths)
    if 0 < imin < npositions - 1:
        y1, y2, y3 = positions[imin - 1], positions[imin], positions[imin + 1]
        f1, f2, f3 = widths[imin - 1], widths[imin], widths[imin + 1]
        denom = f1 - 2 * f2 + f3
        if denom != 0:
            return y2 + 0.5 * (f1 - f3) / denom * (y2 - y1)
    return positions[imin]


# -----------------------------------------------------------------
# diagnostic: does the crystal geometry produce the correct focusing kick?
# -----------------------------------------------------------------
beam, footprint = run_beamline(mosaicity_fwhm_deg=0.2)

xf = footprint.get_column(1, nolost=1)
vxf = footprint.get_column(4, nolost=1)
a, b = np.polyfit(xf, vxf, 1)
print(f"footprint: outgoing-direction slope dvx/dx = {a:.6f} (ideal point focus at q=10 m: {-1.0/10.0:.6f})")

# -----------------------------------------------------------------
# panels (b,c): image-plane spread and focal-position shift vs mosaicity,
# at fixed geometry -- one beam per mosaicity value, reused for both panels
# -----------------------------------------------------------------
mosaicities = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
stds = []
shifts = []
for m in mosaicities:
    beam_m, _ = run_beamline(mosaicity_fwhm_deg=m, nrays=50000)
    X = beam_m.get_column(1, nolost=1)
    I = beam_m.get_column(23, nolost=1)
    std = np.sqrt(np.average((X - np.average(X, weights=I)) ** 2, weights=I)) * 1e3
    stds.append(std)
    shift = find_focal_shift(beam_m)
    shifts.append(shift)
    print(f"mosaicity FWHM = {m:.3f} deg -> image X std = {std:.4f} mm, focal shift = {shift:.4f} m")
mosaicities = np.array(mosaicities)
stds = np.array(stds)
shifts = np.array(shifts)

# -----------------------------------------------------------------
# plot
# -----------------------------------------------------------------
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 5))

ticket = beam.histo2(1, 3, nbins_h=100, nbins_v=100, xrange=[-0.05, 0.05], yrange=[-0.004, 0.004],
                     nolost=1, ref=23)
im = ax1.pcolormesh(ticket['bin_h_edges'], ticket['bin_v_edges'], ticket['histogram'].T, cmap='jet')
ax1.set_xlabel("$X$ [m] (sagittal / focusing plane)")
ax1.set_ylabel("$Z$ [m] (dispersion plane)")
ax1.set_title(f"(a) image at $q=10$ m, mosaicity $0.2^\\circ$ FWHM\n"
              f"I={ticket['intensity']:.0f}, FWHM$_X$={ticket['fwhm_h']*1e3:.2f} mm")
plt.colorbar(im, ax=ax1)

ax2.loglog(mosaicities, stds, 'o-')
ax2.loglog(mosaicities, stds[0] * (mosaicities / mosaicities[0]), '--', color='gray',
          label="linear scaling")
ax2.set_xlabel("mosaicity FWHM [deg]")
ax2.set_ylabel("image-plane $X$ std [mm]")
ax2.set_title("(b) image blur vs mosaicity\n(geometry, $p=q=10$ m, fixed)")
ax2.legend(fontsize=9)
ax2.grid(True, which='both', alpha=0.3)

ax3.semilogx(mosaicities, shifts, 'o-')
ax3.set_xlabel("mosaicity FWHM [deg]")
ax3.set_ylabel("focal position shift [m]\n(true focus $-$ nominal $q=10$ m)")
ax3.set_title("(c) focal position shift vs mosaicity\n(geometry, $p=q=10$ m, fixed)")
ax3.grid(True, which='both', alpha=0.3)
ax3.axhline(0.0, color='gray', lw=0.8)

fig.tight_layout()

outfile = "../figures/mosaic_crystal_vonhamos.pdf"
fig.savefig(outfile)
print("saved:", outfile)
