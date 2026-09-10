"""
Lorentzian photon-energy distribution for a SHADOW4 SourceGeometrical.

SourceGeometrical has built-in energy distributions for a single line, several
lines, a uniform box, a Gaussian and an arbitrary user-defined spectrum
(set_energy_distribution_userdefined), but no Lorentzian option. This module
builds a Lorentzian (Cauchy) spectrum as a tabulated (abscissas, ordinates)
pair and feeds it through set_energy_distribution_userdefined, which samples
it by numerical inversion of the cumulative distribution (srxraylib Sampler1D).
"""
import numpy as np


def lorentzian_spectrum(center=8000.0, fwhm=4.0, half_range_in_fwhm=100, n_points=4001):
    """
    Tabulated Lorentzian (Cauchy) spectral density.

    Parameters
    ----------
    center : float
        Peak photon energy [eV].
    fwhm : float
        Full width at half maximum [eV].
    half_range_in_fwhm : float
        Half-width of the tabulated grid, in units of fwhm. The Lorentzian has
        heavy tails, so this must be large enough that the truncation does not
        bias the sampling over the angular/energy range of interest.
    n_points : int
        Number of points of the tabulated grid.

    Returns
    -------
    tuple
        (energies, ordinates), numpy arrays of shape (n_points,).
    """
    gamma = fwhm / 2.0
    half_range = half_range_in_fwhm * fwhm
    energies = np.linspace(center - half_range, center + half_range, n_points)
    ordinates = (1.0 / np.pi) * gamma / ((energies - center) ** 2 + gamma ** 2)
    return energies, ordinates


def set_energy_distribution_lorentzian(light_source, center=8000.0, fwhm=4.0,
                                        half_range_in_fwhm=100, n_points=4001):
    """
    Configures a SourceGeometrical light source with a Lorentzian photon-energy
    distribution.

    Parameters
    ----------
    light_source : instance of SourceGeometrical
    center, fwhm, half_range_in_fwhm, n_points : see lorentzian_spectrum()

    Returns
    -------
    instance of SourceGeometrical
        The same light_source, for chaining.
    """
    energies, ordinates = lorentzian_spectrum(center=center, fwhm=fwhm,
                                               half_range_in_fwhm=half_range_in_fwhm,
                                               n_points=n_points)
    light_source.set_energy_distribution_userdefined(energies, ordinates, unit='eV')
    return light_source


if __name__ == "__main__":
    from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical

    light_source = SourceGeometrical(name='check', nrays=200000, seed=123)
    light_source.set_spatial_type_point()
    light_source.set_depth_distribution_off()
    light_source.set_angular_distribution_uniform(hdiv1=0, hdiv2=0, vdiv1=0, vdiv2=0)
    set_energy_distribution_lorentzian(light_source, center=8000.0, fwhm=4.0)
    beam = light_source.get_beam()

    E = beam.get_column(11) if False else beam.get_photon_energy_eV()

    import numpy as np
    print("sampled mean:", np.mean(E), "expected center: 8000.0")
    fwhm_check = np.percentile(np.abs(E - 8000.0), 50) * 2  # rough median-based check
    print("rough FWHM check (2 * median |E-E0|, exact only for Lorentzian):", fwhm_check, "expected 4.0")
