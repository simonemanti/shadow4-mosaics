import numpy as np
import os
import glob
import pytest

from shadow4_mosaics.crystal import MosaicCrystal


@pytest.fixture
def crystal():
    return MosaicCrystal()


def test_bragg_law(crystal):

    energy = 8000.0
    hc = 12398.4193

    d = crystal.d_spacing_A
    theta = crystal.bragg_angle(energy)
    wavelength = hc / energy

    assert np.isclose(2 * d * np.sin(theta), wavelength, rtol=1e-6)


def test_bragg_vectorization(crystal):

    energies = np.linspace(7900, 8100, 5)
    thetas = crystal.bragg_angle(energies)

    assert thetas.shape == energies.shape
    assert np.all(np.isfinite(thetas))


def test_volume_reconstruction(crystal):

    d = crystal.d_spacing_A
    c = 2.0 * d
    a = 2.46

    V_reconstructed = np.sqrt(3)/2 * a**2 * c
    V_dabax = crystal.unitCellVolume_A3

    assert np.isclose(V_dabax, V_reconstructed, rtol=0.02)


def test_asymmetry_factor(crystal):

    energy = 8000.0
    asym = crystal.asymmetry_factor(energy)

    assert np.isclose(asym, -1.0, rtol=1e-6)

def test_psi_factors(crystal):

    energies = np.linspace(7900, 8100, 5)

    psi0 = crystal.psi0(energies)
    psiH = crystal.psiH(energies)
    psiHb = crystal.psiH_bar(energies)

    assert psi0.shape == energies.shape
    assert psiH.shape == energies.shape
    assert psiHb.shape == energies.shape

    assert np.all(np.isfinite(psi0))
    assert np.all(np.isfinite(psiH))
    assert np.all(np.isfinite(psiHb))

def test_structure_factors(crystal):

    energies = np.linspace(7900, 8100, 5)

    F0 = crystal.F0(energies)
    FH = crystal.FH(energies)
    FHb = crystal.FH_bar(energies)

    assert F0.shape == energies.shape
    assert FH.shape == energies.shape
    assert FHb.shape == energies.shape

    assert np.all(np.isfinite(F0))
    assert np.all(np.isfinite(FH))
    assert np.all(np.isfinite(FHb))


def teardown_module(module):

    for f in glob.glob("*.dat"):
        try:
            os.remove(f)
        except Exception:
            pass