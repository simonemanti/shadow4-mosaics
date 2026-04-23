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

def test_deviation(crystal):

    energy = 8000.0

    dev = crystal.deviation_of_incoming_photon(energy)

    assert np.isfinite(dev)
    assert abs(dev) < 1e-6

def test_mu_Q_cm_units(crystal):

    energy = 8000.0

    mu = crystal.mu_cm_inv(energy)
    Qs = crystal.Q_s_cm_inv(energy)
    Qp = crystal.Q_p_cm_inv(energy)
    Q = crystal.Q_cm_inv(energy)
    theta = crystal.bragg_angle(energy)

    assert np.isfinite(mu)
    assert np.isfinite(Qs)
    assert np.isfinite(Qp)
    assert np.isfinite(Q)
    assert np.isclose(Q, Qs)
    assert np.isclose(Qp, Qs * np.cos(2 * theta) ** 2)
    assert Qp <= Qs

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

def test_wavevector_geometry(crystal):

    energy = 8000.0

    V0 = crystal.vectorK0direction(energy).components()
    Bh_dir = crystal.vectorHdirection().components()
    Bh = crystal.vectorH().components()
    K0 = crystal.vectorK0(energy).components()
    Kh = crystal.vectorKh(energy).components()
    Vh = crystal.vectorKhdirection(energy).components()

    V0 = np.array(V0)
    Bh_dir = np.array(Bh_dir)
    Bh = np.array(Bh)
    K0 = np.array(K0)
    Kh = np.array(Kh)
    Vh = np.array(Vh)

    assert V0.shape == (3,)
    assert Bh_dir.shape == (3,)
    assert Bh.shape == (3,)
    assert K0.shape == (3,)
    assert Kh.shape == (3,)
    assert Vh.shape == (3,)

    assert np.all(np.isfinite(V0))
    assert np.all(np.isfinite(Bh))
    assert np.all(np.isfinite(K0))
    assert np.all(np.isfinite(Kh))

def teardown_module(module):

    for f in glob.glob("*.dat"):
        try:
            os.remove(f)
        except Exception:
            pass
