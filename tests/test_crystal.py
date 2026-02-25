import numpy as np
import os
import glob

from shadow4_mosaics.crystal import MosaicCrystal


def test_bragg_law_and_cleanup():

    energy = 8000.0  # eV
    hc = 12398.4193  # eV·Å

    crystal = MosaicCrystal()

    d = crystal.d_spacing_A
    theta = crystal.bragg_angle(energy)

    wavelength = hc / energy

    # --- physics consistency ---
    assert np.isclose(2 * d * np.sin(theta), wavelength, rtol=1e-6)

    # --- vectorization check ---
    energies = np.linspace(7900, 8100, 5)
    thetas = crystal.bragg_angle(energies)
    assert thetas.shape == energies.shape

    # --- cleanup dabax files ---
    for f in glob.glob("*.dat"):
        try:
            os.remove(f)
        except Exception:
            pass
