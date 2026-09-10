"""
Diffraction profiles (reflectivity vs angle and vs energy) of a mosaic HOPG
crystal, 0.2 deg FWHM mosaicity, 1 mm thick, at 8 keV.

Adapted from mosaic_gpt.py: same secondary-extinction reflectivity formula
(paper eq. eq:reflectivity / eq:a_and_A), restricted to a Gaussian mosaic
distribution, but with Q_s, Q_p, mu, d-spacing and the Bragg angle computed
from real HOPG structure factors via shadow4_mosaics.crystal.MosaicCrystal
instead of hardcoded numbers.

Produces ltx/figures/diffraction_profiles_HOPG.pdf, used in the paper
(section "X-ray reflectivity of mosaic crystals").
"""
import numpy as np
import matplotlib.pyplot as plt

from shadow4_mosaics.crystal import MosaicCrystal


# -----------------------------
# Gaussian mosaic distribution w(Delta), sigma = kappa [rad]
# -----------------------------
def w_gaussian(Delta, kappa):
    return (1.0 / (kappa * np.sqrt(2.0 * np.pi))) * np.exp(-Delta**2 / (2.0 * kappa**2))


# -----------------------------
# eq:reflectivity : r(Delta) = a / [ (1+a) + sqrt(1+2a) coth(A sqrt(1+2a)) ]
# a = w(Delta) Q / mu ,  A = mu t / sin(thetaB)
# -----------------------------
def reflectivity(Delta, Q, mu, t, thetaB, kappa):
    A = mu * t / np.sin(thetaB)
    a = w_gaussian(Delta, kappa) * Q / mu
    s = np.sqrt(1.0 + 2.0 * a)
    return a / (1.0 + a + s / np.tanh(A * s))


# -----------------------------
# Crystal: HOPG (0 0 2), 0.2 deg FWHM mosaicity, 1 mm thick
# -----------------------------
crystal = MosaicCrystal(
    material="Graphite",
    hkl=(0, 0, 2),
    thickness_cm=0.1,          # 1 mm
    mosaicity_fwhm_deg=0.2,
)

E0 = 8000.0  # eV
t = crystal.thickness_cm
kappa = np.deg2rad(crystal.mosaicity_fwhm_deg) / (2.0 * np.sqrt(2.0 * np.log(2.0)))

# angular half-range of the profile (both panels cover the same Delta window)
angle_half_range_deg = 1.0
Delta = np.deg2rad(np.linspace(-angle_half_range_deg, angle_half_range_deg, 2001))

# -----------------------------
# Panel (a): reflectivity vs angle, at fixed energy E0
# -----------------------------
thetaB0 = crystal.bragg_angle(E0)
Qs0 = crystal.Q_s_cm_inv(E0)
Qp0 = crystal.Q_p_cm_inv(E0)
mu0 = crystal.mu_cm_inv(E0)

r_s_angle = reflectivity(Delta, Qs0, mu0, t, thetaB0, kappa)
r_p_angle = reflectivity(Delta, Qp0, mu0, t, thetaB0, kappa)

# -----------------------------
# Panel (b): reflectivity vs energy, at fixed incidence angle = thetaB(E0)
# The energy range is set so that Delta(E) spans the same angular window
# as panel (a), via the local dispersion dtheta/dE evaluated at E0.
# -----------------------------
dtheta_dE = crystal.bragg_angle(E0 + 1.0) - thetaB0  # rad / eV
E_half_range = np.deg2rad(angle_half_range_deg) / abs(dtheta_dE)
energy = np.linspace(E0 - E_half_range, E0 + E_half_range, 2001)

thetaB_E = crystal.bragg_angle(energy)
Qs_E = crystal.Q_s_cm_inv(energy)
Qp_E = crystal.Q_p_cm_inv(energy)
mu_E = crystal.mu_cm_inv(energy)
Delta_E = thetaB0 - thetaB_E

r_s_energy = reflectivity(Delta_E, Qs_E, mu_E, t, thetaB_E, kappa)
r_p_energy = reflectivity(Delta_E, Qp_E, mu_E, t, thetaB_E, kappa)

# -----------------------------
# Plot
# -----------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.4, 6.4))

ax1.plot(np.rad2deg(Delta) * 60.0, r_s_angle, lw=1.8, label=r"$\sigma$ (s)")
ax1.plot(np.rad2deg(Delta) * 60.0, r_p_angle, lw=1.8, ls="--", label=r"$\pi$ (p)")
ax1.set_xlabel(r"$\Delta$ [arcmin]")
ax1.set_ylabel("Reflectivity")
ax1.set_title(f"vs angle, $E$ = {E0/1000:.0f} keV")
ax1.grid(True, alpha=0.4)
ax1.legend(fontsize=9)

ax2.plot(energy - E0, r_s_energy, lw=1.8, label=r"$\sigma$ (s)")
ax2.plot(energy - E0, r_p_energy, lw=1.8, ls="--", label=r"$\pi$ (p)")
ax2.set_xlabel(r"$E - E_0$ [eV]")
ax2.set_ylabel("Reflectivity")
ax2.set_title(fr"vs energy, $\theta$ = $\theta_B$({E0/1000:.0f} keV)")
ax2.grid(True, alpha=0.4)
ax2.legend(fontsize=9)

fig.suptitle("HOPG (002), 0.2$^\\circ$ FWHM mosaicity, 1 mm thick")
fig.tight_layout()

outfile = "../figures/diffraction_profiles_HOPG.pdf"
fig.savefig(outfile)
def scalar(x):
    return np.asarray(x).ravel()[0]


print(f"theta_B(E0) = {scalar(np.rad2deg(thetaB0)):.4f} deg")
print(f"Q_s(E0) = {scalar(Qs0):.5f} cm^-1, Q_p(E0) = {scalar(Qp0):.5f} cm^-1, mu(E0) = {scalar(mu0):.5f} cm^-1")
print(f"kappa (sigma) = {np.rad2deg(kappa)*60:.3f} arcmin")
print(f"energy half-range = {scalar(E_half_range):.1f} eV")
print(f"saved: {outfile}")

plt.show()
