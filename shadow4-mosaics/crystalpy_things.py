import numpy
import scipy.constants as codata

from crystalpy.diffraction.GeometryType import BraggDiffraction
from crystalpy.diffraction.DiffractionSetupDabax import DiffractionSetupDabax

#
# main
#
if __name__ == "__main__":

    a = DiffractionSetupDabax(geometry_type        = BraggDiffraction(),   # GeometryType object
                            crystal_name           = "Graphite", # string
                            thickness              = 1e-3,       # meters
                            miller_h               = 0,          # int
                            miller_k               = 0,          # int
                            miller_l               = 2,          # int
                            asymmetry_angle        = 0,          # radians
                            azimuthal_angle        = 0.0,
                            dabax                  = None)

    print(a)

    energy = 8000.0
    energies = numpy.linspace(energy, energy + 100, 2)
    print("Photon energy: %g deg " % (energy))
    print("d_spacing: %g A " % (a.dSpacing()))
    print("unitCellVolumw: %g A**3 " % (a.unitcellVolume()))
    print("Bragg angle: %g deg " % (numpy.degrees(a.angleBragg(energy))))
    print("Bragg angle [array] [deg] ", numpy.degrees(a.angleBragg(energies)))
    print("Asymmerey factor b: ", a.asymmetryFactor(energy))

    print("F0 ", a.F0(energy))
    print("F0 [array] ", a.F0(energies))
    print("FH ", a.FH(energy))
    print("FH [array] ", a.FH(energies))
    print("FH_bar ", a.FH_bar(energy))
    print("FH_bar [array] ", a.FH_bar(energies))

    print("PSI0 ", a.psi0(energy))
    print("PSIH ", a.psiH(energy))
    print("PSIH_bar ", a.psiH_bar(energy))
    #
    print("V0: ", a.vectorK0direction(energy).components())
    print("Bh direction: ", a.vectorHdirection().components())
    print("Bh: ", a.vectorH().components())
    print("K0: ", a.vectorK0(energy).components())
    print("Kh: ", a.vectorKh(energy).components())
    print("Vh: ", a.vectorKhdirection(energy).components())
    #
    #
    from crystalpy.util.Photon import Photon

    print("Difference to ThetaB uncorrected: ",
          a.deviationOfIncomingPhoton(Photon(energy_in_ev=energy, direction_vector=a.vectorK0(energy))))
    #
    #
    print("Asymmerey factor b: ", a.asymmetryFactor(energy))
    print("Bragg angle: %g deg " % (a.angleBragg(energy) * 180 / numpy.pi))

    wavelength = codata.h * codata.c / codata.e / energy

    mu = -2 * numpy.pi / wavelength * a.psi0(energy).imag # eq 3.172 in Zachariasen
    print("mu [m^-1]: ", mu)
    print("mu [cm^-1]: ", mu / 100)

    Q_mos = numpy.pi ** 2 * numpy.abs(a.psiH(energy) * a.psiH_bar(energy)) / wavelength / numpy.sin(2 * a.angleBragg(energy))
    print("Q [m^-1]: ", Q_mos)
    print("Q [cm^-1]: ", Q_mos / 100)


    # thetaB = numpy.deg2rad(13.38257103)
    # eta = numpy.deg2rad(0.1)  # mosaicity, degrees
    # Q = 0.16693157449312368 #        cm^-1
    # mu = 9.6590899717931258 #     cm^-1
    # t0 = 0.1   # cm