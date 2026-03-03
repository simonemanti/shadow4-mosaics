import numpy as np


class MosaicPDF:

    def __init__(self, alpha, thetaD, mosaic_fwhm):

        self.alpha = alpha
        self.thetaD = thetaD

        self.sigma_phi = mosaic_fwhm / (2*np.sqrt(2*np.log(2)))

        delta = alpha - thetaD
        C0 = np.cos(delta)

        if np.isclose(delta, 0.0):
            s1 = np.sin(thetaD)*np.sin(alpha)
        else:
            numerator = np.sin(thetaD)*np.sin(alpha)*np.arccos(C0)
            denominator = np.sqrt(1 - C0**2)
            s1 = numerator/denominator

        self.sigma_beta = self.sigma_phi / np.sqrt(s1)


    def pdf(self, beta):

        return (1/(np.sqrt(2*np.pi)*self.sigma_beta)) * \
               np.exp(-beta**2/(2*self.sigma_beta**2))


    def sample_beta(self, size=None):

        return np.random.normal(0.0, self.sigma_beta, size=size)
