import numpy as np

from shadow4_mosaics.pdf import MosaicPDF


def test_pdf_normalization():

    alpha = np.deg2rad(15)
    thetaD = alpha
    mosaic_fwhm = np.deg2rad(0.1)

    pdf = MosaicPDF(alpha, thetaD, mosaic_fwhm)

    beta = np.linspace(-6*pdf.sigma_beta,
                        6*pdf.sigma_beta,
                        20000)

    integral = np.trapezoid(pdf.pdf(beta), beta)

    assert np.isclose(integral, 1.0, rtol=1e-3)


def test_mosaicity_scaling():

    alpha = np.deg2rad(15)
    thetaD = alpha

    fwhm1 = np.deg2rad(0.1)
    fwhm2 = np.deg2rad(0.2)

    pdf1 = MosaicPDF(alpha, thetaD, fwhm1)
    pdf2 = MosaicPDF(alpha, thetaD, fwhm2)

    ratio = pdf2.sigma_beta / pdf1.sigma_beta

    # doubling mosaicity should double sigma_beta
    assert np.isclose(ratio, 2.0, rtol=1e-6)
