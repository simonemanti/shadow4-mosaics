
#
# Memorandum: 
#
# Install from sources: 
#     git clone https://github.com/simonemanti/shadow4-mosaics
#     cd shadow4-mosaics
#     python -m pip install -e . --no-deps --no-binary :all:
#
# Upload to pypi (when uploading, increment the version number):
#     python setup.py register (only once, not longer needed)
#     python setup.py sdist
#     python -m twine upload dist/...
#          
# Install from pypi:
#     pip install shadow4-hybrid
#


from setuptools import setup

PACKAGES = [
    "shadow4_mosaics",
]

INSTALL_REQUIRES = (
    'setuptools',
    'numpy',
    'scipy',
    'syned>=1.0.47',
    'srxraylib>=1.0.63',
    'crystalpy',
    'shadow4>=0.1.68',
)

setup(name='shadow4-mosaics',
      version='0.0.1',
      description='mosaic crystals for shadow4 in python',
      author='Simone Manti, Manuel Sanchez del Rio',
      author_email='srio@esrf.fr',
      url='https://github.com/simonemanti/shadow4-mosaics/',
      packages=PACKAGES,
      install_requires=INSTALL_REQUIRES,
     )

