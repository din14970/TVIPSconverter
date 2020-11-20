from setuptools import setup, find_packages

with open("README.md") as f:
    readme = f.read()

setup(
    name="tvipsconverter",
    version="0.1.2",
    description=(
        "GUI converter for movie data from TVIPS cameras"
        "into other formats like .blo, .tiff and .hspy."
        "Mainly for orientation mapping (PED) or 4D STEM experiments."),
    url='https://github.com/din14970/TVIPSconverter',
    author='Niels Cautaerts, Paddy Harrison',
    author_email='nielscautaerts@hotmail.com',
    license='GPL-3.0',
    long_description=readme,
    long_description_content_type="text/markdown",
    classifiers=['Topic :: Scientific/Engineering :: Physics',
                 'Intended Audience :: Science/Research',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python :: 3.8'],
    keywords='TEM',
    packages=find_packages(exclude=["*tests*", "*examples*", "garbage"]),
    entry_points={
          'console_scripts': [
              'tvipsconverter = tvipsconverter.widgets:main',
          ],
      },
    package_data={'': ['tvipsconverter/widget_2.ui']},
    include_package_data=True,
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy>=1.1.0',
        'tifffile',
        'Pillow',
        'PyQt5>=5.13.2',
        'h5py>=2.10.0',
        'scikit-image>=0.17.2',
    ],
)
