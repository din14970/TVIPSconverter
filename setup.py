from setuptools import setup, find_packages

with open("README.md") as f:
    readme = f.read()

setup(
    name="tvipsconverter",
    version="0.0.6",
    description=("GUI converter for data from TVIPS cameras into .blo mainly"
                 " for orientation mapping (PED) or 4D STEM."),
    url='https://github.com/din14970/TVIPSconverter',
    author='Niels Cautaerts',
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
        'numpy==1.17.4',
        'matplotlib==3.1.1',
        'scipy>=1.1.0',
        'tifffile==2019.7.26.2',
        'Pillow==6.2.1',
        'PyQt5>=5.13.2',
        'h5py>=2.10.0'
    ],
)
