from setuptools import setup, find_packages

with open("README.md") as f:
    readme = f.read()

setup(
    name="tvipsconverter",
    version="0.0.2",
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
    packages=find_packages("tvipsconverter",
                           exclude=["*tests*", "*examples*"]),
    entry_points={
          'console_scripts': [
              'tvipsconverter = tvipsconverter.widgets:main',
          ],
      },
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'tifffile',
        'Pillow',
        'PyQt5'
    ],
)
