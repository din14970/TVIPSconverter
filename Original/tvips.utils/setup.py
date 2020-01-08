# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
#with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
#    long_description = f.read()


setup(
    name='tvips',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.6.8',
    #version_format='{tag}.dev{commitcount}+{gitsha}',
    setup_requires=['setuptools-git-version'],

    description='Tools for TVIPS products',
#    long_description=long_description,

    # The project's main homepage.
    url='http://www.tvips.com',

    # Author details
    author='Marco Oster/TVIPS 2018',
    author_email='support@tvips.com',

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development',

        # Pick your license as you wish (should match "license" above)
        #'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',

    ],

    # What does your project relate to?
    keywords='tvips tietz camera microscopy electron',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=['tvips', 'tvips.utils'],
    install_requires=['mrcfile', 'tifffile', 'numpy'],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'burst2mrcstack=tvips.utils.burst2mrcstack:main',
            'recorder=tvips.utils.recorder:main',
            'extract_vbf=tvips.utils.extract_blovbf:main',
        ],
    },
    
)
