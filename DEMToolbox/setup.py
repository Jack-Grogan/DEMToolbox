from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'DEM Post Processing Tools Package'
LONG_DESCRIPTION = 'A package that contains tools for post processing DEM data'

# Setting up
setup(
        name="DEMPPT", 
        version=VERSION,
        author="Jack R Grogan",
        author_email="Jackrgrogan@hotmail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], 

        keywords=['python', 'dem', 'post processing', 'tools'], # add any keywords that

        classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: DEM Users",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    ],
)

