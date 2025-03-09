import setuptools
import os

with open("README.md", "r") as f:
    long_description = f.read()

path = os.path.abspath(os.path.dirname(__file__))
about = {}
with open(os.path.join(path, "DEMToolbox", "__version__.py")) as f:
    exec(f.read(), about)

def requirements(filename):
    # The dependencies are the same as the contents of requirements.txt
    with open(filename) as f:
        return [line.strip() for line in f if line.strip()]


# What packages are required for this module to be executed?
required = requirements("requirements.txt")

DESCRIPTION = 'DEM Post Processing Tools Package'

# Setting up
setuptools.setup(
        name="DEMPPT", 
        version=about["__version__"],
        author="Jack R Grogan",
        author_email="Jackrgrogan@hotmail.com",
        description='DEM Post Processing Tools Package',
        long_description=long_description,
        packages=setuptools.find_packages(),
        install_requires=[], 
        keywords=['python', 'dem', 'post processing', 'tools'],
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: DEM Users",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX :: Linux",
        ],
)

