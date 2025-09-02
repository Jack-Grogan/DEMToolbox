![logo](https://github.com/Jack-Grogan/DEMToolbox/blob/main/docs/images/logo.png) 

# DEMToolbox
## Post Processing Tools for Analysis of DEM Simulations

DEMToolbox provides a range of post processing tools for analysing DEM 
simulations. Performance optimisations have tried to be attained where possible
through use of [numpy](https://numpy.org/) whos core is written in optimised 
C code. Often users will want to apply this libraries functionality to many 
simulation output files. This problem is "embarrassingly parrallel" and can be sped up using `ProcesssPoolExecutor` from [concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html).

## Getting Started

To install locally run:

```zsh
pip install git+https://github.com/Jack-Grogan/DEMToolbox/
```

If running code on a HPC add this code into your batch run script prior to script execution:

```bash
export VENV_DIR="${HOME}/virtual-environments"
export VENV_PATH="${VENV_DIR}/DEMToolbox"

# Create a master venv directory if necessary
mkdir -p ${VENV_DIR}

# Check if virtual environment exists and create it if not
if [[ ! -d ${VENV_PATH}  ]]; then
	python -m venv --system-site-packages ${VENV_PATH}
fi

# Activate the virtual environment
source ${VENV_PATH}/bin/activate

python -m pip install --upgrade pip
pip install git+https://github.com/Jack-Grogan/DEMToolbox/
```

## Lacey Mixing Index

```python
#TODO
```

