![logo](https://github.com/Jack-Grogan/DEMToolbox/blob/main/docs/images/logo.png) 

# DEMToolbox
## Post Processing Tools for Analysis of DEM Simulations

DEMToolbox provides a range of post processing tools for analysing DEM 
simulations. Performance optimisations have tried to be attained where possible
through use of [numpy](https://numpy.org/) whose core is written in optimised 
C code. Often users will want to apply this libraries functionality to many 
simulation output files. This problem is "embarrassingly parallel" and can be 
sped up using `ProcesssPoolExecutor` from 
[concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html).

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

The theory behind the Lacey mixing index can be found [here](https://github.com/Jack-Grogan/DEMToolbox/tree/main/DEMToolbox/mixing) 

### Defining the binary particle system required by Lacey
The Lacey mixing index requires two particle types to be present within the 
powder bed that are perfectly segregated prior to mixing. Lacey is most
effective when these two particle types are present in equal volumes 
within the system. DEMToolbox provides users with the functionality to define
an equal volume segregated powder along a provided vector with 
`sample_1d_volume`:

```python
import pyvista as pv
from DEMToolbox.particle_sampling import sample_1d_volume

settled_data = pv.read("settled_particles.vtk")
sample_vector = [0, 0, 1]

settled_data, samples = sample_1d_volume(settled_data,
                                          sample_vector,
                                          resolution=2
)

settled_data.save("updated_settled_particles.vtk")
```

The returned `settled_data` is now updated with a column that if 
not defined is titled `f"{sample_vector[0]}_{sample_vector[1]}_{sample_vector[2]}_volume_sample"`.
Visualising in ParaView we can see the particles are perfectly segregated
into equal volumes:

![z_split_segregated](https://github.com/Jack-Grogan/DEMToolbox/blob/main/docs/images/z_split_segregated.png) 

Alternatively radial divisions of equal volume can be defined with the 
function `sample_1d_volume_cylinder`:

```python
import pyvista as pv
from DEMToolbox.particle_sampling import sample_1d_volume

settled_data = pv.read("settled_particles.vtk")
cylinder_point = [0, 0, 0]
cylinder_vector = [0, 0, 1]

settled_data, samples = sample_1d_volume_cylinder(settled_data,
                                                   cylinder_point,
                                                   cylinder_vector,
                                                   resolution=2,
)

settled_data.save("updated_settled_particles.vtk")
```

![r_split_segregated](https://github.com/Jack-Grogan/DEMToolbox/blob/main/docs/images/r_split_segregated.png) 

Lacey needs to track how these two particles different only in colour
disperse. Each particle id's associated colour must therefore be appended to
each frame in the simulation prior to calculating the lacey index. Liggghts
will reorder the vtk files rows between timesteps. The list can therefore
not be simply appended in the same order as in the settled state. Instead the
colour value needs to be added on the appropriate id that is unique for each
particle. This can be achieved by passing the `ParticleAttribute` attribute of
the returned `samples` to the function `append_attribute` along with the 
particles file you desire to append the split data to:

```python
import pyvista as pv
from DEMToolbox.utilities import append_attribute

mixed_data = pv.read("mixed_particles.vtk")

mixed_data = append_attribute(mixed_data, samples.ParticleAttribute)

mixed_data.save("updated_mixed_particles.vtk")
```

The mixed data now has each unique particle coloured appropriately:

![z_split_mixed](https://github.com/Jack-Grogan/DEMToolbox/blob/main/docs/images/z_split_mixed.png) 

### Defining samples throughout the system as required by Lacey

The next step towards calculating Lacey is to divide the particles into samples
based on their position. Samples should be of the same volume within each study
and across studies you wish to compare. ***IMPORTANT: Lacey mixing indices calculated from 
different volume samples are not comparable***. Care should be taken when selecting an
appropriate number of samples. With too few samples your Lacey mixing index will be
unrepresentative of the true system mixedness (at the extreme case of only 1 
sample a perfectly segregated system will register as perfectly mixed). With too
many samples, progression in the lacey mixing index will be noisy (at the extreme 
case each sample will contain only 1 particle causing a truly perfectly mixed 
system to register as perfectly segregated).

DEMToolbox provides two 3d sampling functions for dividing a powder system into
samples: `sample_3d` and `sample_3d_cylinder`. `sample_3d` defines samples as
cuboids with dimensions defined by the resolution along the provided 3 orthogonal
vectors. `sample_3d_cylinder` currently only works with cylinder whose principle
axis is parallel to the z axis. `sample_3d_cylinder` creates samples azimuthally,
radially and vertically. The functionality for both of these functions is the 
same as the 1d sampling functions discussed above. The difference comes in how we
implement them. For defining samples we need update each particles sample ID at
every timestep as opposed to appending a previous states sample ID's.

```python
import pyvista as pv
from DEMToolbox.utilities import append_attribute

settled_data = pv.read("updated_settled_particles.vtk")
mixed_data = pv.read("updated_mixed_particles.vtk")

# Define a bounding box in which samples will be generated
bounds = [0, 1, 0, 1, 0, 1]

# define three orthogonal vectors
vector_1 = [1, 0, 0]
vector_2 = [0, 1, 0]
vector_3 = [0, 0, 1]

# Number of splits in vector_1, vector_2 and vector_3 respectively
resolution = [20, 20, 20]

# Sample settled data
settled_data, settled_data_samples = sample_3d(settled_data,
                                               bounds,
                                               vector_1,
                                               vector_2,
                                               vector_3,
                                               resolution,
)

# Sample mixed data
mixed_data, mixed_data_samples = sample_3d(mixed_data,
                                           bounds,
                                           vector_1,
                                           vector_2,
                                           vector_3,
                                           resolution,
)

settled_data.save("updated_settled_particles.vtk")
mixed_data.save("updated_mixed_particles.vtk")

```

If not specified otherwise `sample_3d` will append a column titled 
`"3D_samples"` to the vtk files. The generated samples can be visualised
in ParaView:

![lacey_samples](https://github.com/Jack-Grogan/DEMToolbox/blob/main/docs/images/lacey_samples.png) 

### Calculating the Lacey mixing index.

Having defined a binary particle system and created samples the lacey 
mixing index can be calculated using the function `macro_scale_lacey_mixing`:

```python
from DEMToolbox.mixing import macro_scale_lacey_mixing

settled_data, settled_lacey = macro_scale_lacey_mixing(mixed_data, 
                                                       samples.ParticleAttribute,
                                                       settled_data_samples,
                                                       )

mixed_data, mixed_lacey = macro_scale_lacey_mixing(mixed_data, 
                                                   samples.ParticleAttribute,
                                                   mixed_data_samples,
                                                   )

settled_data.save("updated_settled_particles.vtk")
mixed_data.save("updated_mixed_particles.vtk")
```

The settled and mixed data now have a column assigning the samples target 
particle volume fraction to each particle in the sample allowing for 
visualisation in ParaView of the initially segregated state:

![segregated_conc](https://github.com/Jack-Grogan/DEMToolbox/blob/main/docs/images/segregated_conc.png) 

and the mixed state:

![mixed_conc](https://github.com/Jack-Grogan/DEMToolbox/blob/main/docs/images/mixed_conc.png) 