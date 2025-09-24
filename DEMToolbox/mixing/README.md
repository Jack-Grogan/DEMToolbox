# Mixing
## The Lacey Mixing Index
### Theory

The Lacey mixing index, equation 1, is a measure of the sample variance of a target 
particles concentration in a binary particle system. Lacey first 
proposed the mixing index in 1954 as an extention to the mixing index 
proposed by Kramers [[1]](#1).

```math
\begin{align}
  M = \frac{\sigma_0^2 - \sigma^2}{\sigma_0^2 - \sigma_r^2}
\end{align} \tag{1}
``` 

#### System Variance

In line with the work of Kristensen [[2]](#2) and Fan et al. [[3]](#3) the 
liklihood of sampling a given particle is proportional to the particles volume
fraction within the sample. DEMToolbox calculates the system variance from
equation 2.

```math
\begin{align}
  \sigma^2 = \sum_{N_i=0}^{N_s}\frac{V_i}{V}[\phi_{0,i} - \bar{\phi}_0]^2
\end{align}
\tag{2}
```

When samples encompass all particles in the system the mean target particle
volume fraction across the samples, $\bar{\phi}_0$, equals the bulk target 
particle volume fraction within the system, $P_0$. The system variance 
computed by equation 2 extends the number fraction weighted variance outlined
by Chandratilleke et al. [[4]](#4), [[5]](#5) to polydisperse systems.

#### Segregared Variance

The perfectly segregated variance of the binary particle system can be derived 
from equation 2. A perfectly segregated system occurs when the systems samples
have either a concentration of 0 or 1. We can therefore separate equation 2
into the summation of samples with a target particle volume fraction of 1,
and the summation of samples with a target particle volume fraction of 0, 
equation 3.

```math
\begin{align}
  \sigma_0^2 = {\sum_{N_{0,i}=0}^{N_0}\frac{V_i}{V}[\phi_{0,i} - \bar{\phi}_0]^2}
           + {\sum_{N_{1,i}=0}^{N_1}\frac{V_i}{V}[\phi_{0,i} - \bar{\phi}_0]^2}
\end{align}  
\tag{3}
```

```math
\begin{align}
  \sigma_0^2 = {\sum_{N_{0,i}=0}^{N_0}\frac{V_i}{V}[1 - \bar{\phi}_0]^2}
           + {\sum_{N_{1,i}=0}^{N_1}\frac{V_i}{V}[0 - \bar{\phi}_0]^2}
\end{align}  
\tag{4}
```

```math
\begin{align}
  \sigma_0^2 = \frac{V_0}{V}(1 - \bar{\phi}_0)^2 + \bar{\phi}_0^2\frac{V - V_0}{V}
\end{align}
\tag{5}
```

Assuming samples encompass all particles within the system $\bar{\phi}_0 = P_0$.

```math
\begin{align}
  \sigma_0^2 = \frac{V_0}{V}(1 - P_0)^2 + P_0^2 \frac{V - V_0}{V}
\end{align}
\tag{6}
```

$V_0/V = P_0$

```math
\begin{align}
  \sigma_0^2 =P_0(1 - P_0)^2 + P_0^2 (1 - P_0)
\end{align}
\tag{7}
```


```math
\begin{align}
  \sigma_0^2 = P_0(1 - P_0)
\end{align}
\tag{8}
```

#### Mixed Variance

The variance of a perfectly mixed sample can be calculated by modelling the number of target particles as a binomial random variable. This method defines the perfectly mixed state as one in which the probability of sampling a target particle equals its probability in the bulk system. This method provides a more realistic definition of the perfectly mixed state, accounting for the natural variability between samples, unlike assuming a mixed variance of 0, which ignores unavoidable fluctuations even in a well-mixed system. Assuming each sample contains $\bar{n}$ particles, the variance in the number of target particles can be expressed by equation 9.

```math
\begin{align}
  \text{Var}[n_0] = \bar{n} P_0(1 - P_0)
\end{align}
\tag{9}
```

Instead of the variance in the number of target particles, we focus on the variance in their volume fraction across samples,  $\text{Var}[n_0 \cdot \bar{v}/V_s]$. Assuming minimal variation in mean particle size and total particle volume between samples, the variance in sample volume for a perfectly mixed powder bed can be expressed by equation 10.
        
```math
\begin{align}
  \text{Var}[\varphi_0] =  \left( \frac{\bar{v}}{V_s} \right)^2 \cdot \bar{n} P_0(1 - P_0)
\end{align}  
\tag{10}
```

The mean particle volume, $\bar{v}$, divided by the volume of particles within each sample, $V_s$, is equal to the mean number of particles within a sample $n$. The variance in the volume fraction of the target particle type in the perfectly mixed state, $\sigma_r$, can be finally be expressed by equation 11.

```math
\begin{align}
  \sigma_r^2 =  \frac{P_0(1 - P_0)}{\bar{n}}
\end{align}
\tag{11}
```

Defining the variance of a completely mixed state using a binomial distribution can yield Lacey mixing indices greater than 1; however, such values reflect random sampling fluctuations rather than increased homogeneity.


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

The returned `settled_data` is now updated with a column that, if 
not defined, is titled `f"{sample_vector[0]}_{sample_vector[1]}_{sample_vector[2]}_volume_sample"`.
Visualising in ParaView we can see the particles are perfectly segregated
into equal volumes:

![z_split_segregated](https://github.com/Jack-Grogan/DEMToolbox/blob/main/docs/images/z_split_segregated.png) 

Alternatively radial divisions of equal volume can be defined with the 
function `sample_1d_volume_cylinder`, the remainder of this example will
however use the `sample_1d_volume` example above.

```python
import pyvista as pv
from DEMToolbox.particle_sampling import sample_1d_volume_cylinder

settled_data = pv.read("settled_particles.vtk")
cylinder_point = [0, 0, 0]
cylinder_vector = [0, 0, 1]

settled_data, samples_cylinder = sample_1d_volume_cylinder(settled_data,
                                                           cylinder_point,
                                                           cylinder_vector,
                                                           resolution=2,
)

settled_data.save("updated_settled_particles.vtk")
```

![r_split_segregated](https://github.com/Jack-Grogan/DEMToolbox/blob/main/docs/images/r_split_segregated.png) 

Lacey needs to track how these two particles, different only in colour,
disperse. Each particle id's associated colour must therefore be appended to
each frame in the simulation prior to calculating the lacey index. Liggghts
will reorder the vtk file's rows between timesteps. The list can therefore
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
every timestep as opposed to appending a previous state's sample ID's.

```python
import pyvista as pv
from DEMToolbox.utilities import append_attribute

settled_data = pv.read("updated_settled_particles.vtk")
mixed_data = pv.read("updated_mixed_particles.vtk")

# Define a bounding box in which samples will be generated
bounds = [-0.03, 0.03, -0.03, 0.03, 0, 0.08]

# define three orthogonal vectors
vector_1 = [1, 0, 0]
vector_2 = [0, 1, 0]
vector_3 = [0, 0, 1]

# Number of splits in vector_1, vector_2 and vector_3 respectively
resolution = [10, 10, 30]

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


## Nomenclature

$M$ = Lacey mixing index \
$\sigma_0^2$ = Perfectly segregated variance \
$\sigma_r^2$ = Perfectly mixed variance \
$\sigma^2$ = System variance \
$V_i$ = Volume of particles in sample $i$ \
$V_s$ = Mean sample particle volume \
$V$ = Total volume of particles in the system \
$V_0$ = Total volume of target particles in the system \
$\bar{n}$ = Mean number of particles per sample \
$N_s$ = Number of Samples \
$\phi_{0,i}$ = Target particle volume fraction in sample $i$ \
$\bar{\phi}_0$ = Mean target particle volume fraction across the samples \
$P_0$ = Bulk target particle volume fraction

## References

<a id="1">[1]</a> 
P. M. C. Lacey, 
Developments in the theory of particle mixing,
Journal of Applied Chemistry 4 (5) (1954) 257–268. arXiv:
https://onlinelibrary.wiley.com/doi/pdf/10.1002/jctb.5010040504,
doi:https://doi.org/10.1002/jctb.5010040504.
URL https://www.sciencedirect.com/science/article/pii/S1674200121000614

<a id="2">[2]</a> 
H. Kristensen, 
Statistical properties of random and non-random mixtures of dry solids. part i. a general expression for the variance of the composition of samples, 
Powder Technology 7 (5) (1973) 249–257. doi:https://doi.org/10.1016/0032-5910(73)80031-2.
URL https://www.sciencedirect.com/science/article/pii/0032591073800312

<a id="3">[3]</a> 
L. Fan, J. Too, R. Rubison, F. Lai, 
Studies on multicomponent solids mixing and mixtures part iii. mixing indices, 
Powder Technology 24 (1) (1979) 73–89. doi:https://doi.org/10.1016/0032-5910(79)80009-1.
URL https://www.sciencedirect.com/science/article/pii/0032591079800091

<a id="4">[4]</a> 
G. R. Chandratilleke, Y. Zhou, A. Yu, J. Bridgwater, 
Effect of blade speed on granular flow and mixing in a cylindrical mixer, 
Industrial & engineering chemistry research 49 (11) (2010) 5467–5478.

<a id="5">[5]</a> 
R. Chandratilleke, A. Yu, J. Bridgwater, K. Shinohara, 
Flow and mixing of cohesive particles in a vertical bladed mixer,
Industrial & Engineering Chemistry Research 53 (10) (2014)4119–4130. 
arXiv:https://doi.org/10.1021/ie403877v,
doi:10.1021/ie403877v.
URL https://doi.org/10.1021/ie403877v
