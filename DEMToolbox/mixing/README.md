# Mixing Theory
## Lacey Mixing Index

The Lacey mixing index, equation 1, is a measure of the sample variance of a target 
particles concentration in a binary particle system. Lacey first 
proposed the mixing index in 1954 as an extention to the mixing index 
proposed by Kramers [[1]](#1).

$$
  M = \frac{\sigma_0^2 - \sigma^2}{\sigma_0^2 - \sigma_r^2}\tag{1}
$$

### System Variance

In line with the work of Kristensen [[2]](#2) and Fan et al. [[3]](#3) the 
liklihood of sampling a given particle is proportional to the particles volume
fraction within the sample. DEMToolbox calculates the system variance from
equation 2

$$
  \sigma = \sum_{N_i=0}^{N_s}\frac{V_i}{V}[\phi_{0,i} - \bar{\phi}_0]^2
  \tag{2}
$$

When samples encompass all particles in the system the mean target particle
volume fraction across the samples, $\bar{\phi}_0$, equals the bulk target 
particle volume fraction within the system, $P_0$. The system variance 
computed by equation 2 extends the number fraction weighted variance outlined
by Chandratilleke et al. [[4]](#4), [[5]](#5) to polydisperse systems.

### Segregared Variance

The perfectly segregated variance of the binary particle system can be derived 
from equation 2. A perfectly segregated system occurs when the systems samples
have either a concentration of 0 or 1. We can therefore separate equation 2
into the summation of samples with a target particle volume fraction of 1,
and the summation of samples with a target particle volume fraction of 0, 
equation 3

$$
  \sigma_0 = \sum_{N_{0,i}=0}^{N_0}\frac{V_i}{V}[\phi_{0,i} - \bar{\phi}_0]^2 
  + \sum_{N_{1,i}=0}^{N_1}\frac{V_i}{V}[\phi_{0,i} - \bar{\phi}_0]^2\tag{3}
$$

$$
  \sigma_0 = \sum_{N_{0,i}=0}^{N_0}\frac{V_i}{V}[1 - \bar{\phi}_0]^2 
  + \sum_{N_{1,i}=0}^{N_1}\frac{V_i}{V}[0 - \bar{\phi}_0]^2\tag{4}
$$

$$
  \sigma_0 = \frac{V_0}{V}(1 - \bar{\phi}_0)^2 
  + \bar{\phi}_0^2\frac{V - V_0}{V}\tag{5}
$$

Assuming samples encompass all particles within the system $\bar{\phi}_0 = P_0$

$$
  \sigma_0 = \frac{V_0}{V}(1 - P_0)^2 
  + P_0^2 \frac{V - V_0}{V}\tag{6}
$$

$V_0/V = P_0$

$$
  \sigma_0 =P_0(1 - P_0)^2 
  + P_0^2 (1 - P_0)\tag{7}
$$


$$
  \sigma_0 =P_0(1 - P_0)\tag{8}
$$

## Mixed Variance

The variance of a perfectly mixed sample can be calculated by modelling the number of target particles as a binomial random variable. This method defines the perfectly mixed state as one in which the probability of sampling a target particle equals its probability in the bulk system. This method provides a more realistic definition of the perfectly mixed state, accounting for the natural variability between samples, unlike assuming a mixed variance of 0, which ignores unavoidable fluctuations even in a well-mixed system. Assuming each sample contains $\bar{n}$ particles, the variance in the number of target particles can be expressed by equation 9.

$$
\text{Var}[n_0] = \bar{n} P_0(1 - P_0)\tag{9}
$$

Instead of the variance in the number of target particles, we focus on the variance in their volume fraction across samples,  $\text{Var}[n_0 \cdot \bar{v}/V_s]$. Assuming minimal variation in mean particle size and total particle volume between samples, the variance in sample volume for a perfectly mixed powder bed can be expressed by equation 10.
        
$$
\text{Var}[\varphi_0] =  \left( \frac{\bar{v}}{V_s} \right)^2 \cdot \bar{n} P_0(1 - P_0)\tag{10}
$$

The mean particle volume, $\bar{v}$, divided by the volume of particles within each sample, $V_s$, is equal to the mean number of particles within a sample $n$. The variance in the volume fraction of the target particle type in the perfectly mixed state, $\sigma_r$, can be finally be expressed by equation 11.

$$
\sigma_r =  \frac{P_0(1 - P_0)}{\bar{n}}\tag{11}
$$

Defining the variance of a completely mixed state using a binomial distribution can yield Lacey mixing indices greater than 1; however, such values reflect random sampling fluctuations rather than increased homogeneity.

## Nomenclature

$M$ = Lacey mixing index \
$\sigma_0$ = Perfectly segregated variance \
$\sigma_r$ = Perfectly mixed variance \
$\sigma$ = System variance \
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
