# Velocity
## Velocity Vector Fields
### Theory

The projection of a vector **a** onto a vector **b** is given by equation 1.

```math
\begin{align}
\text{proj}_{\vec{b}} \vec{a} = \left(\frac{\vec{a} \cdot \vec{b}}{||\vec{b}||}\right) \frac{\vec{b}}{||\vec{b}||}
\end{align}
\tag{1}
```

Equation 1 can be used to calculate the component of a particles velocity in the direction of a given vector. In DEMToolbox this is used to calculate the component of a particles velocity in two orthogonal vectors defining a plane allowing for a 2D vector field to be constructed. The vectors defining the plane are user defined and can be set to any orientation. The velocity component in the direction normal to the plane is ignored. The projection of particles in the two orthogonal vectors $\text{dim}_1$ and $\text{dim}_2$ are calculated be equations 2 and 3 respectively.

```math
\begin{align}
\text{proj}_{\vec{\text{dim}_1}} \vec{v} = \left(\frac{\vec{v} \cdot \vec{\text{dim}_1}}{||\vec{\text{dim}_1}||}\right) \frac{\vec{\text{dim}_1}}{||\vec{\text{dim}_1}||}
\end{align}
\tag{2}
``` 

```math
\begin{align}
\text{proj}_{\vec{\text{dim}_2}} \vec{v} = \left(\frac{\vec{v} \cdot \vec{\text{dim}_2}}{||\vec{\text{dim}_2}||}\right) \frac{\vec{\text{dim}_2}}{||\vec{\text{dim}_2}||}
\end{align}
\tag{3}
```

The projected velocity on the plane is then given by equation 4.

```math
\begin{align}
\vec{v}_{\text{proj}} = \text{proj}_{\vec{\text{dim}_1}} \vec{v} + \text{proj}_{\vec{\text{dim}_2}} \vec{v}
\end{align}
\tag{4}
```

To ease visualisation the projected velocity vectors are then binned into a 2D grid on the plane defined by $\text{dim}_1$ and $\text{dim}_2$. The average velocity
vector in each bin is then calculated on a number basis (i.e. each particle contributes equally to the average to the bins average velocity).

### Implementation

The `velocity_vector_field` function calculates the 2D binned velocity vector field for a given set of particles. To calculate the resolved velocity field the user provides a plane with a thickness allowing for a vector field to be calculated in a slice through the system. The plane is defined by a point on the plane and two orthogonal vectors defining the plane orientation. The thickness of the plane is defined by a distance normal to the plane. Half of the plane thickness is added and subtracted from the point on the plane to define the slice in which particles are considered for the velocity field. The number of bins in each direction is defined by the user as a resolution parameter. The function internally calls the `sample_2d_slice` function to get the particles in the slice and bin them in 2D space. The particles in each bin then have their velocity projected onto the plane and the average velocity vector in each bin is calculated.

To calculate the velocity vector field the following code can be used. Firstly define
the plane with a point on the plane and two orthogonal vectors defining the plane orientation. The thickness of the plane is defined by a distance normal to the plane. The number of bins in each direction is defined by the user as a resolution parameter.
Provide your particle data in the form of a  PyVista PolyData object with velocity vectors stored in a column named 'v' and unique particle ids stored in a column named 'id'. Note if your velocity vectors or ids are stored in under a different column name change the `velocity_column` and `particle_id_column` parameters in the function call below.

```python
from DEMToolbox.velocity import velocity_vector_field
import pyvista as pv

# Load particle data
particle_data = pv.read('particle_data.vtk')

# Define plane
point_on_plane = [0.0, 0.0, 0.0]
dim_1 = [1.0, 0.0, 0.0]
dim_2 = [0.0, 0.0, 1.0]

bounds = [-0.03, 0.03, -0.03, 0.03, -0.008, 0.088] 
plane_thickness = 1 # Much larger than container height width include all particles

# Define binning resolution as 15 bins in x and 24 bins in z 
resolution = [15, 24]

# Calculate velocity vector field
results = velocity_vector_field(
    particle_data,
    bounds,
    point_on_plane,
    dim_1,
    dim_2,
    plane_thickness,
    resolution,
    velocity_column='v',
    append_column="mean_resolved_velocity",
    particle_id_column='id'
)

updated_particle_data = results[0]
velocity_field = results[1]
occupancy = results[2]
samples = results[3]
```

The function returns the particle data updated with the projected velocity vectors in each bin applied to each particle in the bin. This data is by default stored in a column named 'mean_resolved_velocity' but this can be changed by the user with the `append_column` parameter. The function also returns the the average velocity vector in each bin of shape (number of $\text{dim}_2$ bins, number of $\text{dim}_1$ bins, 3) where the last dimension is the x, y and z components of the average velocity vector in each bin. The occupancy of each bin is also returned of shape (number of $\text{dim}_2$ bins, number of $\text{dim}_1$ bins). Finally a ParticleSamples object is returned for the internally called `sample_2d_slice` function.

The samples generated can be seen in the figure below. Each 2D sample has been labelled with the sample id.
![2d_samples](https://github.com/Jack-Grogan/DEMToolbox/blob/main/docs/images/2D_sampling.png)

The position of each sample in the returned 2d occupancy and velocity matrices can be seen in the matrix below. Index [0, 0] of the matrices corresponds to minimum bound of vector 1 and minimum bound of vector 2. 

$$
\text{Samples} = \begin{pmatrix}
  0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 & 11 & 12 & 13 & 14 \\
  15 & 16 & 17 & 18 & 19 & 20 & 21 & 22 & 23 & 24 & 25 & 26 & 27 & 28 & 29 \\ 
  30 & 31 & 32 & 33 & 34 & 35 & 36 & 37 & 38 & 39 & 40 & 41 & 42 & 43 & 44 \\
  45 & 46 & 47 & 48 & 49 & 50 & 51 & 52 & 53 & 54 & 55 & 56 & 57 & 58 & 59 \\
  60 & 61 & 62 & 63 & 64 & 65 & 66 & 67 & 68 & 69 & 70 & 71 & 72 & 73 & 74 \\
  75 & 76 & 77 & 78 & 79 & 80 & 81 & 82 & 83 & 84 & 85 & 86 & 87 & 88 & 89 \\
  90 & 91 & 92 & 93 & 94 & 95 & 96 & 97 & 98 & 99 & 100 & 101 & 102 & 103 & 104 \\
  105 & 106 & 107 & 108 & 109 & 110 & 111 & 112 & 113 & 114 & 115 & 116 & 117 & 118 & 119 \\
  120 & 121 & 122 & 123 & 124 & 125 & 126 & 127 & 128 & 129 & 130 & 131 & 132 & 133 & 134 \\
  135 & 136 & 137 & 138 & 139 & 140 & 141 & 142 & 143 & 144 & 145 & 146 & 147 & 148 & 149 \\
  150 & 151 & 152 & 153 & 154 & 155 & 156 & 157 & 158 & 159 & 160 & 161 & 162 & 163 & 164 \\
  165 & 166 & 167 & 168 & 169 & 170 & 171 & 172 & 173 & 174 & 175 & 176 & 177 & 178 & 179 \\
  180 & 181 & 182 & 183 & 184 & 185 & 186 & 187 & 188 & 189 & 190 & 191 & 192 & 193 & 194 \\
  195 & 196 & 197 & 198 & 199 & 200 & 201 & 202 & 203 & 204 & 205 & 206 & 207 & 208 & 209 \\
  210 & 211 & 212 & 213 & 214 & 215 & 216 & 217 & 218 & 219 & 220 & 221 & 222 & 223 & 224 \\
  225 & 226 & 227 & 228 & 229 & 230 & 231 & 232 & 233 & 234 & 235 & 236 & 237 & 238 & 239 \\
  240 & 241 & 242 & 243 & 244 & 245 & 246 & 247 & 248 & 249 & 250 & 251 & 252 & 253 & 254 \\
  255 & 256 & 257 & 258 & 259 & 260 & 261 & 262 & 263 & 264 & 265 & 266 & 267 & 268 & 269 \\
  270 & 271 & 272 & 273 & 274 & 275 & 276 & 277 & 278 & 279 & 280 & 281 & 282 & 283 & 284 \\
  285 & 286 & 287 & 288 & 289 & 290 & 291 & 292 & 293 & 294 & 295 & 296 & 297 & 298 & 299 \\
  300 & 301 & 302 & 303 & 304 & 305 & 306 & 307 & 308 & 309 & 310 & 311 & 312 & 313 & 314 \\
  315 & 316 & 317 & 318 & 319 & 320 & 321 & 322 & 323 & 324 & 325 & 326 & 327 & 328 & 329 \\
  330 & 331 & 332 & 333 & 334 & 335 & 336 & 337 & 338 & 339 & 340 & 341 & 342 & 343 & 344 \\
  345 & 346 & 347 & 348 & 349 & 350 & 351 & 352 & 353 & 354 & 355 & 356 & 357 & 358 & 359 \\
\end{pmatrix}
$$

## Mean Velocity Vector Fields

The `velocity_vector_field` function covered above calculates the binned mean velocity vector field for a given simulation frame. To calculate the mean velocity vector field over a time period the `mean_velocity_vector_field` function can be used. This function uses a list of vector field arrays and occupancy arrays generated from the `velocity_vector_field` function to calculate the mean velocity vector field over a time period. The time and depths averaged mean velocity vector field is calculated by summing the velocity vectors in each bin weighted by the occupancy of each bin and dividing by the total occupancy of each bin over the time period. Average velocity is calculated with `numpy.nanmean` so that bins with no occupancy are ignored in the bins average velocity. Average occupancy should be calculated including zero values so that bins with no occupancy are correctly represented in the average occupancy field.

```python
from DEMToolbox.velocity import mean_velocity_vector_field
import numpy as np

# Load list of velocity fields and occupancy arrays
velocity_fields = [np.load('velocity_field_0.npy'), np.load('velocity_field_1.npy'), ...]
occupancies = [np.load('occupancy_0.npy'), np.load('occupancy_1.npy'), ...]

# Calculate mean velocity vector field
mean_velocity_field = mean_velocity_vector_field(velocity_fields, occupancies)
```
