# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.24] - 2026-02-22

### Added
None

### Changed
DEMToolbox.mixing.lacey_mixing has been renamed to DEMToolbox.mixing.lacey_mixing_curve_fit to better reflect the functionality of the module. The function lacey_mixing_curve_fit has been updated to return the covariance of the fitted parameters instead of the r2 score, as this is more informative for assessing the quality of the fit. The docstring for lacey_mixing_curve_fit has been updated to reflect these changes.

### Deprecated
DEMToolbox.mixing.lacey_mixing is now deprecated. Users should switch to using DEMToolbox.mixing.lacey_mixing_curve_fit instead. r2 score is no longer returned by lacey_mixing_curve_fit, so users should not rely on this metric for assessing fit quality. This change was implemented due to the different ways in which r2 score can be calculated with one of the methods allowing for negative r2 scores and the other not, which could lead to confusion.

### Removed
DEMToolbox.mixing.lacey_mixing has been removed. Users should switch to using DEMToolbox.mixing.lacey_mixing_curve_fit instead. The r2 score is no longer returned by lacey_mixing_curve_fit.

### Fixed
None

### Security
None

## [0.0.23] - 2025-10-06

### Added
Added data regarding bounds and centers to ParticleSamples object returned from
sampling functions. This aids post procssing as the user can now easily access the
bounds and centers of each sample in each dimension. Added tests for this new
functionality.

### Changed
None

### Deprecated
None

### Removed
None

### Fixed
None

### Security
None

## [0.0.22] - 2025-09-02

### Added
Added functionality to sample_1d, sample_2d and sample_3d to allow user to
specify bounds as arrays and lists rather than just lists or vtk files.

### Changed
None

### Deprecated
None

### Removed
None

### Fixed
None

### Security
None


## [0.0.21] - 2025-09-02

### Added
None

### Changed
Internal changes in sample_1d to reflect the change away from the need for container data.

### Deprecated
None

### Removed
None

### Fixed
None

### Security
None

## [0.0.20] - 2025-09-02

### Added
Added a long awaited changelog. Added functionality for particle sampling
across the bounds of a user defined bounding box rather than requiring a 
vtk file. Added aditional test for this new functionality.

### Changed
Changed the container data field in sample inputs to bounds to match more
general functionality.

### Deprecated
None

### Removed
2D slice bounds, and as a consequnce vector field bounds, now in cartesian 
coordintes rather than the specified vector coordinate system. The bounds must therefore be given in the form [ $x$<sub>min</sub>, $x$<sub>max</sub>, $y$<sub>min</sub>, $y$<sub>max</sub>, $z$<sub>min</sub>, $z$<sub>max</sub> ].

### Fixed
Ambiguity in resolution fields.

### Security
None
