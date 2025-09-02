# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
