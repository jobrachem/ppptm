# API reference

Public signatures include types and defaults from the Python annotations. Symbol
badges distinguish classes, methods, properties, and standalone functions; headings
use qualified names such as `ppptm.Model.fit`.

Most users start with `ppptm.Model.new_HG`. The lower-level factories and
Gaussian-process classes are available when its defaults need customization.

The reference is split by topic:

- [Model](model.md): `Model`, which builds, fits and evaluates a transformation
  model.
- [Marginals](marginals.md): `G`, which builds the parametric marginal
  distribution.
- [Coefficients](coefficients.md): `H`, which builds the spatial spline
  coefficients.
- [Locations](locations.md): `Locations`, `LocationVars` and grid helpers for
  sample and inducing locations.
- [Gaussian processes](gaussian-processes.md): `GPKernel` and the
  predictive-process variables behind spatially varying parameters.
- [Composite model](composite.md): `CompositeTransformations`, which couples a
  marginal model with a batram dependence model.
- [Data and plotting](data.md): the example data set and helpers for long-format
  tables and maps.
