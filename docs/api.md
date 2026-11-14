# API reference

Public signatures include types and defaults from the Python annotations. Every
class has its own section, so methods and properties appear beneath their owner in
the navigation. Symbol badges distinguish classes, methods, properties, and
standalone functions; headings use qualified names such as `ppptm.Model.fit`.

Most users start with `ppptm.Model.new_HG`. The lower-level factories and
Gaussian-process classes are available when its defaults need customization.

## Core model class: `Model`

**Class**

::: ppptm.Model

**Class constructors**

::: ppptm.Model.new_HG

::: ppptm.Model.new_G

**Properties**

::: ppptm.Model.parameters

::: ppptm.Model.spatial_coef

**Fitting and evaluation methods**

::: ppptm.Model.fit

::: ppptm.Model.log_prob

::: ppptm.Model.init_dist

**Transformation methods**

::: ppptm.Model.hg

::: ppptm.Model.hgi

::: ppptm.Model.g

::: ppptm.Model.gi

::: ppptm.Model.h

::: ppptm.Model.hi

## Marginal factory class: `G`

**Class**

::: ppptm.G

**Distribution constructors**

::: ppptm.G.new_gaussian

::: ppptm.G.new_skewnorm

::: ppptm.G.new_skewt

::: ppptm.G.new_gamma

::: ppptm.G.new_weibull

**Parameter constructors**

::: ppptm.G.new_param

::: ppptm.G.new_param_const

::: ppptm.G.new_param_locwise

::: ppptm.G.new_amplitude

::: ppptm.G.new_length_scale

## Transformation-coefficient factory class: `H`

**Class**

::: ppptm.H

**Coefficient constructor**

::: ppptm.H.new_coef

**Kernel-parameter constructors**

::: ppptm.H.new_amplitude

::: ppptm.H.new_amplitude_const

::: ppptm.H.new_amplitude_locwise

::: ppptm.H.new_length_scale

## Composite-dependence class: `CompositeTransformations`

**Class**

::: ppptm.CompositeTransformations

**Methods**

::: ppptm.CompositeTransformations.init_dependence_model

::: ppptm.CompositeTransformations.log_score

::: ppptm.CompositeTransformations.sample

**Properties**

::: ppptm.CompositeTransformations.locs

::: ppptm.CompositeTransformations.y

## Location metadata class: `Locations`

**Class**

::: ppptm.Locations

**Constructors and conversion methods**

::: ppptm.Locations.new_from_unordered

::: ppptm.Locations.from_2d_to_3d

**Properties**

::: ppptm.Locations.ordered

::: ppptm.Locations.ordered_subset

::: ppptm.Locations.nloc

## Location-variable class: `LocationVars`

**Class**

::: ppptm.LocationVars

**Class constructors**

::: ppptm.LocationVars.new_from

::: ppptm.LocationVars.new_from_ordered

**Methods**

::: ppptm.LocationVars.copy

**Properties**

::: ppptm.LocationVars.sample_locs

::: ppptm.LocationVars.inducing_locs

::: ppptm.LocationVars.locs

## Kernel class: `GPKernel`

**Class**

::: ppptm.GPKernel

**Method**

::: ppptm.GPKernel.new

**Properties**

::: ppptm.GPKernel.amplitude

::: ppptm.GPKernel.length_scale

## Scalar predictive-process class: `ParamPredictiveProcessGP`

**Class**

::: ppptm.ParamPredictiveProcessGP

**Sampling methods**

::: ppptm.ParamPredictiveProcessGP.sample_array

::: ppptm.ParamPredictiveProcessGP.sample_df

`GPVar` is an alias for `ParamPredictiveProcessGP`.

## Coefficient predictive-process class: `RandomWalkParamPredictiveProcessGP`

**Class**

::: ppptm.RandomWalkParamPredictiveProcessGP

**Sampling methods**

::: ppptm.RandomWalkParamPredictiveProcessGP.sample_array

::: ppptm.RandomWalkParamPredictiveProcessGP.sample_df

`GPTMCoef` and `SpatPTMCoef` are aliases for
`RandomWalkParamPredictiveProcessGP`.

## Location functions

::: ppptm.unit_grid

::: ppptm.unit_grid_vars

## Data, summary, and plotting functions

::: ppptm.load_americas

::: ppptm.long_df

::: ppptm.long_df_multiple

::: ppptm.summary_at_2d_locs

::: ppptm.plot_df
