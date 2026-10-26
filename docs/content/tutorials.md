# Tutorials

Short notebooks that build on each other, grouped by topic. Within a topic they go from first use to advanced. New words are in the [glossary](glossary).

## Write kernels

Write, build and run a kernel; pick the plane types; stop a sample when it is done; reach neighbouring samples.

```{toctree}
:maxdepth: 1

tutorials/01_your_first_kernel
tutorials/02_arguments_and_shapes
tutorials/03_stop_when_youre_done
tutorials/09_per_sample_control
tutorials/08_choosing_a_dispatch_policy
```

## Derivatives

Gradients backward and forward, through branches and stops, and with a derivative you supply yourself.

```{toctree}
:maxdepth: 1

tutorials/04_gradients_backward
tutorials/05_gradients_forward
tutorials/07_gradients_through_branches_and_stops
vocabulary/03_your_own_derivative
vocabulary/05_tapes_and_events
```

## Reductions and scatters

Read a table, scatter into bins, and sum across a batch.

```{toctree}
:maxdepth: 1

tutorials/06_lookups_scatters_and_sums
```

## Your own vocabulary

Declare the objects of your field once and every kernel speaks them.

```{toctree}
:maxdepth: 1

vocabulary
vocabulary/01_a_family_of_kernels
vocabulary/02_extend_by_inheritance
vocabulary/04_capstone
```

Next: the [Domains](domains) pages, one worked case per field.
