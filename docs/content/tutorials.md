# Tutorials

Nine notebooks (new words are in the [glossary](glossary)), each building on the last: write and run a kernel, pick the
right plane type, write a kernel that stops itself, take gradients
backward and forward, read/scatter/reduce across a batch, differentiate
through branches and stops, and choose how a branch between several
formulas is compiled, and reach neighbouring samples with `.at` and `sample_index`.

```{toctree}
:maxdepth: 1

tutorials/01_your_first_kernel
tutorials/02_arguments_and_shapes
tutorials/03_stop_when_youre_done
tutorials/04_gradients_backward
tutorials/05_gradients_forward
tutorials/06_lookups_scatters_and_sums
tutorials/07_gradients_through_branches_and_stops
tutorials/08_choosing_a_dispatch_policy
tutorials/09_per_sample_control
```

Next: {doc}`hawk in your domain <vocabulary>` — declare a kernel's own vocabulary once and
reuse it across a whole family of models.
