# Vocabulary

Five tutorials that teach writing and extending a kernel's own vocabulary,
end to end: declaring a shared shape once for a family of kernels, growing
it by ordinary Python inheritance, giving a function a derivative you
supply, a capstone that runs the whole machinery at scale and hands it
to PyTorch, and the tape and stop-rule guard a runtime needs to reverse a
whole rollout. Two further pages round it out: the decision between a
primitive, `raw_device`, and a native op, and the contributor path for the
last of those.

```{toctree}
:maxdepth: 1

vocabulary/01_a_family_of_kernels
vocabulary/02_extend_by_inheritance
vocabulary/03_your_own_derivative
vocabulary/04_capstone
vocabulary/05_tapes_and_events
vocabulary/when_hawk_cant_spell_it
```
