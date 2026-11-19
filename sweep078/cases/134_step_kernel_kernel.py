"""134_step_kernel_kernel: step_kernel_kernel (examples/autodiff_vs_torch.py: file (example))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

MU = 1.0


def step_kernel():
    """Trace the primal RK4 step -- the same physics as
    ``early_termination.py``'s ``et_step``, with no termination test."""
    import hawk
    import hawk.math as hm

    def rhs(s):
        r2 = s[0] * s[0] + s[1] * s[1]
        inv_r3 = hm.rsqrt(r2 * r2 * r2)
        return hm.vec(s[2], s[3], -MU * s[0] * inv_r3, -MU * s[1] * inv_r3)

    @hawk.kernel
    def rk4_step(x: hawk.Vector[4], dt: hawk.Param,
                x_next: hawk.Mutable[hawk.Vector[4]]):
        k1 = rhs(x)
        k2 = rhs(x + (0.5 * dt) * k1)
        k3 = rhs(x + (0.5 * dt) * k2)
        k4 = rhs(x + dt * k3)
        x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return rk4_step


step_kernel_kernel = step_kernel()


CASE = {
    "id": "134_step_kernel_kernel",
    "source": [
        "examples/autodiff_vs_torch.py: file (example)",
    ],
    "kernels": [
        "step_kernel_kernel",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
