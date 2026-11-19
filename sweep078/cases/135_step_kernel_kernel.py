"""135_step_kernel_kernel: step_kernel_kernel (examples/two_body_hybrid.py: file (example))."""
# ruff: noqa: F821, F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

MU = 1.0

HIDDEN = 32


def step_kernel(hidden: int = HIDDEN):
    """Trace the RK4 step. The MLP is evaluated inline, four times per step.

    The weights are four ``Table`` arguments: one copy shared by every
    trajectory, not one per sample. The kernel stays a pure function of
    ``(state, weights, dt)`` and carries no state of its own.
    """
    import hawk
    import hawk.math as hm

    def hybrid_rhs(s, w1, b1, w2, b2):
        """Gravity plus the network's correction, as one traced expression."""
        r2 = s[0] * s[0] + s[1] * s[1]
        inv_r3 = hm.rsqrt(r2 * r2 * r2)
        act = []
        for h in range(hidden):
            pre = b1.at(h=h)
            for i in range(4):
                pre = pre + w1.at(h=h, i=i) * s[i]
            act.append(hm.tanh(pre))
        corr = []
        for o in range(2):
            out = b2.at(o=o)
            for h in range(hidden):
                out = out + w2.at(o=o, h=h) * act[h]
            corr.append(out)
        return hm.vec(
            s[2],
            s[3],
            -MU * s[0] * inv_r3 + corr[0],
            -MU * s[1] * inv_r3 + corr[1],
        )

    @hawk.kernel
    def two_body_hybrid_step(
        x: hawk.Vector[4],
        w1: hawk.Table["h":hidden, "i":4],
        b1: hawk.Table["h":hidden],
        w2: hawk.Table["o":2, "h":hidden],
        b2: hawk.Table["o":2],
        dt: hawk.Param,
        x_next: hawk.Mutable[hawk.Vector[4]],
    ):
        k1 = hybrid_rhs(x, w1, b1, w2, b2)
        k2 = hybrid_rhs(x + (0.5 * dt) * k1, w1, b1, w2, b2)
        k3 = hybrid_rhs(x + (0.5 * dt) * k2, w1, b1, w2, b2)
        k4 = hybrid_rhs(x + dt * k3, w1, b1, w2, b2)
        x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return two_body_hybrid_step


step_kernel_kernel = step_kernel()


CASE = {
    "id": "135_step_kernel_kernel",
    "source": [
        "examples/two_body_hybrid.py: file (example)",
    ],
    "kernels": [
        "step_kernel_kernel",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
