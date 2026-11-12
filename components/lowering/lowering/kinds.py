# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The names of the op kinds a lowering form leaves behind that the shared writer reads (the form that makes them is another component's)."""

#: The kind of a per-element read of a plane the lane's warp holds.
WARP_READ = "warp_read"

#: The kind of the sum of one value per lane over a warp.
WARP_SUM_KIND = "warp_sum"
