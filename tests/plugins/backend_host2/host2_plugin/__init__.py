# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A backend registered from outside hawk: the host backend with a marker line at the end of its unit."""

from dataclasses import replace

from ospreys.cpu.backend import Host


class Host2(Host):
    id = "host2"
    about = "for the CPU, marked"

    def finish(self, source, lowered, options):
        return replace(source, text=source.text + "// host2\n")


BACKEND = Host2()
