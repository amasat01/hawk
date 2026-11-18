# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""hawk's refusals.

:class:`HawkError` is what every refusal raised by hawk, or relayed from a compiled
library, is.
"""

#: What every door says that asks for the embedded C target (``targets`` naming ``"c"``,
#: ``embedded_profile=``, ``hawk.embedded``).
EMBEDDED_C = "the embedded C target is not available in this version: build for 'host' or 'cuda'"

__all__ = ["EMBEDDED_C", "HawkError", "ToolchainAbsent"]


class HawkError(Exception):
    """A refusal, with a sentence that names the thing and the way out.

    A refusal relayed from a compiled library keeps the library's ``code`` (a short word a
    caller may act on); one hawk raises itself has ``code`` ``None``.
    """

    code = None


class ToolchainAbsent(HawkError):  # noqa: N818 - the public name callers catch
    """The machine lacks a tool a build needs; ``toolchain`` names which, as a library states it."""

    def __init__(self, message: str = "", *, toolchain: str = "") -> None:
        """Make the refusal with its ``message`` and the name of the missing ``toolchain``."""
        super().__init__(message)
        self.toolchain = toolchain
