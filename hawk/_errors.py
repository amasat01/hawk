# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The public name of the refusal type: ``hawk.HawkError`` IS :class:`raptor.errors.RaptorError`, so ``except hawk.HawkError``
catches every refusal, including the ones raised below hawk. hawk's modules import the name from here."""

from raptor.errors import RaptorError as HawkError

#: What every door says that asks for the embedded C target (``targets`` naming ``"c"``, ``embedded_profile=``, ``hawk.embedded``).
EMBEDDED_C = "the embedded C target is not available in this version: build for 'host' or 'cuda'"

__all__ = ["HawkError"]
