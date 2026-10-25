# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""HAWK's tensor-typed IR: nodes, compound quantities, THE canonical
walk and access-class inference.

An INTERNAL package: ``__all__`` is the small "IR access" surface a
downstream package may read; everything else may change between
releases. :mod:`hawk` itself must never import it — consumers read
``Walk.arg_spec``/``Walk.slot_of``, never enumerate ``leaves``/``order``."""

# Internal names for hawk's own modules/tests (`X as X` = not on `__all__`).
from ..types import TensorType as TensorType
from ..types import Wire as Wire
from .access import Access as Access
from .access import infer
from .compound import Quantity as Quantity
from .compound import QuantitySpan as QuantitySpan
from .contraction import ContractionOperand as ContractionOperand
from .contraction import RecognizedContraction as RecognizedContraction
from .contraction import recognize
from .lanes import KeyRun as KeyRun
from .lanes import LaneFacts as LaneFacts
from .lanes import key_run as key_run
from .loop_nodes import Loop as Loop
from .loop_nodes import LoopCarry as LoopCarry
from .loop_nodes import LoopCount as LoopCount
from .loop_nodes import LoopIndex as LoopIndex
from .loop_nodes import LoopValue as LoopValue
from .loop_nodes import TapeRead as TapeRead
from .loops import ACCUMULATOR as ACCUMULATOR
from .loops import RECURRENCE as RECURRENCE
from .loops import LoopClass as LoopClass
from .loops import build_loop as build_loop
from .loops import classify_loop as classify_loop
from .nodes import DISPATCH_KIND as DISPATCH_KIND
from .nodes import DISPATCH_POLICIES as DISPATCH_POLICIES
from .nodes import AccumWrite, Assign, At, Const, Op
from .nodes import Dispatch as Dispatch
from .nodes import HawkError as HawkError
from .nodes import Leaf as Leaf
from .nodes import Node as Node
from .nodes import Primitive as Primitive
from .nodes import RoleConst as RoleConst
from .nodes import SampleIndex as SampleIndex
from .nodes import SampleSum as SampleSum
from .nodes import Select as Select
from .nodes import Sink as Sink
from .nodes import TapeWrite as TapeWrite
from .nodes import WideWrite as WideWrite
from .ops import OP_KINDS as OP_KINDS
from .ops import make
from .ops import result_type as result_type
from .segment import OFFSETS_DTYPE as OFFSETS_DTYPE
from .segment import OFFSETS_ROLE as OFFSETS_ROLE
from .segment import SegmentInfo as SegmentInfo
from .segment import ClassifyUnit as ClassifyUnit
from .segment import SegmentUnit as SegmentUnit
from .segment import classify_name as classify_name
from .segment import split_segmented as split_segmented
from .cost import BranchCost as BranchCost
from .cost import DispatchCost as DispatchCost
from .cost import dispatch_costs as dispatch_costs
from .walk import DispatchInfo as DispatchInfo
from .walk import Walk as Walk
from .walk import canonical
from .walk import canonical_nodes as canonical_nodes
from .walk import static_selectors as static_selectors

__all__ = [
    "AccumWrite", "Assign", "At", "Const", "Op", "canonical", "infer", "make",
    "recognize",
]
