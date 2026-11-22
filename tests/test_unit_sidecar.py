# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The sidecar of a unit the lowering library answered, against a written-out sidecar for each kind of unit.

The units are lowered by the library for real, so the tests skip where the library or its binding is not at hand. The two include
paths of a sidecar are machine values and are left out of the comparison. The written sidecars are the ones the old build wrote for
the same units (the comparison tool in the build's tools folder checks every unit of the suite against them)."""

from __future__ import annotations

import pytest

import hawk
from hawk import Grid, Mutable, Param, Scalar, Terminated, Vector, kernel
from _unit_sidecar import lowered, sidecar_of_kernel
from hawk.ext import Guard, Kind

pytest.importorskip("_osprey")
try:
    from hawk import ospreys
    ospreys.serving("host")
except Exception as why:                 # no host OSPREY at hand
    pytest.skip(f"the host OSPREY is not available: {why}", allow_module_level=True)

import test_dispatch_mix_and_guard as DM
import test_split_stages as S
from test_w25_keyed_sum_over_limit import ROWS


@kernel
def axpb(x: Scalar, y: Mutable[Scalar]):
    y = 2.0 * x + 1.0


@kernel
def masked(x: Scalar, terminated: Terminated, y: Mutable[Scalar]):
    y = x + 1.0


@kernel(steps=1)
def stepping_one(dt: Param, t_final: Scalar, terminated: Terminated, x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    v_start = v
    v = v_start - dt * x
    x = x + dt * v_start
    tn = t + dt
    t = tn
    terminated = tn >= t_final


N = 4100


@kernel
def ordered_sum(lane: Grid["key", "row":ROWS], j: Vector[N], r: Scalar) -> Vector[N]:
    return hawk.sum(j * r, over="row", contract="ordered")


EXPECTED = {'active_set': {'slots': ['y', 'x', 'active_count', 'active_map', 'terminated'],
'zeroed_outputs': [],
                'aether_abi': 'aether-abi/2',
                'arg_dtypes': {'active_count': 'uint32', 'active_map': 'int32', 'terminated': 'bool', 'x': 'float64', 'y': 'float64'},
                'arg_mirrors': {'y': 'ScalarHandle', 'x': 'ScalarHandle', 'active_count': 'ScalarHandle', 'active_map': 'ScalarHandle', 'terminated': 'ScalarHandle'},
                'arg_shapes': {'active_count': None, 'active_map': None, 'terminated': 1, 'x': 1, 'y': 1},
                'arg_spec': [['mutable', 'y'],
                             ['plane_in', 'x'],
                             ['lookup', 'active_count'],
                             ['lookup', 'active_map'],
                             ['terminated', 'terminated']],
                'arg_widths': {'terminated': 1, 'x': 1, 'y': 1},
                'buffers': [],
                'digest': '6317d7db58421704180d2e80e8765d000dff1dbac294c32114f013421364fe48',
                'planes_written': ['y'],  # masked never stores to its mask, so the mask is read, not written
                'slot_shapes': {'x': [], 'y': [], 'active_count': [], 'active_map': [], 'terminated': []},
                'dispatch': [],
                'exec_access': 'sample_local',
                'exec_targets': ['host'],
                'format': 'host',
                'host_entry': 'masked_host',
                'kernel': 'masked',
                'launch': {'lane_space': {'axes': [{'extent': None, 'label': None}], 'group': 1}},
                'mutables': [{'default': None, 'dtype': 'scalar', 'name': 'y', 'width': 1}],
                'params': [],
                'params_schema': 2,
                'pattern': 'pure',
                'scalar_type': 'float64',
                'schema_version': 3,
                'terminated_readonly': True,
                'wide_inputs': [],
                'plane_outputs': []},
 'finishing': {'slots': ['t', 'v', 'x', 't_final', 'finished_count', 'terminated', 'dt'],
'zeroed_outputs': [],
               'aether_abi': 'aether-abi/2',
               'arg_dtypes': {'finished_count': 'uint32', 't': 'float64', 't_final': 'float64', 'terminated': 'bool', 'v': 'float64', 'x': 'float64', 'dt': 'float64'},
               'arg_mirrors': {'t': 'ScalarHandle', 'v': 'ScalarHandle', 'x': 'ScalarHandle', 't_final': 'ScalarHandle', 'finished_count': 'ScalarHandle', 'terminated': 'ScalarHandle', 'dt': 'value'},
               'arg_shapes': {'finished_count': None, 't': 1, 't_final': 1, 'terminated': 1, 'v': 1, 'x': 1},
               'arg_spec': [['mutable', 't'],
                            ['mutable', 'v'],
                            ['mutable', 'x'],
                            ['plane_in', 't_final'],
                            ['lookup', 'finished_count'],
                            ['terminated', 'terminated'],
                            ['uniform', 'dt']],
               'arg_widths': {'t': 1, 't_final': 1, 'terminated': 1, 'v': 1, 'x': 1},
               'buffers': [],
               'digest': '3939f9e6858480ddc19a967fe0860c2e1fbdfdb5db36355b287dd47a34dee0d5',
               'planes_written': ['t', 'terminated', 'v', 'x'],
               'slot_shapes': {'t': [], 'v': [], 'x': [], 't_final': [], 'finished_count': [], 'terminated': [], 'dt': []},
               'dispatch': [],
               'exec_access': 'sample_local',
               'exec_targets': ['host'],
               'finish': {'counter': 'finished_count', 'mask': 'terminated', 'steps': 1},
               'format': 'host',
               'host_entry': 'stepping_one_host',
               'kernel': 'stepping_one',
               'launch': {'lane_space': {'axes': [{'extent': None, 'label': None}], 'group': 1}},
               'mutables': [{'default': None, 'dtype': 'scalar', 'name': 't', 'prior': True, 'width': 1},
                            {'default': None, 'dtype': 'scalar', 'name': 'v', 'prior': True, 'width': 1},
                            {'default': None, 'dtype': 'scalar', 'name': 'x', 'prior': True, 'width': 1}],
               'params': [{'dtype': 'float', 'name': 'dt'}],
               'params_schema': 2,
               'pattern': 'pure',
               'scalar_type': 'float64',
               'schema_version': 3,
               'terminated_readonly': False,
               'wide_inputs': [],
               'plane_outputs': []},
 'plain': {'slots': ['y', 'x'],
'zeroed_outputs': [],
           'aether_abi': 'aether-abi/2',
           'arg_dtypes': {'x': 'float64', 'y': 'float64'},
           'arg_mirrors': {'y': 'ScalarHandle', 'x': 'ScalarHandle'},
           'arg_shapes': {'x': 1, 'y': 1},
           'arg_spec': [['mutable', 'y'], ['plane_in', 'x']],
           'arg_widths': {'x': 1, 'y': 1},
           'buffers': [],
           'digest': '3d95771edbfd5108d9427eb5d1918ab586ec6095fcb700b37619601d772b3cbc',
           'planes_written': ['y'],
           'slot_shapes': {'x': [], 'y': []},
           'dispatch': [],
           'exec_access': 'sample_local',
           'exec_targets': ['host'],
           'format': 'host',
           'host_entry': 'axpb_host',
           'kernel': 'axpb',
           'launch': {'lane_space': {'axes': [{'extent': None, 'label': None}], 'group': 1}},
           'mutables': [{'default': None, 'dtype': 'scalar', 'name': 'y', 'width': 1}],
           'params': [],
           'params_schema': 2,
           'pattern': 'pure',
           'scalar_type': 'float64',
           'schema_version': 3,
           'terminated_readonly': True,
           'wide_inputs': [],
           'plane_outputs': []},
 'segment_branch': {'slots': ['x', 'bs', 'dispatch0_offsets', 'finished_count', 'terminated'],
'zeroed_outputs': [],
                    'aether_abi': 'aether-abi/2',
                    'arg_dtypes': {'bs': 'float64', 'dispatch0_offsets': 'int64', 'finished_count': 'uint32', 'terminated': 'bool', 'x': 'float64'},
                    'arg_mirrors': {'x': 'ScalarHandle', 'bs': 'GRefMirror', 'dispatch0_offsets': 'ScalarHandle', 'finished_count': 'ScalarHandle', 'terminated': 'ScalarHandle'},
                    'arg_shapes': {'bs': 3, 'dispatch0_offsets': None, 'finished_count': None, 'terminated': 1, 'x': 1},
                    'arg_spec': [['mutable', 'x'],
                                 ['plane_in', 'bs'],
                                 ['lookup', 'dispatch0_offsets'],
                                 ['lookup', 'finished_count'],
                                 ['terminated', 'terminated']],
                    'arg_widths': {'bs': 3, 'terminated': 1, 'x': 1},
                    'buffers': [],
                    'digest': 'c5ddf7219b53de9f1fb9db998fe97f775e224f772495ddaf0331fcb2b14efaec',
                    'planes_written': ['terminated', 'x'],
                    'slot_shapes': {'x': [], 'bs': [3], 'dispatch0_offsets': [], 'finished_count': [], 'terminated': []},
                    'dispatch': [{'group_units': [{'name': 'walker__k0', 'digest': 'c5ddf7219b53de9f1fb9db998fe97f775e224f772495ddaf0331fcb2b14efaec'}, {'name': 'walker__k1', 'digest': 'ea6bc2020b2d0c24dd7ace8f61c39e534dd49c77c8ff449f4850376a8e671fe6'}, {'name': 'walker__k2', 'digest': 'c992a01001bbc27a4567f63bcb7e51c00c1e7beb02739716211bf1e336c92091'}, {'name': 'walker__cls', 'digest': '36c768a17cb08e743f8d50e9c961cc396f3081a8b957c1612de92450663f6c1e'}],
'K': 3,
                                  'class': 'dispatch0_class',
                                  'classify': 'walker__cls',
                                  'name': 'dispatch0',
                                  'offsets': 'dispatch0_offsets',
                                  'selector_reads': [['plane_in', 'k']],
                                  'stale': 'dispatch0_stale',
                                  'static_selector': True,
                                  'strategy': 'segmented',
                                  'units': ['walker__k0', 'walker__k1', 'walker__k2']}],
                    'exec_access': 'sample_local',
                    'exec_targets': ['host'],
                    'finish': {'counter': 'finished_count', 'mask': 'terminated', 'steps': 1},
                    'format': 'host',
                    'host_entry': 'walker__k0_host',
                    'kernel': 'walker__k0',
                    'launch': {'lane_space': {'axes': [{'extent': None, 'label': None}], 'group': 1}},
                    'mutables': [{'default': None, 'dtype': 'scalar', 'name': 'x', 'prior': True, 'width': 1}],
                    'params': [],
                    'params_schema': 2,
                    'pattern': 'pure',
                    'scalar_type': 'float64',
                    'schema_version': 3,
                    'terminated_readonly': False,
                    'wide_inputs': [],
                    'plane_outputs': []},
 'segment_classify': {'slots': ['dispatch0_class', 'k'],
'zeroed_outputs': [],
                      'aether_abi': 'aether-abi/2',
                      'arg_dtypes': {'dispatch0_class': 'int64', 'k': 'int32'},
                      'arg_mirrors': {'dispatch0_class': 'ScalarHandle', 'k': 'ScalarHandle'},
                      'arg_shapes': {'dispatch0_class': 1, 'k': 1},
                      'arg_spec': [['mutable', 'dispatch0_class'], ['plane_in', 'k']],
                      'arg_widths': {'dispatch0_class': 1, 'k': 1},
                      'buffers': [],
                      'digest': '36c768a17cb08e743f8d50e9c961cc396f3081a8b957c1612de92450663f6c1e',
                      'planes_written': ['dispatch0_class'],
                      'slot_shapes': {'dispatch0_class': [], 'k': []},
                      'dispatch': [],
                      'exec_access': 'sample_local',
                      'exec_targets': ['host'],
                      'format': 'host',
                      'host_entry': 'walker__cls_host',
                      'index_planes': ['k'],
                      'kernel': 'walker__cls',
                      'launch': {'lane_space': {'axes': [{'extent': None, 'label': None}], 'group': 1}},
                      'mutables': [{'default': None, 'dtype': 'int', 'name': 'dispatch0_class', 'width': 1}],
                      'params': [],
                      'params_schema': 2,
                      'pattern': 'pure',
                      'scalar_type': 'float64',
                      'schema_version': 3,
                      'terminated_readonly': True,
                      'wide_inputs': [],
                      'plane_outputs': []},
 'split_stage': {'source': {'digest': 'dae7e4259c93bbc6e5a5cdc638506027f80c8620e4237f040d129f09c6f9e859', 'name': 'standard_source', 'slots': ['z', 'mu', 'mu_comp', 't', 't_comp', 'x', 'n_samples']},
'slots': ['mu', 'mu_comp', 'x', 'n_samples'],
'zeroed_outputs': ['mu', 'mu_comp'],
                 'aether_abi': 'aether-abi/2',
                 'arg_dtypes': {'mu': 'float64', 'mu_comp': 'float64', 'x': 'float64', 'n_samples': 'uint32'},
                 'arg_mirrors': {'mu': 'ScalarHandle', 'mu_comp': 'ScalarHandle', 'x': 'ScalarHandle', 'n_samples': 'value'},
                 'arg_shapes': {'mu': None, 'mu_comp': None, 'x': 1},
                 'arg_spec': [['plane_out', 'mu'], ['plane_out', 'mu_comp'], ['plane_in', 'x'], ['nsamples', 'n_samples']],
                 'arg_widths': {'x': 1},
                 'buffers': [],
                 'cell_row': 32,
                 'cells': {'cut0': {'columns': 1, 'internal': True, 'op': 'sum'},
                           'mu': {'columns': 1, 'op': 'sum'},
                           't': {'columns': 1, 'op': 'sum'}},
                 'digest': '98d1aafaa37467ce61fce84c586f12c59a9df39e0242ca1950d4adbf7d5f88e1',
                 'planes_written': ['mu', 'mu_comp'],
                 'slot_shapes': {'mu': [], 'mu_comp': [], 'x': [], 'n_samples': []},
                 'dispatch': [],
                 'exec_access': 'sample_local',
                 'exec_targets': ['host'],
                 'format': 'host',
                 'host_entry': 'standard_host',
                 'kernel': 'standard',
                 'launch': {'lane_space': {'axes': [{'extent': None, 'label': None}], 'group': 1}},
                 'mutables': [],
                 'params': [],
                 'params_schema': 2,
                 'pattern': 'pure',
                 'scalar_type': 'float64',
                 'schema_version': 3,
                 'stages': [{'cells_in': [], 'cells_out': ['mu'], 'kernel': 'standard'},
                            {'cells_in': ['mu'], 'cells_out': ['cut0'], 'kernel': 'standard_2'},
                            {'cells_in': ['mu', 'cut0'], 'cells_out': ['t'], 'kernel': 'standard_3'}],
                 'terminated_readonly': True,
                 'unused_planes': ['z', 't'],
                 'wide_inputs': [],
                 'plane_outputs': ['mu', 'mu_comp']},
 'team_stage': {'slots': ['tms_out_m', 'tms_out_p'],
'zeroed_outputs': [],
                'aether_abi': 'aether-abi/2',
                'arg_dtypes': {'tms_out_m': 'int32', 'tms_out_p': 'float64'},
                'arg_mirrors': {'tms_out_m': 'ScalarHandle', 'tms_out_p': 'ScalarHandle'},
                'arg_shapes': {'tms_out_m': None, 'tms_out_p': None},
                'arg_spec': [['plane_out', 'tms_out_m'], ['plane_out', 'tms_out_p']],
                'arg_widths': {},
                'buffers': [],
                'digest': 'e56bbc4c7766b7b5dce155f928f8456c081f0dc2c44865f6bbe106ff962a12ff',
                'planes_written': ['tms_out_m', 'tms_out_p'],
                'slot_shapes': {'tms_out_m': [], 'tms_out_p': []},
                'dispatch': [],
                'exec_access': 'sample_local',
                'exec_targets': ['host'],
                'format': 'host',
                'host_entry': 'ordered_sum_2_host',
                'kernel': 'ordered_sum_2',
                'launch': {'lane_space': {'axes': [{'extent': None, 'label': None}], 'group': 1}},
                'mutables': [],
                'params': [],
                'params_schema': 2,
                'pattern': 'pure',
                'scalar_type': 'float64',
                'scatter_chunks': 256,
                'schema_version': 3,
                'terminated_readonly': True,
                'wide_inputs': [],
                'plane_outputs': ['tms_out_m', 'tms_out_p']}}


def written_by_the_library(target_kernel, unit_name, kind=None):
    """The sidecar of unit unit_name of the kernel's answer."""
    return sidecar_of_kernel(target_kernel, unit_name, kind=kind)


def test_a_plain_unit():
    assert written_by_the_library(axpb, "axpb") == EXPECTED["plain"]


def test_the_first_stage_of_a_split_kernel_carries_every_stage_and_cell():
    assert written_by_the_library(S.standard, "standard") == EXPECTED["split_stage"]


def test_a_team_stage_after_the_first_declares_no_plane_of_the_kernel():
    assert written_by_the_library(ordered_sum, "ordered_sum_2") == EXPECTED["team_stage"]


def test_a_branch_unit_of_a_segmented_split_lists_its_group():
    walker = hawk.resolve_dispatch(DM.walker, {"dispatch0": "segmented"})
    assert written_by_the_library(walker, "walker__k0") == EXPECTED["segment_branch"]


def test_the_classify_unit_of_a_segmented_split():
    walker = hawk.resolve_dispatch(DM.walker, {"dispatch0": "segmented"})
    assert written_by_the_library(walker, "walker__cls") == EXPECTED["segment_classify"]


def test_an_active_set_unit():
    assert written_by_the_library(masked, "masked", Kind("active_probe", guard=Guard(active_set=True))) == EXPECTED["active_set"]


def test_a_finishing_unit_takes_its_step_counts_from_the_finish_record():
    assert written_by_the_library(stepping_one, "stepping_one") == EXPECTED["finishing"]


@hawk.kernel
def add_vec(key: hawk.Index, v: hawk.Vector[3], acc: hawk.Mutable[hawk.Vector[3], "bin":9]):
    hawk.scatter(acc, contract="ordered")[key] += v


@hawk.kernel
def add_one(key: hawk.Index, x: hawk.Scalar, acc: hawk.Mutable[hawk.Scalar, "bin":9]):
    hawk.scatter(acc, contract="ordered")[key] += x


@pytest.mark.parametrize("target_kernel, shape", [(add_vec, [3]), (add_one, [])], ids=["vector", "scalar"])
def test_the_record_of_a_sink_with_named_axes_states_the_payload_as_leading_axes(target_kernel, shape):
    """The runtime binds the plane (payload, then named axes) from the record alone: the plane's axes are the payload's, then
    the declared ones."""
    units = [u["meta"] for u in lowered(target_kernel)["units"]]
    stated = [a for m in units for a in m["launch"]["plane_axes"] if m["body"]["slots"][a["slot"]]["name"] == "acc"]
    assert stated and all(a["extents"] == [*shape, 9] for a in stated)

