"""Tests for ``pharmaml.ml.harness`` -- the ZeRO stages + EarlyStopping.

The lost round-188/189 engine rebuilt per ``scratch/round189_state_after_rollback.md``
§3: the ZeRO stage memory arithmetic (paper Table 3), the ceil-division
partitioner, the stage-3 gather/scatter protocol over the autograd tape,
and the EarlyStopping five-branch patience contract.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pharmaml.ml.autograd import Linear, Module, ReLU, Sequential
from pharmaml.ml.harness import (
    EarlyStopping,
    ShardedStep,
    TrainStepResult,
    ZeROConfig,
    ZeROMemory,
    partition_flat,
    shard_plan,
    sharded_step,
    zero_memory_model,
    zero_stage_description,
)

# -- ZeROConfig ---------------------------------------------------------------- #


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_config_accepts_valid_stages(stage: int) -> None:
    assert ZeROConfig(stage=stage).stage == stage


@pytest.mark.parametrize("stage", [-1, 4, 10])
def test_config_rejects_invalid_stages(stage: int) -> None:
    with pytest.raises(ValueError):
        ZeROConfig(stage=stage)


@pytest.mark.parametrize("world", [0, -2])
def test_config_rejects_invalid_world_size(world: int) -> None:
    with pytest.raises(ValueError):
        ZeROConfig(world_size=world)


def test_stage_descriptions_exist_for_all_stages() -> None:
    for stage in range(4):
        assert f"stage {stage}" in zero_stage_description(stage)


def test_stage_description_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        zero_stage_description(5)


# -- the memory model -------------------------------------------------------------- #


def test_stage0_replicates_everything() -> None:
    mem = zero_memory_model(1000, world_size=4, stage=0)
    # 16 bytes per param: 2 (p) + 2 (g) + 12 (adam states)
    assert mem.per_rank_bytes == 16000
    assert mem.total_bytes == 64000


def test_stage1_partitions_optimizer_only() -> None:
    mem = zero_memory_model(1000, world_size=4, stage=1)
    assert mem.params_bytes == 2000
    assert mem.grads_bytes == 2000
    assert mem.optimizer_bytes == 3000  # ceil(12 * 1000 / 4)
    assert mem.per_rank_bytes == 7000


def test_stage2_partitions_grads_too() -> None:
    mem = zero_memory_model(1000, world_size=4, stage=2)
    assert mem.grads_bytes == 500
    assert mem.optimizer_bytes == 3000
    assert mem.per_rank_bytes == 5500


def test_stage3_partitions_params_too() -> None:
    mem = zero_memory_model(1_000_000, world_size=4, stage=3)
    assert mem.per_rank_bytes == 4_000_000


def test_memory_savings_grow_monotone_with_stage() -> None:
    per_rank = [
        zero_memory_model(10_000, world_size=8, stage=s).per_rank_bytes
        for s in range(4)
    ]
    assert per_rank == sorted(per_rank, reverse=True)


def test_world_size_one_degenerates_to_replicated() -> None:
    for stage in range(4):
        mem = zero_memory_model(100, world_size=1, stage=stage)
        assert mem.per_rank_bytes == 1600


def test_non_divisible_shard_counts_use_ceil() -> None:
    mem = zero_memory_model(1001, world_size=4, stage=3)
    # each bucket: ceil(bytes * 1001 / 4)
    assert mem.params_bytes == math.ceil(2 * 1001 / 4)
    assert mem.optimizer_bytes == math.ceil(12 * 1001 / 4)


def test_memory_as_dict_fields() -> None:
    payload = zero_memory_model(10, world_size=2, stage=2).as_dict()
    assert set(payload) == {
        "stage", "world_size", "n_params",
        "params_bytes_per_rank", "grads_bytes_per_rank",
        "optimizer_bytes_per_rank", "per_rank_bytes",
    }


def test_memory_rejects_negative_params() -> None:
    with pytest.raises(ValueError):
        zero_memory_model(-1)


def test_zero_params_is_zero_memory() -> None:
    assert zero_memory_model(0, world_size=4, stage=3).per_rank_bytes == 0


def test_memory_dataclass_total() -> None:
    mem: ZeROMemory = zero_memory_model(10, world_size=3, stage=1)
    assert mem.total_bytes == mem.per_rank_bytes * 3


# -- the partitioner ------------------------------------------------------------------ #


def test_partition_covers_every_element_exactly_once() -> None:
    n, world = 10, 3
    covered: list[int] = []
    for rank in range(world):
        lo, hi = partition_flat(n, world, rank)
        covered.extend(range(lo, hi))
    assert covered == list(range(n))


def test_partition_shards_differ_by_at_most_one() -> None:
    n, world = 100, 7
    sizes = [partition_flat(n, world, r)[1] - partition_flat(n, world, r)[0]
             for r in range(world)]
    assert max(sizes) - min(sizes) <= 1


def test_partition_remainder_goes_to_early_ranks() -> None:
    assert partition_flat(10, 3, 0) == (0, 4)
    assert partition_flat(10, 3, 1) == (4, 7)
    assert partition_flat(10, 3, 2) == (7, 10)


def test_partition_exact_multiple() -> None:
    assert partition_flat(9, 3, 2) == (6, 9)


def test_partition_single_rank_owns_all() -> None:
    assert partition_flat(5, 1, 0) == (0, 5)


def test_partition_empty_tensor() -> None:
    for rank in range(3):
        assert partition_flat(0, 3, rank) == (0, 0)


def test_partition_rejects_bad_args() -> None:
    with pytest.raises(ValueError):
        partition_flat(10, 0, 0)
    with pytest.raises(ValueError):
        partition_flat(10, 3, 3)
    with pytest.raises(ValueError):
        partition_flat(-1, 3, 0)


# -- the shard plan ---------------------------------------------------------------------- #


def _two_layer() -> Sequential:
    return Sequential(Linear(4, 3, seed=1), ReLU(), Linear(3, 2, seed=2))


def test_plan_stage0_is_replicated() -> None:
    # 2 Linears x (weight + bias) = 4 tape parameters
    plan = shard_plan(_two_layer().parameters(), ZeROConfig(stage=0))
    assert len(plan) == 4
    assert all(entry["owner"] == "all ranks (replicated)" for entry in plan)
    assert all(entry["moves"] == [] for entry in plan)


def test_plan_stage3_partitions_and_moves() -> None:
    plan = shard_plan(
        _two_layer().parameters(), ZeROConfig(stage=3, world_size=2)
    )
    total_owners = sum(len(entry["owners"]) for entry in plan)
    assert total_owners == 8  # four params x two ranks (shards non-empty)
    for entry in plan:
        assert "all-gather before forward" in entry["moves"]
        assert "reduce-scatter after backward" in entry["moves"]


def test_plan_stage2_has_no_param_partition() -> None:
    plan = shard_plan(
        _two_layer().parameters(), ZeROConfig(stage=2, world_size=2)
    )
    for entry in plan:
        for owner in entry["owners"]:
            assert owner["params"] is False
            assert owner["grads"] is True


def test_plan_flat_offsets_are_contiguous() -> None:
    params = _two_layer().parameters()
    plan = shard_plan(params, ZeROConfig(stage=1, world_size=2))
    offsets = [entry["flat_offset"] for entry in plan]
    sizes = [entry["size"] for entry in plan]
    assert offsets[1] == offsets[0] + sizes[0]


# -- the stage-3 step protocol --------------------------------------------------------------- #


def test_sharded_step_stage0_runs_forward() -> None:
    net = _two_layer()
    calls: list[int] = []

    def fb(module: Module) -> float:
        calls.append(1)
        return 0.7

    result = sharded_step(net, fb, ZeROConfig(stage=0))
    assert result.loss == 0.7
    assert result.gathered == []
    assert result.scattered == []
    assert result.updated == 0


def test_sharded_step_stage3_gathers_every_param() -> None:
    net = _two_layer()

    def fb(module: Module) -> float:
        return 0.5

    result = sharded_step(net, fb, ZeROConfig(stage=3, world_size=2))
    assert result.gathered == [0, 1, 2, 3]  # weight+bias of both Linears
    assert result.scattered == [0, 1, 2, 3]
    assert result.updated == 4
    assert result.stage == 3


def test_sharded_step_counts_steps() -> None:
    net = _two_layer()
    trainer = ShardedStep(net, ZeROConfig(stage=1))
    trainer.step(lambda m: 1.0)
    trainer.step(lambda m: 0.5)
    assert trainer.n_steps == 2


def test_sharded_step_result_as_dict() -> None:
    net = _two_layer()
    result = sharded_step(net, lambda m: 0.25, ZeROConfig(stage=3, world_size=1))
    payload = result.as_dict()
    assert isinstance(result, TrainStepResult)
    assert payload["loss"] == 0.25
    assert payload["stage"] == 3


def test_stage3_leaves_parameters_intact() -> None:
    net = _two_layer()
    before = [np.asarray(p.data).copy() for p in net.parameters()]

    def fb(module: Module) -> float:
        return 0.0

    sharded_step(net, fb, ZeROConfig(stage=3, world_size=4))
    after = [np.asarray(p.data) for p in net.parameters()]
    for b, a in zip(before, after):
        assert np.array_equal(b, a)


def test_stage1_2_do_not_simulate_shards() -> None:
    net = _two_layer()

    def fb(module: Module) -> float:
        return 1.0

    r1 = sharded_step(net, fb, ZeROConfig(stage=1))
    r2 = sharded_step(net, fb, ZeROConfig(stage=2))
    assert r1.scattered == [] and r1.updated == 4
    assert r2.scattered == [0, 1, 2, 3] and r2.updated == 4


# -- EarlyStopping: the five branches ----------------------------------------------------------- #


def test_branch_improve() -> None:
    stop = EarlyStopping(patience=2)
    assert stop.step(1.0)
    assert stop.last_branch == "improve"
    assert stop.best == 1.0
    assert stop.bad_epochs == 0


def test_branch_tick() -> None:
    stop = EarlyStopping(patience=2)
    stop.step(1.0)
    assert stop.step(1.2)
    assert stop.last_branch == "tick"
    assert stop.bad_epochs == 1


def test_branch_stop_without_snapshot() -> None:
    stop = EarlyStopping(patience=0, restore_best=True)
    assert stop.step(1.0)  # improve
    assert not stop.step(1.1)  # patience 0 -> immediate stop
    assert stop.last_branch == "stop"
    assert stop.stopped


def test_branch_restore_with_module() -> None:
    net = _two_layer()
    stop = EarlyStopping(patience=1, restore_best=True)
    assert stop.step(1.0, module=net)  # improve + snapshot
    drift = [np.asarray(p.data) + 5.0 for p in net.parameters()]
    for p, d in zip(net.parameters(), drift):
        p.data = d
    assert stop.step(1.5, module=net)  # tick (patience 1 still left)
    assert not stop.step(1.4, module=net)  # patience spent -> restore
    assert stop.last_branch == "restore"
    assert stop.restored
    for p, d in zip(net.parameters(), drift):
        assert not np.array_equal(np.asarray(p.data), d)  # back to best


def test_branch_plateau_min_delta() -> None:
    stop = EarlyStopping(patience=1, min_delta=0.5, restore_best=False)
    stop.step(1.0)
    stop.step(1.2)  # tick (miss by 0.2 < 0.5 but patience left)
    assert not stop.step(1.3)  # miss within noise -> plateau
    assert stop.last_branch == "plateau"
    assert stop.plateaued


def test_after_stop_every_step_is_stop() -> None:
    stop = EarlyStopping(patience=0)
    stop.step(1.0)
    stop.step(1.1)
    assert not stop.step(0.5)
    assert stop.last_branch == "stop"


def test_min_delta_must_be_earned() -> None:
    stop = EarlyStopping(patience=3, min_delta=0.1)
    assert stop.step(1.0)
    assert stop.step(0.95)  # 0.05 < min_delta -> NOT an improvement, tick
    assert stop.last_branch == "tick"
    assert stop.best == 1.0


def test_mode_max_treats_bigger_as_better() -> None:
    stop = EarlyStopping(mode="max", patience=1)
    assert stop.step(0.5)
    assert stop.best == 0.5
    assert stop.step(0.9)
    assert stop.best == 0.9
    assert stop.step(0.7)
    assert not stop.step(0.6)


def test_improved_epochs_log() -> None:
    stop = EarlyStopping(patience=5)
    for v in (1.0, 0.9, 0.95, 0.8):
        stop.step(v)
    assert len(stop.improved_epochs) == 3  # 1.0, 0.9, 0.8


@pytest.mark.parametrize("patience", [-1])
def test_patience_validation(patience: int) -> None:
    with pytest.raises(ValueError):
        EarlyStopping(patience=patience)


def test_min_delta_validation() -> None:
    with pytest.raises(ValueError):
        EarlyStopping(min_delta=-0.1)


def test_mode_validation() -> None:
    with pytest.raises(ValueError):
        EarlyStopping(mode="auto")


def test_snapshot_and_restore_round_trip() -> None:
    net = _two_layer()
    stop = EarlyStopping()
    stop.snapshot(net)
    original = [np.asarray(p.data).copy() for p in net.parameters()]
    for p in net.parameters():
        p.data = np.asarray(p.data) + 100.0
    assert stop.restore(net)
    for p, o in zip(net.parameters(), original):
        assert np.array_equal(np.asarray(p.data), o)


def test_restore_without_snapshot_returns_false() -> None:
    stop = EarlyStopping()
    assert stop.restore(_two_layer()) is False
    assert stop.restored is False


def test_classic_walk() -> None:
    stop = EarlyStopping(patience=2, min_delta=0.0)
    walk = [stop.step(v) for v in (1.0, 0.9, 0.95, 0.96, 0.97)]
    assert walk == [True, True, True, True, False]
    assert round(stop.best, 2) == 0.9


def test_first_value_always_improves_min() -> None:
    stop = EarlyStopping(mode="min")
    assert stop.step(100.0)
    assert stop.best == 100.0


def test_first_value_always_improves_max() -> None:
    stop = EarlyStopping(mode="max")
    assert stop.step(-100.0)
    assert stop.best == -100.0
