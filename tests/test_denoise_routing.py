import numpy as np

from app.core.sr_engine import DenoisePlan, OnnxEngine, resolve_denoise_plan
from app.core.types import Backend


class _PlainEngine:
    """Stand-in for engines that do not implement plan_denoise (e.g. test doubles)."""


class _PlannerEngine:
    def plan_denoise(self, strength: float, target_scale: int) -> DenoisePlan:
        return DenoisePlan(pre_strength=strength)


def _engine_with_denoiser(scale: int) -> OnnxEngine:
    engine = OnnxEngine(Backend.CPU)
    engine._pair_session = object()  # noqa: SLF001 - unit-level wiring
    engine._pair_scale = scale  # noqa: SLF001
    return engine


def test_resolve_denoise_plan_falls_back_to_builtin_filter():
    plan = resolve_denoise_plan(_PlainEngine(), 0.5, 4)

    assert plan == DenoisePlan(classical_strength=0.5)


def test_resolve_denoise_plan_prefers_engine_planner():
    plan = resolve_denoise_plan(_PlannerEngine(), 0.5, 4)

    assert plan.pre_strength == 0.5


def test_plan_denoise_with_zero_strength_does_nothing():
    engine = _engine_with_denoiser(1)

    assert engine.plan_denoise(0.0, 4) == DenoisePlan()


def test_plan_denoise_routes_scale_one_denoiser_to_pre_denoise():
    engine = _engine_with_denoiser(1)

    plan = engine.plan_denoise(0.6, 4)

    assert plan == DenoisePlan(pre_strength=0.6)


def test_plan_denoise_routes_matching_scale_to_post_blend():
    engine = _engine_with_denoiser(4)

    plan = engine.plan_denoise(0.5, 4)

    assert plan == DenoisePlan(post_strength=0.5)


def test_plan_denoise_falls_back_when_scales_disagree():
    engine = _engine_with_denoiser(4)

    plan = engine.plan_denoise(0.5, 2)

    assert plan == DenoisePlan(classical_strength=0.5)


def test_plan_denoise_without_denoiser_uses_builtin_filter():
    engine = OnnxEngine(Backend.CPU)

    assert engine.plan_denoise(0.4, 4) == DenoisePlan(classical_strength=0.4)


def test_plan_denoise_clamps_out_of_range_strength():
    engine = OnnxEngine(Backend.CPU)

    assert engine.plan_denoise(2.5, 4) == DenoisePlan(classical_strength=1.0)
    assert engine.plan_denoise(-1.0, 4) == DenoisePlan()


def test_pre_denoise_is_noop_without_a_scale_one_denoiser():
    image = np.full((12, 10, 3), 90, dtype=np.uint8)

    np.testing.assert_array_equal(OnnxEngine(Backend.CPU).pre_denoise(image, 0.5), image)
    np.testing.assert_array_equal(
        _engine_with_denoiser(4).pre_denoise(image, 0.5), image
    )
