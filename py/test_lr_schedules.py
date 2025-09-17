from typing import Callable

import pytest

from lr_schedules import aiayn_lr, cosine_lr


def test_aiayn_lr_schedule_behaviour() -> None:
    d_model = 512
    warmup = 4000

    expected_step_one = (d_model**-0.5) * (1 * warmup**-1.5)
    assert aiayn_lr(1, d_model, warmup) == pytest.approx(expected_step_one, abs=1e-12)

    expected_warmup = (d_model**-0.5) * (warmup**-0.5)
    assert aiayn_lr(warmup, d_model, warmup) == pytest.approx(expected_warmup, abs=1e-12)

    assert aiayn_lr(100, d_model, warmup) < aiayn_lr(200, d_model, warmup)
    assert aiayn_lr(5000, d_model, warmup) < aiayn_lr(4000, d_model, warmup)

    with pytest.raises(ValueError):
        aiayn_lr(0, d_model, warmup)


@pytest.fixture(name="cosine_schedule")
def fixture_cosine_schedule() -> tuple[int, int, Callable[[int], float]]:
    total = 100
    warmup = 10
    return total, warmup, cosine_lr(total, warmup)


@pytest.mark.parametrize(
    "step, expected",
    [
        (0, 0.0),
        (10, 1.0),
        (100, 0.0),
    ],
)
def test_cosine_lr_boundaries(
    cosine_schedule: tuple[int, int, Callable[[int], float]], step: int, expected: float
) -> None:
    _, _, schedule = cosine_schedule

    assert schedule(step) == pytest.approx(expected, abs=1e-8)


def test_cosine_lr_midpoint(cosine_schedule: tuple[int, int, Callable[[int], float]]) -> None:
    total, warmup, schedule = cosine_schedule

    mid_step = (total + warmup) // 2
    lr_mid = schedule(mid_step)

    assert 0.49 <= lr_mid <= 0.51


@pytest.mark.parametrize(
    "total, warmup",
    [(-1, 0), (0, 0), (10, 11)],
)
def test_cosine_lr_invalid_inputs(total: int, warmup: int) -> None:
    with pytest.raises(ValueError):
        cosine_lr(total, warmup)
