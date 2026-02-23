from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src import config
from src.rl.FastFoodEnv import FastFoodEnv


@pytest.fixture
def rl_env():
    """Provides a freshly initialized and reset FastFoodEnv instance.

    Returns:
        FastFoodEnv: A clean environment ready for testing.
    """
    env = FastFoodEnv()
    env.reset()
    return env


def test_environment_spaces(rl_env):
    """Verifies that the action and observation spaces conform to expected shapes and bounds.

    Args:
        rl_env (FastFoodEnv): The test environment fixture.
    """
    assert rl_env.action_space.shape == (3,)
    assert (rl_env.action_space.nvec == [2, 2, 2]).all()

    assert rl_env.observation_space.shape == (12,)
    assert (rl_env.observation_space.low == 0.0).all()
    assert (rl_env.observation_space.high == 1.0).all()


def test_reset_function(rl_env):
    """Ensures the reset function correctly reinitializes the environment state and clock.

    Args:
        rl_env (FastFoodEnv): The test environment fixture.
    """
    obs, info = rl_env.reset()

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (12,)
    assert isinstance(info, dict)
    assert rl_env.env.now == 0


def test_action_masks(rl_env):
    """Validates that action masks correctly identify valid actions based on staff availability.

    Args:
        rl_env (FastFoodEnv): The test environment fixture.
    """
    mask = rl_env.action_masks()
    expected = [True, True, True, True, True, True]
    np.testing.assert_array_equal(mask, expected)

    reqs_burger = [
        rl_env.restaurant.burger_cook.request() for _ in range(rl_env.num_burger_cooks)
    ]
    mask = rl_env.action_masks()
    expected = [True, False, True, True, True, True]
    np.testing.assert_array_equal(mask, expected)

    reqs_fries = [
        rl_env.restaurant.fries_cook.request() for _ in range(rl_env.num_fries_cooks)
    ]
    reqs_ice = [
        rl_env.restaurant.ice_cream_cook.request()
        for _ in range(rl_env.num_ice_cream_cooks)
    ]
    mask = rl_env.action_masks()
    expected = [True, False, True, False, True, False]
    np.testing.assert_array_equal(mask, expected)


def test_step_advances_time(rl_env):
    """Checks that taking an environment step advances the internal simulation clock.

    Args:
        rl_env (FastFoodEnv): The test environment fixture.
    """
    initial_time = rl_env.env.now
    rl_env.step([0, 0, 0])
    assert rl_env.env.now == initial_time + 10


@patch("src.rl.FastFoodEnv.COST_WASTED_BURGER", 2.0)
@patch("src.rl.FastFoodEnv.COST_WASTED_FRIES", 1.0)
@patch("src.rl.FastFoodEnv.COST_WASTED_ICE_CREAM", 1.5)
def test_step_reward_calculation(rl_env):
    """Tests the comprehensive reward calculation incorporating financial deltas, reputation penalties, and service bonuses.

    Args:
        rl_env (FastFoodEnv): The test environment fixture.
    """
    rl_env.last_revenue = 10.0
    rl_env.last_waste_cost = 2.0
    rl_env.last_lost_revenue = 0.0
    rl_env.last_walkouts = 0
    rl_env.last_served = 1

    rl_env.stats["captured_revenue"] = [10.0, 10.0]
    rl_env.stats["wasted_burgers"] = [1]
    rl_env.stats["wasted_fries"] = [1]
    rl_env.stats["wasted_ice_cream"] = [1]
    rl_env.stats["lost_revenue"] = [5.0]
    rl_env.stats["balked"] = [1]

    _, reward, _, _, _ = rl_env.step([0, 0, 0])

    assert reward == -2.75


@patch("src.rl.FastFoodEnv.SIM_TIME", 3600)
def test_termination(rl_env):
    """Verifies that the environment triggers the termination signal at the end of the shift.

    Args:
        rl_env (FastFoodEnv): The test environment fixture.
    """
    rl_env.env.run(until=3595)
    _, _, terminated, truncated, _ = rl_env.step([0, 0, 0])

    assert terminated is True
    assert truncated is False
