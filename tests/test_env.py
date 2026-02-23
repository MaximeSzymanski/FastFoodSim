from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src import config
from src.rl.FastFoodEnv import FastFoodEnv

# --- FIXTURES ---


@pytest.fixture
def rl_env():
    """Provides a fresh, unstepped FastFoodEnv instance."""
    env = FastFoodEnv()
    env.reset()
    return env


# --- TESTS: GYM API COMPLIANCE ---


def test_environment_spaces(rl_env):
    """Test that the action and observation spaces are defined correctly for MultiDiscrete."""
    # Action space should be MultiDiscrete([2, 2, 2])
    assert rl_env.action_space.shape == (3,)
    assert (rl_env.action_space.nvec == [2, 2, 2]).all()

    # Observation space should now be Box(9,)
    assert rl_env.observation_space.shape == (9,)
    assert (rl_env.observation_space.low == 0.0).all()
    assert (rl_env.observation_space.high == 1.0).all()


def test_reset_function(rl_env):
    """Test that reset returns the correct shapes and starts time at 0."""
    obs, info = rl_env.reset()

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (9,)
    assert isinstance(info, dict)
    assert rl_env.env.now == 0


# --- TESTS: CUSTOM RL LOGIC ---


def test_action_masks(rl_env):
    """Test that the environment calculates valid actions for all 3 stations."""

    # Scenario 1: All cooks available
    mask = rl_env.action_masks()
    # Format: [No B, Cook B, No F, Cook F, No I, Cook I]
    expected = [True, True, True, True, True, True]
    np.testing.assert_array_equal(mask, expected)

    # Scenario 2: Burger cooks busy
    reqs_burger = [
        rl_env.restaurant.burger_cook.request() for _ in range(rl_env.num_burger_cooks)
    ]
    mask = rl_env.action_masks()
    expected = [True, False, True, True, True, True]
    np.testing.assert_array_equal(mask, expected)

    # Scenario 3: Everyone busy
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
    """Test that taking a step pushes the simulation forward exactly 10 units."""
    initial_time = rl_env.env.now
    # Action [0, 0, 0] = Do Nothing
    rl_env.step([0, 0, 0])
    assert rl_env.env.now == initial_time + 10


@patch("src.rl.FastFoodEnv.COST_WASTED_BURGER", 2.0)
@patch("src.rl.FastFoodEnv.COST_WASTED_FRIES", 1.0)
@patch("src.rl.FastFoodEnv.COST_WASTED_ICE_CREAM", 1.5)
@patch("src.rl.FastFoodEnv.TARGET_BURGER_INV", 10)
@patch("src.rl.FastFoodEnv.TARGET_FRIES_INV", 10)
@patch("src.rl.FastFoodEnv.TARGET_ICE_CREAM_INV", 10)
def test_step_reward_calculation(rl_env):
    """Test reward math including deltas and shaping for all 3 items."""
    # 1. Setup baseline
    rl_env.last_revenue = 10.0
    rl_env.last_waste_cost = 2.0
    rl_env.last_lost_revenue = 0.0

    # 2. Inject stats
    rl_env.stats["captured_revenue"] = [10.0, 10.0]  # Total 20 (Delta 10)
    rl_env.stats["wasted_burgers"] = [1]  # Delta +1 (Cost 2.0)
    rl_env.stats["wasted_fries"] = [1]  # Delta +1 (Cost 1.0)
    rl_env.stats["wasted_ice_cream"] = [1]  # Delta +1 (Cost 1.5)
    rl_env.stats["lost_revenue"] = [5.0]  # Delta 5.0

    # 3. Stock shelves to avoid stockout penalties
    rl_env.restaurant.burger_shelf.items = [1]
    rl_env.restaurant.fries_shelf.items = [1]
    rl_env.restaurant.ice_cream_shelf.items = [1]

    # 4. Take a step
    _, reward, _, _, _ = rl_env.step([0, 0, 0])

    # 5. Verify Math:
    # current_waste = (1*2.0) + (1*1.0) + (1*1.5) = 4.5
    # delta_waste = 4.5 - 2.0 = 2.5
    # financial = (10.0 - 2.5 - 5.0) / 10.0 = 0.25
    # shaping = 0.0 (shelves stocked, no line)
    # Total = 0.25

    assert reward == 0.25


@patch("src.rl.FastFoodEnv.SIM_TIME", 3600)
def test_termination(rl_env):
    """Test that the environment signals termination when the shift is over."""
    rl_env.env.run(until=3595)
    _, _, terminated, truncated, _ = rl_env.step([0, 0, 0])

    assert terminated is True
    assert truncated is False
