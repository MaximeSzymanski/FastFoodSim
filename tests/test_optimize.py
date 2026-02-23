from unittest.mock import MagicMock, patch

import pytest

# Adjust this import to match the exact name of your file (e.g., src.optimization.optimize)
from src.optimization import optimize

# --- TESTS: PROFIT CALCULATION ---


@patch("src.optimization.optimize.config")
# We removed the patch for env.run so SimPy actually processes our fake events!
@patch("src.optimization.optimize.customer_arrivals")
@patch("src.optimization.optimize.burger_cook_loop")
@patch("src.optimization.optimize.fry_cook_loop")
@patch("src.optimization.optimize.ice_cream_cook_loop")  # NEW: Mock the ice cream loop!
@patch("src.optimization.optimize.inventory_manager")
def test_run_sim_for_optuna_math(
    mock_inv, mock_ice_cream, mock_fry, mock_burger, mock_arrivals, mock_config
):
    """
    Test that the profit calculation (Revenue - (Staff Cost + Waste Cost))
    is perfectly accurate without running the real simulation loops.
    """
    # 1. Setup predictable config constants
    mock_config.SIM_TIME = 3600.0  # Exactly 1 hour
    mock_config.WAGE_CASHIER = 15.0
    mock_config.WAGE_BURGER_COOK = 20.0
    mock_config.WAGE_FRIES_COOK = 10.0
    mock_config.WAGE_ICE_CREAM_COOK = 12.0  # NEW: Ice cream cook wage
    mock_config.COST_WASTED_BURGER = 2.0
    mock_config.COST_WASTED_FRIES = 1.0
    mock_config.COST_WASTED_ICE_CREAM = 1.5  # NEW: Ice cream waste cost

    # 2. Inject fake data into the stats dictionary when customer_arrivals is called
    def fake_arrivals(env, restaurant, stats):
        stats["captured_revenue"].extend([10.0, 15.0, 25.0])  # Total = $50.0
        stats["wasted_burgers"].extend([1, 1, 1])  # 3 wasted burgers = $6.0 waste
        stats["wasted_fries"].extend([1, 1])  # 2 wasted fries = $2.0 waste
        stats["wasted_ice_cream"].extend(
            [1, 1]
        )  # NEW: 2 wasted ice creams = $3.0 waste
        yield env.timeout(0)

    # SimPy requires env.process() targets to be generators.
    # We give the other background loops a dummy generator so they don't crash SimPy.
    def dummy_process(env, *args):
        yield env.timeout(0)

    mock_arrivals.side_effect = fake_arrivals
    mock_burger.side_effect = dummy_process
    mock_fry.side_effect = dummy_process
    mock_ice_cream.side_effect = dummy_process  # NEW: Handle ice cream generator
    mock_inv.side_effect = dummy_process

    # 3. Run the function with 1 of each staff member and 1 seed
    avg_profit = optimize.run_sim_for_optuna(
        cashiers=1, burger_cooks=1, fries_cooks=1, ice_cream_cooks=1, n_seeds=1
    )

    # 4. Do the math manually to verify
    # Revenue = $50.0
    # Staff Cost (1 hour) = $15 + $20 + $10 + $12 = $57.0
    # Waste Cost = $6.0 + $2.0 + $3.0 = $11.0
    # Expected Profit = 50.0 - (57.0 + 11.0) = -18.0

    assert avg_profit == -18.0


# --- TESTS: OPTUNA OBJECTIVE ---


@patch("src.optimization.optimize.run_sim_for_optuna")
@patch("src.optimization.optimize.config")
def test_objective_updates_config_and_runs(mock_config, mock_run_sim):
    """Test that the objective function correctly pulls suggestions from Optuna."""

    # 1. Setup a fake Optuna trial
    mock_trial = MagicMock()

    # Force the trial to suggest exactly '3' for every integer
    mock_trial.suggest_int.return_value = 3

    # Force the simulation to return a fake profit of $1000
    mock_run_sim.return_value = 1000.0

    # 2. Run the objective
    result = optimize.objective(mock_trial)

    # 3. Verify it updated the global config for inventory targets
    assert mock_config.TARGET_BURGER_INV == 3
    assert mock_config.TARGET_FRIES_INV == 3
    assert mock_config.TARGET_ICE_CREAM_INV == 3  # NEW: Verify ice cream target

    # 4. Verify it passed the right staff counts to the simulation runner
    mock_run_sim.assert_called_once_with(
        3,  # cashiers
        3,  # burger_cooks
        3,  # fries_cooks
        3,  # ice_cream_cooks (NEW)
        optimize.N_SEEDS,
    )

    # 5. Verify it returned the profit back to Optuna
    assert result == 1000.0
