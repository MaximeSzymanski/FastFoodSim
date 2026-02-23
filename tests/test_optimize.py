from unittest.mock import MagicMock, patch

import pytest

from src.optimization import optimize


@patch("src.optimization.optimize.config")
@patch("src.optimization.optimize.customer_arrivals")
@patch("src.optimization.optimize.burger_cook_loop")
@patch("src.optimization.optimize.fry_cook_loop")
@patch("src.optimization.optimize.ice_cream_cook_loop")
@patch("src.optimization.optimize.inventory_manager")
def test_run_sim_for_optuna_math(
    mock_inv, mock_ice_cream, mock_fry, mock_burger, mock_arrivals, mock_config
):
    """Verifies the accuracy of the profit calculation without executing real simulation loops.

    Args:
        mock_inv (MagicMock): Mocked inventory manager process.
        mock_ice_cream (MagicMock): Mocked ice cream cook loop process.
        mock_fry (MagicMock): Mocked fry cook loop process.
        mock_burger (MagicMock): Mocked burger cook loop process.
        mock_arrivals (MagicMock): Mocked customer arrivals process.
        mock_config (MagicMock): Mocked configuration module.
    """
    mock_config.SIM_TIME = 3600.0
    mock_config.WAGE_CASHIER = 15.0
    mock_config.WAGE_BURGER_COOK = 20.0
    mock_config.WAGE_FRIES_COOK = 10.0
    mock_config.WAGE_ICE_CREAM_COOK = 12.0
    mock_config.COST_WASTED_BURGER = 2.0
    mock_config.COST_WASTED_FRIES = 1.0
    mock_config.COST_WASTED_ICE_CREAM = 1.5

    def fake_arrivals(env, restaurant, stats):
        """Injects predetermined financial data into the simulation statistics.

        Args:
            env (simpy.Environment): The active simulation environment.
            restaurant (FastFoodRestaurant): The restaurant state object.
            stats (dict): The dictionary tracking financial statistics.

        Yields:
            simpy.events.Timeout: A zero-time delay to satisfy SimPy generator requirements.
        """
        stats["captured_revenue"].extend([10.0, 15.0, 25.0])
        stats["wasted_burgers"].extend([1, 1, 1])
        stats["wasted_fries"].extend([1, 1])
        stats["wasted_ice_cream"].extend([1, 1])
        yield env.timeout(0)

    def dummy_process(env, *args):
        """Provides a no-operation generator for mocked SimPy processes.

        Args:
            env (simpy.Environment): The active simulation environment.
            *args: Variable length argument list to absorb process parameters.

        Yields:
            simpy.events.Timeout: A zero-time delay.
        """
        yield env.timeout(0)

    mock_arrivals.side_effect = fake_arrivals
    mock_burger.side_effect = dummy_process
    mock_fry.side_effect = dummy_process
    mock_ice_cream.side_effect = dummy_process
    mock_inv.side_effect = dummy_process

    avg_profit = optimize.run_sim_for_optuna(
        cashiers=1, burger_cooks=1, fries_cooks=1, ice_cream_cooks=1, n_seeds=1
    )

    assert avg_profit == -18.0


@patch("src.optimization.optimize.run_sim_for_optuna")
@patch("src.optimization.optimize.config")
def test_objective_updates_config_and_runs(mock_config, mock_run_sim):
    """Validates that the Optuna objective function correctly sets hyperparameters and triggers the simulation.

    Args:
        mock_config (MagicMock): Mocked configuration module.
        mock_run_sim (MagicMock): Mocked simulation runner function.
    """
    mock_trial = MagicMock()
    mock_trial.suggest_int.return_value = 3

    mock_run_sim.return_value = 1000.0

    result = optimize.objective(mock_trial)

    assert mock_config.TARGET_BURGER_INV == 3
    assert mock_config.TARGET_FRIES_INV == 3
    assert mock_config.TARGET_ICE_CREAM_INV == 3

    mock_run_sim.assert_called_once_with(
        3,
        3,
        3,
        3,
        optimize.N_SEEDS,
    )

    assert result == 1000.0
