import pytest
import simpy

from src.sim.restaurant import FastFoodRestaurant


@pytest.fixture
def env():
    """Provides a fresh SimPy environment instance for testing.

    Returns:
        simpy.Environment: The active simulation environment.
    """
    return simpy.Environment()


@pytest.fixture
def standard_restaurant(env):
    """Provides a baseline restaurant with standard staffing levels.

    Args:
        env (simpy.Environment): The active simulation environment.

    Returns:
        FastFoodRestaurant: The initialized restaurant state object.
    """
    return FastFoodRestaurant(
        env,
        num_cashiers=2,
        num_burger_cooks=3,
        num_fries_cooks=1,
        num_ice_cream_cooks=1,
    )


def test_initial_state(standard_restaurant):
    """Verifies that all non-SimPy attributes initialize with correct default values.

    Args:
        standard_restaurant (FastFoodRestaurant): The test restaurant fixture.
    """
    assert standard_restaurant.customers_waiting_for_food == 0


def test_resource_capacities(standard_restaurant):
    """Checks that human resources are created with the exact requested capacities.

    Args:
        standard_restaurant (FastFoodRestaurant): The test restaurant fixture.
    """
    assert standard_restaurant.cashier.capacity == 2
    assert standard_restaurant.burger_cook.capacity == 3
    assert standard_restaurant.fries_cook.capacity == 1
    assert standard_restaurant.ice_cream_cook.capacity == 1


def test_shelves_are_empty_stores(standard_restaurant, env):
    """Validates that shelves are initialized as SimPy Stores and start completely empty.

    Args:
        standard_restaurant (FastFoodRestaurant): The test restaurant fixture.
        env (simpy.Environment): The test environment fixture.
    """
    assert isinstance(standard_restaurant.burger_shelf, simpy.Store)
    assert isinstance(standard_restaurant.fries_shelf, simpy.Store)
    assert isinstance(standard_restaurant.ice_cream_shelf, simpy.Store)

    assert len(standard_restaurant.burger_shelf.items) == 0
    assert len(standard_restaurant.fries_shelf.items) == 0
    assert len(standard_restaurant.ice_cream_shelf.items) == 0


@pytest.mark.parametrize(
    "cashiers, burgers, fries, ice_cream",
    [
        (1, 1, 1, 1),
        (10, 5, 5, 2),
        (5, 10, 2, 4),
    ],
)
def test_custom_staffing_levels(env, cashiers, burgers, fries, ice_cream):
    """Tests that the restaurant dynamically initializes with various staffing configurations.

    Args:
        env (simpy.Environment): The test environment fixture.
        cashiers (int): Number of cashiers.
        burgers (int): Number of burger cooks.
        fries (int): Number of fries cooks.
        ice_cream (int): Number of ice cream cooks.
    """
    restaurant = FastFoodRestaurant(env, cashiers, burgers, fries, ice_cream)

    assert restaurant.cashier.capacity == cashiers
    assert restaurant.burger_cook.capacity == burgers
    assert restaurant.fries_cook.capacity == fries
    assert restaurant.ice_cream_cook.capacity == ice_cream


def test_zero_staffing_raises_error(env):
    """Ensures a ValueError is raised if any resource capacity is set to zero.

    Args:
        env (simpy.Environment): The test environment fixture.
    """
    with pytest.raises(ValueError, match=r'"capacity" must be > 0\.'):
        FastFoodRestaurant(
            env,
            num_cashiers=0,
            num_burger_cooks=1,
            num_fries_cooks=1,
            num_ice_cream_cooks=1,
        )

    with pytest.raises(ValueError, match=r'"capacity" must be > 0\.'):
        FastFoodRestaurant(
            env,
            num_cashiers=1,
            num_burger_cooks=0,
            num_fries_cooks=1,
            num_ice_cream_cooks=1,
        )

    with pytest.raises(ValueError, match=r'"capacity" must be > 0\.'):
        FastFoodRestaurant(
            env,
            num_cashiers=1,
            num_burger_cooks=1,
            num_fries_cooks=0,
            num_ice_cream_cooks=1,
        )

    with pytest.raises(ValueError, match=r'"capacity" must be > 0\.'):
        FastFoodRestaurant(
            env,
            num_cashiers=1,
            num_burger_cooks=1,
            num_fries_cooks=1,
            num_ice_cream_cooks=0,
        )


def test_environment_binding(standard_restaurant, env):
    """Confirms the restaurant and its resources are correctly bound to the simulation environment.

    Args:
        standard_restaurant (FastFoodRestaurant): The test restaurant fixture.
        env (simpy.Environment): The test environment fixture.
    """
    assert standard_restaurant.env is env
    assert standard_restaurant.cashier._env is env
