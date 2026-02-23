import pytest
import simpy

from src.sim.restaurant import FastFoodRestaurant

# --- FIXTURES ---


@pytest.fixture
def env():
    """Provides a fresh SimPy environment for each test."""
    return simpy.Environment()


@pytest.fixture
def standard_restaurant(env):
    """Provides a baseline restaurant with 2 cashiers, 3 burger cooks, 1 fries cook, and 1 ice cream cook."""
    return FastFoodRestaurant(
        env,
        num_cashiers=2,
        num_burger_cooks=3,
        num_fries_cooks=1,
        num_ice_cream_cooks=1,
    )


# --- TESTS ---


def test_initial_state(standard_restaurant):
    """Test that all non-SimPy attributes start with the correct default values."""
    assert standard_restaurant.customers_waiting_for_food == 0


def test_resource_capacities(standard_restaurant):
    """Test that the human resources are created with the exact capacities requested."""
    assert standard_restaurant.cashier.capacity == 2
    assert standard_restaurant.burger_cook.capacity == 3
    assert standard_restaurant.fries_cook.capacity == 1
    assert standard_restaurant.ice_cream_cook.capacity == 1  # NEW: Ice Cream


def test_shelves_are_empty_stores(standard_restaurant, env):
    """Test that shelves are initialized as SimPy Stores and start completely empty."""
    assert isinstance(standard_restaurant.burger_shelf, simpy.Store)
    assert isinstance(standard_restaurant.fries_shelf, simpy.Store)
    assert isinstance(
        standard_restaurant.ice_cream_shelf, simpy.Store
    )  # NEW: Ice Cream

    # Stores should have 0 items at the very beginning
    assert len(standard_restaurant.burger_shelf.items) == 0
    assert len(standard_restaurant.fries_shelf.items) == 0
    assert len(standard_restaurant.ice_cream_shelf.items) == 0  # NEW: Ice Cream


@pytest.mark.parametrize(
    "cashiers, burgers, fries, ice_cream",
    [
        (1, 1, 1, 1),  # Minimal staffing
        (10, 5, 5, 2),  # High traffic staffing
        (5, 10, 2, 4),  # Unbalanced staffing
    ],
)
def test_custom_staffing_levels(env, cashiers, burgers, fries, ice_cream):
    """Test that the restaurant can be dynamically initialized with various staffing configurations."""
    restaurant = FastFoodRestaurant(env, cashiers, burgers, fries, ice_cream)

    assert restaurant.cashier.capacity == cashiers
    assert restaurant.burger_cook.capacity == burgers
    assert restaurant.fries_cook.capacity == fries
    assert restaurant.ice_cream_cook.capacity == ice_cream  # NEW: Ice Cream


def test_zero_staffing_raises_error(env):
    """
    SimPy requires Resource capacity to be > 0.
    This test ensures we catch the ValueError if someone tries to run a ghost kitchen with 0 staff.
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
        )  # NEW: Ice Cream


def test_environment_binding(standard_restaurant, env):
    """Ensure the restaurant is bound to the exact environment passed to it."""
    assert standard_restaurant.env is env
    # Check that resources belong to the same environment
    assert standard_restaurant.cashier._env is env
