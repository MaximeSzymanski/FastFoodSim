from unittest.mock import patch

import pytest
import simpy

from src.sim.processes import (
    FoodItem,
    burger_cook_loop,
    customer_arrivals,
    customer_journey,
    fry_cook_loop,
    grab_food_from_shelf,
    ice_cream_cook_loop,
    inventory_manager,
)
from src.sim.restaurant import FastFoodRestaurant


@pytest.fixture
def env():
    """Provides a fresh SimPy environment instance for testing.

    Returns:
        simpy.Environment: The active simulation environment.
    """
    return simpy.Environment()


@pytest.fixture
def restaurant(env):
    """Provides a FastFoodRestaurant instance populated with one of each staff member.

    Args:
        env (simpy.Environment): The active simulation environment.

    Returns:
        FastFoodRestaurant: The initialized restaurant state object.
    """
    return FastFoodRestaurant(
        env,
        num_cashiers=1,
        num_burger_cooks=1,
        num_fries_cooks=1,
        num_ice_cream_cooks=1,
    )


@pytest.fixture
def stats():
    """Provides a clean dictionary for tracking simulation statistics.

    Returns:
        dict: The initialized statistics tracker.
    """
    return {
        "wasted_burgers": [],
        "wasted_fries": [],
        "wasted_ice_cream": [],
        "balked": [],
        "lost_revenue": [],
        "reneged": [],
        "wait_times": [],
        "captured_revenue": [],
    }


@patch("src.sim.processes.TARGET_FRIES_INV", 5)
@patch("src.sim.processes.FRIES_TIME", 10.0)
@patch("src.sim.processes.FRIES_BATCH_SIZE", 3)
def test_fry_cook_loop_cooks_when_low(env, restaurant):
    """Verifies that the fry cook process generates a batch of fries when inventory falls below target.

    Args:
        env (simpy.Environment): The test environment fixture.
        restaurant (FastFoodRestaurant): The test restaurant fixture.
    """
    env.process(fry_cook_loop(env, restaurant))
    env.run(until=11)

    assert len(restaurant.fries_shelf.items) == 3
    assert restaurant.fries_shelf.items[0].creation_time == 10.0


@patch("src.sim.processes.TARGET_FRIES_INV", 2)
def test_fry_cook_loop_idles_when_full(env, restaurant):
    """Checks that the fry cook process idles appropriately when the inventory is at or above target.

    Args:
        env (simpy.Environment): The test environment fixture.
        restaurant (FastFoodRestaurant): The test restaurant fixture.
    """
    restaurant.fries_shelf.put(FoodItem(0))
    restaurant.fries_shelf.put(FoodItem(0))

    env.process(fry_cook_loop(env, restaurant))
    env.run(until=6)

    assert len(restaurant.fries_shelf.items) == 2


@patch("src.sim.processes.TARGET_BURGER_INV", 5)
@patch("src.sim.processes.BURGER_BATCH_SIZE", 2)
def test_burger_cook_loop_cooks_when_low(env, restaurant):
    """Validates that the burger cook process produces a batch of burgers when inventory is depleted.

    Args:
        env (simpy.Environment): The test environment fixture.
        restaurant (FastFoodRestaurant): The test restaurant fixture.
    """
    with patch("src.sim.processes.random.triangular", return_value=8.0):
        env.process(burger_cook_loop(env, restaurant))
        env.run(until=9)

    assert len(restaurant.burger_shelf.items) == 2
    assert restaurant.burger_shelf.items[0].creation_time == 8.0


@patch("src.sim.processes.TARGET_ICE_CREAM_INV", 5)
@patch("src.sim.processes.ICE_CREAM_TIME", 5.0)
@patch("src.sim.processes.ICE_CREAM_BATCH_SIZE", 2)
def test_ice_cream_cook_loop_cooks_when_low(env, restaurant):
    """Ensures the ice cream cook process pours a batch of ice cream when inventory requires replenishing.

    Args:
        env (simpy.Environment): The test environment fixture.
        restaurant (FastFoodRestaurant): The test restaurant fixture.
    """
    env.process(ice_cream_cook_loop(env, restaurant))
    env.run(until=6)

    assert len(restaurant.ice_cream_shelf.items) == 2
    assert restaurant.ice_cream_shelf.items[0].creation_time == 5.0


def test_food_item_creation():
    """Verifies that the FoodItem class correctly initializes and stores its creation timestamp."""
    food = FoodItem(creation_time=5.5)
    assert food.creation_time == 5.5


@patch("src.sim.processes.BURGER_SHELF_LIFE", 20)
@patch("src.sim.processes.FRIES_SHELF_LIFE", 15)
@patch("src.sim.processes.ICE_CREAM_SHELF_LIFE", 5)
def test_inventory_manager_expires_old_food(env, restaurant, stats):
    """Tests the inventory manager's ability to accurately identify and discard expired food items.

    Args:
        env (simpy.Environment): The test environment fixture.
        restaurant (FastFoodRestaurant): The test restaurant fixture.
        stats (dict): The test statistics tracking fixture.
    """
    restaurant.burger_shelf.items = [
        FoodItem(0),
        FoodItem(-15),
    ]
    restaurant.fries_shelf.items = [
        FoodItem(0),
        FoodItem(-10),
    ]
    restaurant.ice_cream_shelf.items = [
        FoodItem(7),
        FoodItem(-1),
    ]

    env.process(inventory_manager(env, restaurant, stats))
    env.run(until=11)

    assert len(restaurant.burger_shelf.items) == 1
    assert len(stats["wasted_burgers"]) == 1

    assert len(restaurant.fries_shelf.items) == 1
    assert len(stats["wasted_fries"]) == 1

    assert len(restaurant.ice_cream_shelf.items) == 1
    assert len(stats["wasted_ice_cream"]) == 1


@patch("src.sim.processes.MAX_QUEUE_LENGTH", 0)
@patch("src.sim.processes.PRICE_BURGER", 5.0)
@patch("src.sim.processes.PRICE_FRIES", 2.0)
@patch("src.sim.processes.PRICE_ICE_CREAM", 3.0)
def test_customer_balking_due_to_queue(env, restaurant, stats):
    """Asserts that a customer will balk and leave the restaurant if the cashier queue exceeds the maximum permitted length.

    Args:
        env (simpy.Environment): The test environment fixture.
        restaurant (FastFoodRestaurant): The test restaurant fixture.
        stats (dict): The test statistics tracking fixture.
    """
    with patch("src.sim.processes.random.choices", side_effect=[[1], [0], [0]]):
        env.process(customer_journey(env, "TestCust", restaurant, stats))
        env.run()

    assert len(stats["balked"]) == 1
    assert stats["lost_revenue"][0] == 5.0


@patch("src.sim.processes.MAX_WAIT_TOLERANCE", 5)
def test_customer_reneg_due_to_wait(env, restaurant, stats):
    """Confirms that a customer will renege and abandon the queue if the wait time for a cashier exceeds their patience threshold.

    Args:
        env (simpy.Environment): The test environment fixture.
        restaurant (FastFoodRestaurant): The test restaurant fixture.
        stats (dict): The test statistics tracking fixture.
    """

    def slow_cashier(env, res):
        with res.request() as req:
            yield req
            yield env.timeout(100)

    env.process(slow_cashier(env, restaurant.cashier))

    with patch("src.sim.processes.random.choices", side_effect=[[1], [1], [1]]):
        env.process(customer_journey(env, "TestCust", restaurant, stats))
        env.run(until=10)

    assert len(stats["reneged"]) == 1
    assert len(stats["captured_revenue"]) == 0


@patch("src.sim.processes.CASHIER_MIN", 1)
@patch("src.sim.processes.CASHIER_MAX", 1)
@patch("src.sim.processes.CASHIER_MODE", 1)
@patch("src.sim.processes.PRICE_BURGER", 5.0)
@patch("src.sim.processes.PRICE_FRIES", 2.0)
@patch("src.sim.processes.PRICE_ICE_CREAM", 3.0)
def test_customer_successful_journey(env, restaurant, stats):
    """Tests a complete, successful customer journey from arrival to receiving food and capturing revenue.

    Args:
        env (simpy.Environment): The test environment fixture.
        restaurant (FastFoodRestaurant): The test restaurant fixture.
        stats (dict): The test statistics tracking fixture.
    """
    restaurant.burger_shelf.items.append(FoodItem(0))
    restaurant.fries_shelf.items.append(FoodItem(0))
    restaurant.ice_cream_shelf.items.append(FoodItem(0))

    with patch("src.sim.processes.random.choices", side_effect=[[1], [1], [1]]):
        with patch("src.sim.processes.random.triangular", return_value=1.0):
            env.process(customer_journey(env, "TestCust", restaurant, stats))
            env.run()

    assert len(stats["captured_revenue"]) == 1
    assert stats["captured_revenue"][0] == 10.0


@patch("src.sim.processes.ARRIVAL_AVG", 1)
def test_customer_arrivals_generator(env, restaurant, stats):
    """Verifies that the arrival generator continuously spawns new customer processes at the expected rate.

    Args:
        env (simpy.Environment): The test environment fixture.
        restaurant (FastFoodRestaurant): The test restaurant fixture.
        stats (dict): The test statistics tracking fixture.
    """
    with patch("src.sim.processes.random.expovariate", return_value=2.0):
        env.process(customer_arrivals(env, restaurant, stats))

        with patch("src.sim.processes.customer_journey") as mock_journey:
            env.run(until=5)
            assert mock_journey.call_count == 2
