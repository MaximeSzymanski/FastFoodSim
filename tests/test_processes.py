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
    inventory_manager,
)
from src.sim.restaurant import FastFoodRestaurant

# --- FIXTURES ---


@pytest.fixture
def env():
    return simpy.Environment()


@pytest.fixture
def restaurant(env):
    return FastFoodRestaurant(
        env, num_cashiers=1, num_burger_cooks=1, num_fries_cooks=1
    )


@pytest.fixture
def stats():
    return {
        "wasted_burgers": [],
        "wasted_fries": [],
        "balked": [],
        "lost_revenue": [],
        "reneged": [],
        "wait_times": [],
        "captured_revenue": [],
    }


# --- TESTS: COOK LOOPS ---


@patch("src.sim.processes.TARGET_FRIES_INV", 5)
@patch("src.sim.processes.FRIES_TIME", 10.0)
@patch("src.sim.processes.FRIES_BATCH_SIZE", 3)
def test_fry_cook_loop_cooks_when_low(env, restaurant):
    """Test that the fry cook makes a batch of fries when inventory is below target."""
    # Shelf starts empty (0 < 5). Cook should wait 10.0 units, then produce 3 fries.
    env.process(fry_cook_loop(env, restaurant))
    env.run(until=11)

    assert len(restaurant.fries_shelf.items) == 3
    assert restaurant.fries_shelf.items[0].creation_time == 10.0


@patch("src.sim.processes.TARGET_FRIES_INV", 2)
def test_fry_cook_loop_idles_when_full(env, restaurant):
    """Test that the fry cook idles for 5.0 units when inventory is full."""
    # Pre-stock the shelf so it's not below target
    restaurant.fries_shelf.put(FoodItem(0))
    restaurant.fries_shelf.put(FoodItem(0))

    env.process(fry_cook_loop(env, restaurant))
    env.run(until=6)  # Run just past the 5.0 idle timeout

    # Still only 2 items, meaning the cook didn't make more
    assert len(restaurant.fries_shelf.items) == 2


@patch("src.sim.processes.TARGET_BURGER_INV", 5)
@patch("src.sim.processes.BURGER_BATCH_SIZE", 2)
def test_burger_cook_loop_cooks_when_low(env, restaurant):
    """Test that the burger cook makes a batch of burgers when inventory is below target."""
    # Mock the random triangular cooking time to take exactly 8.0 units
    with patch("src.sim.processes.random.triangular", return_value=8.0):
        env.process(burger_cook_loop(env, restaurant))
        env.run(until=9)

    assert len(restaurant.burger_shelf.items) == 2
    assert restaurant.burger_shelf.items[0].creation_time == 8.0


# --- TESTS: FOOD & INVENTORY ---


def test_food_item_creation():
    """Test that FoodItem stores its creation time correctly."""
    food = FoodItem(creation_time=5.5)
    assert food.creation_time == 5.5


@patch("src.sim.processes.BURGER_SHELF_LIFE", 20)
@patch("src.sim.processes.FRIES_SHELF_LIFE", 15)
def test_inventory_manager_expires_old_food(env, restaurant, stats):
    """Test that the inventory manager correctly identifies and trashes expired food."""

    restaurant.burger_shelf.items = [
        FoodItem(0),
        FoodItem(-15),
    ]  # Age 10 and Age 25 (Expired)
    restaurant.fries_shelf.items = [
        FoodItem(0),
        FoodItem(-10),
    ]  # Age 10 and Age 20 (Expired)

    env.process(inventory_manager(env, restaurant, stats))
    env.run(until=11)  # Run past the 10-second check interval

    assert len(restaurant.burger_shelf.items) == 1
    assert restaurant.burger_shelf.items[0].creation_time == 0
    assert len(stats["wasted_burgers"]) == 1

    assert len(restaurant.fries_shelf.items) == 1
    assert restaurant.fries_shelf.items[0].creation_time == 0
    assert len(stats["wasted_fries"]) == 1


# --- TESTS: CUSTOMER JOURNEY ---


@patch("src.sim.processes.MAX_QUEUE_LENGTH", 0)
@patch("src.sim.processes.PRICE_BURGER", 5.0)
@patch("src.sim.processes.PRICE_FRIES", 2.0)
def test_customer_balking_due_to_queue(env, restaurant, stats):
    """Test that a customer balks if the cashier queue is too long."""
    with patch("src.sim.processes.random.choices", side_effect=[[1], [0]]):
        env.process(customer_journey(env, "TestCust", restaurant, stats))
        env.run()

    assert len(stats["balked"]) == 1
    assert stats["lost_revenue"][0] == 5.0


@patch("src.sim.processes.MAX_WAIT_TOLERANCE", 5)
def test_customer_reneg_due_to_wait(env, restaurant, stats):
    """Test that a customer leaves the line if they wait too long for a cashier."""

    def slow_cashier(env, res):
        with res.request() as req:
            yield req
            yield env.timeout(100)

    env.process(slow_cashier(env, restaurant.cashier))

    with patch("src.sim.processes.random.choices", side_effect=[[1], [1]]):
        env.process(customer_journey(env, "TestCust", restaurant, stats))
        env.run(until=10)

    assert len(stats["reneged"]) == 1
    assert len(stats["captured_revenue"]) == 0


@patch("src.sim.processes.CASHIER_MIN", 1)
@patch("src.sim.processes.CASHIER_MAX", 1)
@patch("src.sim.processes.CASHIER_MODE", 1)
@patch("src.sim.processes.PRICE_BURGER", 5.0)
@patch("src.sim.processes.PRICE_FRIES", 2.0)
def test_customer_successful_journey(env, restaurant, stats):
    """Test a perfect scenario where a customer orders, gets food immediately, and leaves."""
    restaurant.burger_shelf.items.append(FoodItem(0))
    restaurant.fries_shelf.items.append(FoodItem(0))

    with patch("src.sim.processes.random.choices", side_effect=[[1], [1]]):
        with patch("src.sim.processes.random.triangular", return_value=1.0):
            env.process(customer_journey(env, "TestCust", restaurant, stats))
            env.run()

    assert len(stats["captured_revenue"]) == 1
    assert stats["captured_revenue"][0] == 7.0
    assert len(stats["wait_times"]) == 1
    assert stats["wait_times"][0] == 11.0  # 1.0 cashier + 10.0 large order penalty


# --- TESTS: ARRIVALS ---


@patch("src.sim.processes.ARRIVAL_AVG", 1)
def test_customer_arrivals_generator(env, restaurant, stats):
    """Test that the arrival loop continuously spawns customers."""
    with patch("src.sim.processes.random.expovariate", return_value=2.0):
        env.process(customer_arrivals(env, restaurant, stats))

        with patch("src.sim.processes.customer_journey") as mock_journey:
            env.run(until=5)
            assert mock_journey.call_count == 2
