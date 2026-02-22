import random

import simpy

from config import (
    ARRIVAL_AVG,
    BURGER_MAX,
    BURGER_MIN,
    BURGER_MODE,
    CASHIER_MAX,
    CASHIER_MIN,
    CASHIER_MODE,
    FRIES_MAX,
    FRIES_MIN,
)


def prepare_burger(env, name):
    """Sub-process for making a burger."""
    # print(f"[{env.now:.0f}s] Started burger for {name}")
    yield env.timeout(random.triangular(BURGER_MIN, BURGER_MAX, BURGER_MODE))
    # print(f"[{env.now:.0f}s] Finished burger for {name}")


def prepare_fries(env, name):
    """Sub-process for making fries."""
    # print(f"[{env.now:.0f}s] Started fries for {name}")
    # yield env.timeout(FRIES_TIME)
    # If you want uniform distribution instead:
    yield env.timeout(random.uniform(FRIES_MIN, FRIES_MAX))
    # print(f"[{env.now:.0f}s] Finished fries for {name}")


def customer_journey(env, name, restaurant, stats):
    """The full path a customer takes through the restaurant."""
    arrival_time = env.now

    # 1. Wait for and interact with Cashier
    with restaurant.cashier.request() as req:
        yield req
        order_time = random.triangular(CASHIER_MIN, CASHIER_MAX, CASHIER_MODE)
        yield env.timeout(order_time)

    # 2. Wait for a Cook to prepare the food
    with restaurant.cook.request() as req:
        yield req
        # Right now, the cook does the burger, then the fries sequentially.
        yield env.process(prepare_burger(env, name))
        yield env.process(prepare_fries(env, name))

    # 3. Customer leaves
    departure_time = env.now
    stats["wait_times"].append(departure_time - arrival_time)


def customer_arrivals(env, restaurant, stats):
    """Generates new customers based on an exponential decaying distribution."""
    count = 0
    while True:
        # Exponentially distributed arrival times
        yield env.timeout(random.expovariate(1.0 / ARRIVAL_AVG))
        count += 1
        name = f"Customer {count}"
        env.process(customer_journey(env, name, restaurant, stats))
