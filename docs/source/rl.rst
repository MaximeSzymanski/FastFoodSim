Reinforcement Learning
======================

The ``FastFoodEnv`` class wraps the SimPy engine into a **Gymnasium**-compatible interface, allowing standard reinforcement learning libraries to interact with the restaurant simulation.

Observation Space
-----------------

The agent receives a 12-dimensional normalized vector representing the current state of the restaurant:

* **Queues**: Current cashier and pickup area crowding.
* **Clock**: Progress through the shift (crucial for anticipating the lunch rush).
* **Inventory**: Live counts of Burgers, Fries, and Ice Cream.
* **Staff State**: Percentage of idle cooks per station.


Action Space
------------

We use a ``MultiDiscrete([2, 2, 2])`` space, which allows the agent to simultaneously control all three cooking stations:

* ``0``: Do nothing.
* ``1``: Command a cook to start a new batch.

.. note::
   Each index in the array corresponds to a specific station: [Burger, Fries, Ice Cream].

The "Stick and Carrot" Reward
-----------------------------

.. warning::
   Rewarding *only* revenue often leads to "Lazy Agents" that stop cooking to avoid waste costs. If the agent perceives the risk of food expiration as too high, it may choose to never cook, resulting in zero revenue but also zero waste.

To solve this, we implemented a dual-incentive structure:

1. **The Carrot**: A flat ``served_bonus`` for every completed order to encourage throughput.
2. **The Stick**: A heavy ``reputation_penalty`` for walk-outs (balking or reneging) that far exceeds the lost dollar value of the meal.

Training the Tournament
-----------------------

To compare algorithm performance across different architectures, you can run the sequential "Battle Royale" script:

.. code-block:: bash

   python train_all.py

.. hint::
   **MaskablePPO** typically outperforms DQN and A2C in this environment. Because it utilizes an action mask to "know" when a cook is busy, it avoids wasting timesteps on invalid actions.
