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

Action Space & Masking
----------------------

We use a ``MultiDiscrete([2, 2, 2])`` space, which allows the agent to simultaneously control all three cooking stations:

* ``0``: Do nothing.
* ``1``: Command a cook to start a new batch.

Because this is a DES environment, standard algorithms fail when they send commands to busy workers. We use **Action Masking** to dynamically filter out invalid actions (e.g., masking the "cook burger" action if all burger cooks are currently busy).

The "Stick and Carrot" Reward
-----------------------------

.. warning::
   Rewarding *only* revenue often leads to "Lazy Agents" that stop cooking to avoid waste costs. If the agent perceives the risk of food expiration as too high, it may choose to never cook, resulting in zero revenue but also zero waste.

To solve this, we implemented a dual-incentive structure:

1. **The Carrot**: A flat bonus for every completed order to encourage throughput.
2. **The Stick**: A heavy penalty for walk-outs (balking or reneging) that far exceeds the lost dollar value of the meal.

Training the Agent
------------------

To train the Maskable PPO agent (which natively supports our action masks):

.. code-block:: bash

   python -m src.rl.train_ppo

.. hint::
   Because we utilize SimPy generators, Python's ``multiprocessing`` cannot pickle the environment. We strictly use ``DummyVecEnv`` instead of ``SubprocVecEnv`` to ensure stable vectorization during training.
