Configuration & Difficulty
==========================

The simulation parameters are defined in ``src/config.py``. This file acts as the "Source of Truth" for the restaurant's physical laws.

The Difficulty Toggle
---------------------
The ``DIFFICULTY`` constant determines the margins of error for the agent.

.. warning::
   Switching to **NIGHTMARE** mode significantly reduces customer patience. If the agent does not maintain a "buffer" of food, walk-outs will cascade, leading to a "death spiral" in profit.

Key Parameters
--------------
* **Shelf Life**: Determines how long an item remains in a ``simpy.Store`` before the ``inventory_manager`` removes it as waste.
* **Balking Limits**: If the cashier queue exceeds ``MAX_QUEUE_LENGTH``, customers leave immediately without spending.

.. tip::
   When debugging new features, set ``DIFFICULTY="SIMPLE"`` to ensure the underlying logic works before subjecting the agent to the punishing Nightmare constraints.
