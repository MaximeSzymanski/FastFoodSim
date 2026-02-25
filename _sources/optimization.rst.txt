Staffing Optimization (Optuna)
==============================

Finding the "Golden Ratio" of staff is a high-dimensional search problem. We use **Optuna** to automate this.

Optimization Logic
------------------
The ``objective`` function tries to maximize:
:math:`Profit = Revenue - (Wages + Waste + LostRevenue)`

.. important::
   Wages are calculated per second. Having too many staff members (e.g., 5 burger cooks) may result in high service levels but negative net profit due to the high "burn rate" of hourly wages.

Search Space
------------
Optuna suggests integers for:
1. **Staff Count**: Cashiers, Burger Cooks, Fry Cooks, and Ice Cream Cooks.
2. **Inventory Targets**: The "Par" levels that the static manager tries to maintain.

Running a Study
---------------
.. code-block:: python

   # From the project root
   python -m src.optimization.optimize

.. note::
   The optimizer uses ``N_SEEDS=10``. This means every configuration is tested across 10 different "random days" to ensure the results aren't just a fluke of a quiet day.
