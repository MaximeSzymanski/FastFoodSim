# Fast Food Simulation & AI Optimization

This project is a high-fidelity *Discrete Event Simulation (DES)* of a fast-food restaurant built with *SimPy*. It features an autonomous staffing and inventory optimization engine powered by *Optuna* and a *Reinforcement Learning (RL)* environment compatible with *Gymnasium* for training AI managers.

## Trained RL Agent Performance Video
![RL Agent Performance](videos/lunch_rush_ai-episode-0.mp4)

## 🏗 Project Structure

The repository is organized into a modular `src` package to separate simulation logic, optimization, and AI training:

* **`src/sim/`**: Contains the core simulation logic.
    * `restaurant.py`: Defines the `FastFoodRestaurant` class, including human resources (cashiers, cooks) and physical inventory (shelves).
    * `processes.py`: Manages the "life" of the simulation, including customer arrivals, cooking loops, and the active waste management system.
* **`src/rl/`**: The Reinforcement Learning suite.
    * `FastFoodEnv.py`: A Gymnasium-wrapped environment with normalized observations and delta-based rewards.
    * `train.py`: The entry point for training neural networks to manage restaurant operations.
* **`src/optimization/`**: Staffing and inventory optimization using Optuna to maximize restaurant profit.
* **`tests/`**: A comprehensive `pytest` suite covering all components.

## 🚀 Key Code Snippets

### 1. The Active Waste Manager
This process monitors the freshness of inventory and handles waste calculation, which is critical for accurate profit modeling.

```python
def inventory_manager(env, restaurant, stats):
    """Continuously checks shelves and throws away expired food immediately."""
    while True:
        yield env.timeout(10.0) # Align with AI step interval

        # Logic for Burger Expiration
        valid_burgers = []
        for item in restaurant.burger_shelf.items:
            if env.now - item.creation_time > BURGER_SHELF_LIFE:
                stats["wasted_burgers"].append(1)
            else:
                valid_burgers.append(item)
        restaurant.burger_shelf.items = valid_burgers
```
### 2. RL Observation Scaling
Ensuring observations are within a `[0, 1]` range prevents gradient explosion and speeds up Neural Network convergence.

```python 
def _get_obs(self):
    """Gathers and scales 7 observations to a [0, 1] range."""
    queue_len = len(self.restaurant.cashier.queue) / 20.0
    burger_inv = len(self.restaurant.burger_shelf.items) / 30.0
    time_pct = self.env.now / SIM_TIME
    busy_cashiers = self.restaurant.cashier.count / self.num_cashiers

    obs = np.array([
        queue_len, 
        burger_inv, 
        idle_burgers, 
        time_pct, 
        busy_cashiers
    ])
    return np.clip(obs, 0.0, 1.0)
```

## 🛠 Usage & Testing

### Installation
Ensure you are using a virtual environment (Conda or venv) with Python 3.12+.
```bash
pip install simpy optuna gymnasium numpy pytest
```

### Running Tests

The project includes a robust suite of 26 tests. To ensure the simulation logic and RL environment are functioning correctly, run:
```bash
 python -m pytest tests/
```

### Staffing Optimization
To run the Optuna study and find the most profitable number of cashiers and cooks for your restaurant:
```bash
  python -m src.optimization.optimize
```

### AI Training
To start training the Reinforcement Learning agent using the Gymnasium environment:
```bash
  python -m src.rl.train
```
