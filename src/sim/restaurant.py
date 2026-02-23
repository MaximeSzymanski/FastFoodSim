import simpy


class FastFoodRestaurant:
    def __init__(
        self, env, num_cashiers, num_burger_cooks, num_fries_cooks, num_ice_cream_cooks
    ):
        self.env = env

        # Human resources (The AI will monitor and command these!)
        self.cashier = simpy.Resource(env, num_cashiers)
        self.burger_cook = simpy.Resource(env, num_burger_cooks)
        self.fries_cook = simpy.Resource(env, num_fries_cooks)
        # --- NEW: ICE CREAM RESOURCES ---
        self.ice_cream_cook = simpy.Resource(env, num_ice_cream_cooks)

        # Physical inventory shelves
        self.burger_shelf = simpy.Store(env)
        self.fries_shelf = simpy.Store(env)
        self.ice_cream_shelf = simpy.Store(env)  # NEW

        self.customers_waiting_for_food = 0
