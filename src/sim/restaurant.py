import simpy


class FastFoodRestaurant:
    """A simulation container for the fast-food restaurant's resources and state.

    This class manages the human resources (cooks, cashiers), physical inventory
    shelves, and live tracking metrics like pending orders and waiting customers.
    """

    def __init__(
        self,
        env,
        num_cashiers,
        num_burger_cooks,
        num_fries_cooks,
        num_ice_cream_cooks=1,
    ):
        """Initializes the FastFoodRestaurant with required staff and resources.

        Args:
            env (simpy.Environment): The active simulation environment.
            num_cashiers (int): The initial number of cashiers at the front counter.
            num_burger_cooks (int): The initial number of cooks assigned to the burger station.
            num_fries_cooks (int): The initial number of cooks assigned to the fry station.
            num_ice_cream_cooks (int, optional): The initial number of cooks assigned to the dessert station. Defaults to 1.
        """
        self.env = env

        self.cashier = simpy.Resource(env, num_cashiers)
        self.burger_cook = simpy.Resource(env, num_burger_cooks)
        self.fries_cook = simpy.Resource(env, num_fries_cooks)
        self.ice_cream_cook = simpy.Resource(env, num_ice_cream_cooks)

        self.burger_shelf = simpy.Store(env)
        self.fries_shelf = simpy.Store(env)
        self.ice_cream_shelf = simpy.Store(env)

        self.customers_waiting_for_food = 0

        self.pending_burgers = 0
        self.pending_fries = 0
        self.pending_ice_cream = 0
