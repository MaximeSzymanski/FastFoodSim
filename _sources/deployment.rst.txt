Docker Deployment
=================

To ensure consistent environment replication and bypass complex dependency compilation (like Atari emulators), FastFoodSim is fully containerized.

Training Architecture
---------------------
The ``docker-compose.yaml`` orchestrates specialized containers:

* ``fastfood_trainer``: Headless container running the RL algorithms safely without GUI dependencies.
* ``fastfood_reporter``: Evaluates the finalized ``.zip`` model across multiple random seeds to generate the business report.

.. code-block:: bash

   # Build and run the training pipeline
   docker compose build --no-cache
   docker compose up -d

.. warning::
   Do not run the Pygame visualizer inside the Docker container on macOS due to display driver (ALSA/XDG) limitations. Always run the visualizer locally.
