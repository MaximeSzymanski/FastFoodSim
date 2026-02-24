import subprocess
import sys
import time

modules = ["src.rl.train_ppo", "src.rl.train_a2c", "src.rl.train_dqn"]


def run_scripts():
    """Sequentially executes multiple reinforcement learning training scripts.

    Iterates through a predefined list of Python modules, executing each as a
    subprocess. If a script fails or is interrupted by the user, the sequence
    aborts. Prints the total execution time upon completion.
    """
    print("WELCOME TO THE AI KITCHEN BATTLE ROYALE")
    print(f"Preparing to train {len(modules)} models sequentially...\n")

    start_time = time.time()

    for module in modules:
        print("=" * 50)
        print(f"LAUNCHING: python -m {module}")
        print("=" * 50)

        try:
            result = subprocess.run([sys.executable, "-m", module])

            if result.returncode != 0:
                print(f"\nERROR: {module} exited with code {
                        result.returncode}. Stopping tournament.")
                break

        except KeyboardInterrupt:
            print("\nTournament aborted by user.")
            break

    total_time = (time.time() - start_time) / 60
    print("\n" + "=" * 50)
    print(f"TOURNAMENT COMPLETE! Total time: {total_time:.1f} minutes.")
    print("Run 'tensorboard --logdir ./tb_logs/' to see who won!")
    print("=" * 50)


if __name__ == "__main__":
    run_scripts()
