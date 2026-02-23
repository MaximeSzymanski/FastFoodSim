import subprocess
import sys
import time

# List of the three training scripts using Python module dot-notation
modules = ["src.rl.train_ppo", "src.rl.train_a2c", "src.rl.train_dqn"]


def run_scripts():
    print("🏆 WELCOME TO THE AI KITCHEN BATTLE ROYALE 🏆")
    print(f"Preparing to train {len(modules)} models sequentially...\n")

    start_time = time.time()

    for module in modules:
        print("=" * 50)
        print(f"🚀 LAUNCHING: python -m {module}")
        print("=" * 50)

        try:
            # subprocess.run will now execute: python -m src.rl.train_...
            result = subprocess.run([sys.executable, "-m", module])

            # Check if the script crashed or was cancelled via Ctrl+C
            if result.returncode != 0:
                print(
                    f"\n❌ ERROR: {module} exited with code {result.returncode}. Stopping tournament."
                )
                break

        except KeyboardInterrupt:
            print("\n🛑 Tournament aborted by user.")
            break

    total_time = (time.time() - start_time) / 60
    print("\n" + "=" * 50)
    print(f"🏁 TOURNAMENT COMPLETE! Total time: {total_time:.1f} minutes.")
    print("Run 'tensorboard --logdir ./tb_logs/' to see who won!")
    print("=" * 50)


if __name__ == "__main__":
    run_scripts()
