"""
run.py
Single entry point for the whole project. Uses only relative paths (anchored
on this file's location) so it works the same on Windows, macOS and Linux.

Usage:
    python run.py --train     # clean data, train & compare models, save artifacts, generate charts
    python run.py --test      # run the automated test suite
    python run.py --serve     # launch the ScamShield AI web app (default action if no flag given)
    python run.py --all       # train, test, then serve
"""

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def sync_visualizations_to_static():
    """Copy the generated charts into app/static/img so the dashboard can display them."""
    src_dir = os.path.join(ROOT, "visualizations")
    dst_dir = os.path.join(ROOT, "app", "static", "img")
    os.makedirs(dst_dir, exist_ok=True)
    if not os.path.isdir(src_dir):
        return
    for fname in os.listdir(src_dir):
        if fname.lower().endswith(".png"):
            shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))
    print(f"[run.py] Synced visualizations -> {dst_dir}")


def train():
    from src.train import main as train_main
    from src.evaluate import main as evaluate_main

    print("[run.py] Step 1/2: training and comparing models...")
    train_main()
    print("[run.py] Step 2/2: generating evaluation charts...")
    evaluate_main()
    sync_visualizations_to_static()
    print("[run.py] Training pipeline complete. Artifacts saved to models/, charts saved to visualizations/.")


def run_tests():
    print("[run.py] Running automated test suite...")
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], cwd=ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)


def serve():
    required = os.path.join(ROOT, "models", "best_model.pkl")
    if not os.path.exists(required):
        print("[run.py] No trained model found - running training pipeline first...")
        train()
    sync_visualizations_to_static()
    from app.app import app
    port = int(os.environ.get("PORT", 5000))
    print(f"[run.py] Starting ScamShield AI on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)


def main():
    parser = argparse.ArgumentParser(description="ScamShield AI project runner")
    parser.add_argument("--train", action="store_true", help="Run the full data cleaning + training + evaluation pipeline")
    parser.add_argument("--test", action="store_true", help="Run the automated test suite")
    parser.add_argument("--serve", action="store_true", help="Launch the Flask web app")
    parser.add_argument("--all", action="store_true", help="Train, then test, then serve")
    args = parser.parse_args()

    if args.all:
        train()
        run_tests()
        serve()
        return

    ran_something = False
    if args.train:
        train()
        ran_something = True
    if args.test:
        run_tests()
        ran_something = True
    if args.serve:
        serve()
        ran_something = True

    if not ran_something:
        # Default action: just serve (training artifacts are already committed to the repo)
        serve()


if __name__ == "__main__":
    main()
