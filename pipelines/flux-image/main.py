#!/usr/bin/env python3
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {description} failed")
        print(f"STDERR: {result.stderr}")
        sys.exit(1)

    print(f"Success: {description} completed")
    if result.stdout.strip():
        print(f"Output: {result.stdout}")


def main():
    # Ensure we're in the right directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Create local directories if they don't exist
    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    # Build the Docker image
    build_cmd = ["docker", "build", "-t", "flux-pipeline", "."]
    run_command(build_cmd, "Building Docker image")

    # Run the container
    run_cmd = [
        "docker",
        "run",
        "--gpus",
        "all",
        "-v",
        "./models:/app/models",
        "-v",
        "./outputs:/app/outputs",
        "flux-pipeline",
    ]
    run_command(run_cmd, "Running Docker container")


if __name__ == "__main__":
    main()
