"""
setup_project.py

This script helps newcomers set up the project environment automatically.
It will:
1. Check for Anaconda installation (and guide if not installed)
2. Create a new conda environment
3. Install requirements from requirements.txt

Instructions:
- Run this script with Python (not inside a conda environment).
"""

import os
import sys
import subprocess
import shutil

REQUIREMENTS_PATH = os.path.join('src', 'requirements.txt')
ENV_NAME = 'medical_app_env'
PYTHON_VERSION = '3.10'


def is_conda_installed():
    return shutil.which('conda') is not None


def print_anaconda_install_instructions():
    print("Anaconda is not installed.")
    print("Please download and install Anaconda from: https://www.anaconda.com/products/distribution")
    print("After installation, re-run this script.")
    sys.exit(1)


def create_conda_env(env_name, python_version):
    print(f"Creating conda environment '{env_name}' with Python {python_version}...")
    subprocess.run([
        'conda', 'create', '-y', '-n', env_name, f'python={python_version}'
    ], check=True)


def install_requirements(env_name, requirements_path):
    print(f"Installing requirements from {requirements_path} in environment '{env_name}'...")
    subprocess.run([
        'conda', 'run', '-n', env_name, 'pip', 'install', '-r', requirements_path
    ], check=True)


def main():
    print("==== Project Setup Script ====")
    if not is_conda_installed():
        print_anaconda_install_instructions()

    # Check if environment already exists
    envs = subprocess.check_output(['conda', 'env', 'list']).decode()
    if ENV_NAME in envs:
        print(f"Conda environment '{ENV_NAME}' already exists. Skipping creation.")
    else:
        create_conda_env(ENV_NAME, PYTHON_VERSION)

    # Install requirements
    if not os.path.exists(REQUIREMENTS_PATH):
        print(f"requirements.txt not found at {REQUIREMENTS_PATH}. Please check your project structure.")
        sys.exit(1)
    install_requirements(ENV_NAME, REQUIREMENTS_PATH)

    print("\nSetup complete!")
    print(f"To activate the environment, run: conda activate {ENV_NAME}")
    print("You can now start using the project.")

if __name__ == "__main__":
    main()
