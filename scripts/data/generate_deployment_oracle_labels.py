"""Compatibility wrapper for the legacy root command."""
from runpy import run_module

if __name__ == "__main__":
    run_module("scripts.generate_deployment_oracle_labels", run_name="__main__")
