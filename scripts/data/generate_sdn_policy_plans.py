"""Compatibility wrapper for the legacy root command."""
from runpy import run_module

if __name__ == "__main__":
    run_module("scripts.generate_sdn_policy_plans", run_name="__main__")
