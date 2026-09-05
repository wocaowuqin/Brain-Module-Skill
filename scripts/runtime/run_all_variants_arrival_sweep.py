"""Compatibility wrapper for the legacy root command."""
from runpy import run_module

if __name__ == "__main__":
    run_module("scripts.run_all_variants_arrival_sweep", run_name="__main__")
