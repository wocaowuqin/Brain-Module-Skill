"""Compatibility wrapper for the legacy root command."""
from runpy import run_module

if __name__ == "__main__":
    run_module("scripts.summarize_sdn_resource_intervals", run_name="__main__")
