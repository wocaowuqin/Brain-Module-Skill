"""Compatibility wrapper for the legacy root command."""
from runpy import run_module

if __name__ == "__main__":
    run_module("scripts.check_rate8_sft_migration_regression", run_name="__main__")
