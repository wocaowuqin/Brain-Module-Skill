"""Compatibility wrapper for the legacy root command."""
from runpy import run_module

if __name__ == "__main__":
    run_module("scripts.convert_legacy_full8_to_sdn_runtime", run_name="__main__")
