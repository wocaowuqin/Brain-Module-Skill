"""Compatibility wrapper for the legacy root command."""
from runpy import run_module

if __name__ == "__main__":
    run_module("scripts.build_sdn_strict_gate_dataset", run_name="__main__")
