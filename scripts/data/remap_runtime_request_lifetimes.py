"""Compatibility wrapper for the legacy root command."""
from runpy import run_module

if __name__ == "__main__":
    run_module("scripts.remap_runtime_request_lifetimes", run_name="__main__")
