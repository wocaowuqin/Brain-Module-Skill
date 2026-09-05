"""Compatibility wrapper for the legacy root command."""
from runpy import run_module

if __name__ == "__main__":
    run_module("scripts.check_brain_module_skill_deployment", run_name="__main__")
