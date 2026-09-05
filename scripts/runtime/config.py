"""Configuration and input loading for the runtime workflow.

The implementation is temporarily delegated to the legacy-compatible runtime
module so callers can migrate imports without changing behavior.
"""
from scripts.run_sdn_runtime_requests import load_json, read_jsonl, load_sfc_plans, load_reroute_events, load_vnf_control_events
__all__ = ["load_json", "read_jsonl", "load_sfc_plans", "load_reroute_events", "load_vnf_control_events"]
