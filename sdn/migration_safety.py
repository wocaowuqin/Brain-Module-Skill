"""Pure validation helpers for make-before-break SFC migration plans."""

from __future__ import annotations

from typing import Any, Mapping


def segment_rule_map(plan: Mapping[str, Any]) -> dict[tuple[int, str, int], int]:
    rules: dict[tuple[int, str, int], int] = {}
    for segment in plan.get("segments") or []:
        target_ip = str(segment["target_ip"])
        udp_port = int(segment["udp_port"])
        outputs = segment.get("switch_outputs") or {}
        for raw_dpid in segment.get("path") or []:
            dpid = int(raw_dpid)
            ports = outputs.get(str(dpid), outputs.get(dpid))
            if not ports:
                raise ValueError(f"segment has no output for switch {dpid}")
            output = int(ports[0])
            key = (dpid, target_ip, udp_port)
            previous = rules.get(key)
            if previous is not None and previous != output:
                raise ValueError(f"conflicting rule inside candidate plan at {key}")
            rules[key] = output
    return rules


def make_before_break_compatible(
    current_plan: Mapping[str, Any],
    candidate_plan: Mapping[str, Any],
) -> tuple[bool, str]:
    """Mirror Ryu's live-overlap gate before endpoint or bandwidth reservation."""

    try:
        current = segment_rule_map(current_plan)
        candidate = segment_rule_map(candidate_plan)
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"invalid_segment_rules:{exc}"
    for key in current.keys() & candidate.keys():
        if current[key] != candidate[key]:
            return False, (
                "live_rule_overlap:"
                f"switch={key[0]},target={key[1]}:{key[2]},"
                f"old_output={current[key]},new_output={candidate[key]}"
            )
    current_multicast = current_plan.get("multicast") or {}
    candidate_multicast = candidate_plan.get("multicast") or {}
    for key in ("root_dpid", "dst_ip", "switch_outputs"):
        if candidate_multicast.get(key) != current_multicast.get(key):
            return False, f"multicast_changed:{key}"
    return True, ""
