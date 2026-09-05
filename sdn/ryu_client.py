"""Stdlib REST client that converts switch paths into Ryu tree payloads."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Iterable, Mapping, Sequence


class RyuSFTClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    def _request(self, path: str, method: str = "GET", payload=None):
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ryu {method} {path} returned HTTP {exc.code}: {body}"
            ) from exc

    def status(self):
        return self._request("/sft/status")

    def configure(self, **values):
        return self._request("/sft/config", method="POST", payload=values)

    def install_tree(
        self,
        group_id: int,
        dst_ip: str,
        switch_outputs,
        source_dpid: int | None = None,
    ):
        payload = {
            "group_id": int(group_id),
            "dst_ip": str(dst_ip),
            "switch_outputs": switch_outputs,
        }
        if source_dpid is not None:
            payload["source_dpid"] = int(source_dpid)
        return self._request(
            "/sft/group",
            method="POST",
            payload=payload,
        )

    def reroute(
        self,
        group_id: int,
        dst_ip: str,
        switch_outputs,
        source_dpid: int | None = None,
        drain_seconds: float = 0.0,
    ):
        payload = {
            "group_id": int(group_id),
            "dst_ip": str(dst_ip),
            "switch_outputs": switch_outputs,
            "drain_seconds": float(drain_seconds),
        }
        if source_dpid is not None:
            payload["source_dpid"] = int(source_dpid)
        return self._request(
            "/sft/reroute",
            method="POST",
            payload=payload,
        )

    def delete_tree(self, group_id: int):
        return self._request(f"/sft/group/{int(group_id)}", method="DELETE")

    def install_sfc(self, plan):
        payload = self._sfc_payload(plan)
        return self._request(
            "/sft/sfc",
            method="POST",
            payload=payload,
        )

    @staticmethod
    def _sfc_payload(plan):
        multicast = dict(plan["multicast"])
        multicast.setdefault("group_id", int(plan["request_id"]))
        return {
            "request_id": int(plan["request_id"]),
            "segments": plan["segments"],
            "multicast": multicast,
        }

    def install_sfc_batch(self, plans):
        payloads = [self._sfc_payload(plan) for plan in plans]
        if not payloads:
            return []
        response = self._request(
            "/sft/sfc/batch",
            method="POST",
            payload={"requests": payloads},
        )
        results = response.get("results")
        if not isinstance(results, list) or len(results) != len(payloads):
            raise RuntimeError("Ryu returned an invalid SFC batch response")
        return results

    def delete_sfc(self, request_id: int):
        return self._request(f"/sft/sfc/{int(request_id)}", method="DELETE")

    def prepare_sfc_migration(self, plan, migration_token=None):
        payload = self._sfc_payload(plan)
        if migration_token is not None:
            payload["migration_token"] = str(migration_token)
        return self._request(
            "/sft/sfc/migration/prepare", method="POST", payload=payload
        )

    def commit_sfc_migration(
        self, request_id: int, migration_token: str, drain_seconds: float = 0.01
    ):
        return self._request(
            "/sft/sfc/migration/commit",
            method="POST",
            payload={
                "request_id": int(request_id),
                "migration_token": str(migration_token),
                "drain_seconds": float(drain_seconds),
            },
        )

    def abort_sfc_migration(self, request_id: int, migration_token: str):
        return self._request(
            "/sft/sfc/migration/abort",
            method="POST",
            payload={
                "request_id": int(request_id),
                "migration_token": str(migration_token),
            },
        )

    def reroute_path(
        self,
        group_id: int,
        dst_ip: str,
        switch_path: Sequence[int],
        leaf_ports: Mapping[int, Iterable[int]],
        drain_seconds: float = 0.0,
    ):
        links = self.status().get("links", [])
        outputs = path_to_switch_outputs(switch_path, links, leaf_ports)
        return self.reroute(
            group_id,
            dst_ip,
            outputs,
            source_dpid=int(switch_path[0]),
            drain_seconds=drain_seconds,
        )


def path_to_switch_outputs(
    switch_path: Sequence[int],
    links: Sequence[Mapping[str, int]],
    leaf_ports: Mapping[int, Iterable[int]],
):
    """Convert [1, 2, 4] and Ryu links into per-switch output ports."""

    path = [int(dpid) for dpid in switch_path]
    if len(path) < 2:
        raise ValueError("switch_path must contain at least two switches")
    outputs = {}
    for src, dst in zip(path, path[1:]):
        candidates = [
            int(link["src_port"])
            for link in links
            if int(link["src_dpid"]) == src and int(link["dst_dpid"]) == dst
        ]
        if len(candidates) != 1:
            raise ValueError(f"no unique directed link from {src} to {dst}")
        outputs[str(src)] = [candidates[0]]
    last = path[-1]
    ports = sorted({int(port) for port in leaf_ports.get(last, [])})
    if not ports:
        raise ValueError(f"leaf_ports has no ports for switch {last}")
    outputs[str(last)] = ports
    return outputs
