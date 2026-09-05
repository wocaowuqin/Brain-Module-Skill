"""Run the stateful online HRL planner in a dedicated Python process.

The worker owns the only HRL environment and resource ledger. Requests still
enter it serially in arrival order; process isolation only removes GIL and
allocator contention with the Mininet/Ryu deployment pipeline.
"""

from __future__ import annotations

import gc
import multiprocessing
from multiprocessing.connection import Connection
import threading
import traceback
from typing import Any


def _planner_worker(
    connection: Connection,
    planner_kwargs: dict[str, Any],
    cpu_affinity: list[int] | None,
) -> None:
    planner = None
    gc_runtime: dict[str, Any] = {"enabled": False}
    try:
        if cpu_affinity:
            try:
                import psutil

                process = psutil.Process()
                process.cpu_affinity([int(cpu) for cpu in cpu_affinity])
                if hasattr(psutil, "ABOVE_NORMAL_PRIORITY_CLASS"):
                    process.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
            except (ImportError, OSError, ValueError):
                pass
        from sdn.online_hrl_planner import OnlineLegacyHRLPlanner

        planner = OnlineLegacyHRLPlanner(**planner_kwargs)
        # The frozen model, NetworkX topology and environment form a large
        # long-lived object graph.  Scanning that graph at CPython's default
        # generation-0 threshold caused periodic 0.5-0.8 second planning
        # stalls.  Freeze it once, then collect only after substantially more
        # short-lived planning allocations.  Reference counting is unchanged.
        gc.collect()
        if hasattr(gc, "freeze"):
            gc.freeze()
        gc.set_threshold(50_000, 100, 100)
        gc_runtime = {
            "enabled": True,
            "frozen": bool(hasattr(gc, "freeze")),
            "threshold": list(gc.get_threshold()),
        }
        ready_metadata = planner.metadata()
        ready_metadata["planner_gc_tuning"] = gc_runtime
        connection.send({"ok": True, "kind": "ready", "value": ready_metadata})
        while True:
            message = connection.recv()
            operation = message.get("operation")
            if operation == "close":
                connection.send({"ok": True, "kind": "closed", "value": None})
                break
            if operation == "plan_next":
                value = planner.plan_next(message["request"])
            elif operation == "release":
                value = planner.release(int(message["request_id"]))
            elif operation == "metadata":
                value = planner.metadata()
            else:
                raise ValueError(f"unsupported isolated HRL operation: {operation}")
            connection.send({"ok": True, "kind": operation, "value": value})
    except EOFError:
        pass
    except BaseException as exc:
        try:
            connection.send({
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if planner is not None:
            planner.close()
        connection.close()


class IsolatedOnlineLegacyHRLPlanner:
    """Synchronous RPC facade for one stateful planner subprocess."""

    def __init__(
        self,
        *,
        reserve_cpu_affinity: bool = False,
        **planner_kwargs: Any,
    ) -> None:
        context = multiprocessing.get_context("spawn")
        self._parent_affinity: list[int] | None = None
        planner_affinity: list[int] | None = None
        if reserve_cpu_affinity:
            try:
                import psutil

                parent_process = psutil.Process()
                available = list(parent_process.cpu_affinity())
                if len(available) >= 4:
                    reserve_count = 2 if len(available) >= 8 else 1
                    planner_affinity = available[-reserve_count:]
                    parent_affinity = available[:-reserve_count]
                    parent_process.cpu_affinity(parent_affinity)
                    self._parent_affinity = available
            except (ImportError, OSError, ValueError):
                planner_affinity = None
        parent, child = context.Pipe(duplex=True)
        self._connection = parent
        self._rpc_lock = threading.Lock()
        self._process = context.Process(
            target=_planner_worker,
            args=(child, dict(planner_kwargs), planner_affinity),
            name="online-hrl-isolated",
            daemon=True,
        )
        self._closed = False
        self._planner_affinity = planner_affinity
        self._process.start()
        child.close()
        ready = self._receive()
        self._initial_metadata = dict(ready or {})
        self._planner_gc_tuning = dict(
            self._initial_metadata.get("planner_gc_tuning") or {}
        )

    def _receive(self) -> Any:
        try:
            response = self._connection.recv()
        except EOFError as exc:
            raise RuntimeError(
                f"isolated HRL process exited with code {self._process.exitcode}"
            ) from exc
        if not response.get("ok", False):
            raise RuntimeError(
                "isolated HRL planner failed: "
                f"{response.get('error_type')}: {response.get('error')}\n"
                f"{response.get('traceback', '')}"
            )
        return response.get("value")

    def _request(self, operation: str, **payload: Any) -> Any:
        if self._closed:
            raise RuntimeError("isolated HRL planner is closed")
        # Planning and deployment-cleanup threads may call plan_next/release at
        # the same time.  A duplex Pipe preserves messages but concurrent recv
        # calls do not preserve request/response ownership, so serialize each
        # complete RPC pair.  The worker remains the single ledger authority.
        with self._rpc_lock:
            self._connection.send({"operation": operation, **payload})
            return self._receive()

    def plan_next(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._request("plan_next", request=request)

    def release(self, request_id: int) -> bool:
        return bool(self._request("release", request_id=int(request_id)))

    def metadata(self) -> dict[str, Any]:
        metadata = dict(self._request("metadata"))
        metadata["process_isolation"] = True
        metadata["planner_pid"] = self._process.pid
        metadata["planner_cpu_affinity"] = self._planner_affinity
        metadata["planner_gc_tuning"] = self._planner_gc_tuning
        return metadata

    def close(self) -> None:
        if self._closed:
            return
        try:
            with self._rpc_lock:
                self._connection.send({"operation": "close"})
                self._receive()
        finally:
            self._closed = True
            self._connection.close()
            self._process.join(timeout=10.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5.0)
            if self._parent_affinity:
                try:
                    import psutil

                    psutil.Process().cpu_affinity(self._parent_affinity)
                except (ImportError, OSError, ValueError):
                    pass
