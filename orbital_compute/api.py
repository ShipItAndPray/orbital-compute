from __future__ import annotations

"""REST API for the orbital compute simulator.

Uses only Python's built-in http.server — no Flask or other dependencies.

Endpoints
---------
GET  /status      — constellation status overview
GET  /satellites  — list all satellites with current state
GET  /jobs        — list all jobs (pending, running, completed)
POST /jobs        — submit a new job (JSON body: {"workload": "<type>"})
GET  /contacts    — upcoming ground-station contact windows
GET  /metrics     — fleet utilization, job throughput, power stats
"""

import json
import threading
import traceback
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional
from urllib.parse import urlparse, parse_qs

from .scheduler import ComputeJob, JobStatus, JobType
from .simulator import Simulation, SimulationConfig, SatelliteNode
from .workloads import WORKLOAD_CATALOG, create_job, WorkloadGenerator


class SimulationServer:
    """Wraps a Simulation instance and exposes it via REST."""

    def __init__(self, sim: Simulation, host: str = "127.0.0.1", port: int = 8080):
        self.sim = sim
        self.host = host
        self.port = port
        self._server: Optional[HTTPServer] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Data accessors (thread-safe reads of simulation state)
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        stats = self.sim.scheduler.stats()
        nodes_summary = []
        for node in self.sim.nodes:
            last_power = node.power_history[-1] if node.power_history else {}
            last_thermal = node.thermal_history[-1] if node.thermal_history else {}
            running_job = self.sim.scheduler.running_jobs.get(node.name)
            nodes_summary.append({
                "name": node.name,
                "battery_pct": last_power.get("battery_pct", 0),
                "solar_w": last_power.get("solar_w", 0),
                "temp_c": last_thermal.get("temp_c", 0),
                "computing": last_power.get("computing", False),
                "in_eclipse": last_power.get("in_eclipse", False),
                "running_job": running_job.job_id if running_job else None,
            })
        return {
            "time": self.sim.current_time.isoformat(),
            "scheduler": stats,
            "satellites": nodes_summary,
        }

    def get_satellites(self) -> list[dict]:
        result = []
        for node in self.sim.nodes:
            ts = max(node.total_steps, 1)
            last_power = node.power_history[-1] if node.power_history else {}
            last_thermal = node.thermal_history[-1] if node.thermal_history else {}
            running_job = self.sim.scheduler.running_jobs.get(node.name)

            result.append({
                "name": node.name,
                "orbit": {
                    "altitude_km": node.satellite.altitude_km,
                    "inclination_deg": node.satellite.inclination_deg,
                },
                "power": {
                    "battery_pct": last_power.get("battery_pct", 0),
                    "solar_w": last_power.get("solar_w", 0),
                    "load_w": last_power.get("load_w", 0),
                    "in_eclipse": last_power.get("in_eclipse", False),
                },
                "thermal": {
                    "temp_c": last_thermal.get("temp_c", 0),
                    "heat_w": last_thermal.get("heat_w", 0),
                    "radiated_w": last_thermal.get("radiated_w", 0),
                    "throttle_pct": last_thermal.get("throttle", 0),
                },
                "compute": {
                    "utilization_pct": round(node.compute_steps / ts * 100, 1),
                    "running_job": running_job.job_id if running_job else None,
                },
                "contacts": {
                    "total_windows": len(node.contact_windows),
                    "total_minutes": round(
                        sum(w.duration_seconds for w in node.contact_windows) / 60, 1
                    ),
                },
            })
        return result

    def get_jobs(self) -> dict:
        pending = []
        for j in self.sim.scheduler.job_queue:
            pending.append(_job_to_dict(j))

        running = []
        for sat_name, j in self.sim.scheduler.running_jobs.items():
            d = _job_to_dict(j)
            d["assigned_satellite"] = sat_name
            running.append(d)

        completed = []
        for j in self.sim.scheduler.completed_jobs:
            completed.append(_job_to_dict(j))

        return {
            "pending": pending,
            "running": running,
            "completed": completed,
            "total": len(pending) + len(running) + len(completed),
        }

    def submit_job(self, workload_key: str) -> dict:
        """Submit a new job to the scheduler."""
        if workload_key not in WORKLOAD_CATALOG:
            return {"error": f"Unknown workload type: {workload_key}",
                    "available": list(WORKLOAD_CATALOG.keys())}

        with self._lock:
            job = create_job(workload_key, submit_time=self.sim.current_time)
            self.sim.scheduler.submit_job(job)

        return {"submitted": _job_to_dict(job)}

    def get_contacts(self) -> list[dict]:
        result = []
        for node in self.sim.nodes:
            windows = []
            for w in node.contact_windows:
                windows.append({
                    "ground_station": w.ground_station,
                    "start": w.start_time.isoformat(),
                    "end": w.end_time.isoformat(),
                    "duration_s": round(w.duration_seconds, 1),
                    "max_elevation_deg": round(w.max_elevation_deg, 1),
                })
            result.append({
                "satellite": node.name,
                "windows": windows,
            })
        return result

    def get_metrics(self) -> dict:
        stats = self.sim.scheduler.stats()

        total_compute_s = sum(j.duration_seconds for j in self.sim.scheduler.completed_jobs)
        total_possible_s = (
            self.sim.config.sim_duration_hours * 3600 * self.sim.config.n_satellites
        )

        # Power stats across fleet
        avg_batteries = []
        avg_temps = []
        max_temps = []
        for node in self.sim.nodes:
            if node.power_history:
                avg_batteries.append(
                    sum(h["battery_pct"] for h in node.power_history) / len(node.power_history)
                )
            if node.thermal_history:
                avg_temps.append(
                    sum(h["temp_c"] for h in node.thermal_history) / len(node.thermal_history)
                )
                max_temps.append(max(h["temp_c"] for h in node.thermal_history))

        return {
            "fleet_utilization_pct": round(
                total_compute_s / max(total_possible_s, 1) * 100, 1
            ),
            "total_compute_hours": round(total_compute_s / 3600, 2),
            "jobs": {
                "submitted": stats["total_jobs"],
                "completed": stats["completed"],
                "running": stats["running"],
                "queued": stats["queued"],
                "completion_rate_pct": round(
                    stats["completed"] / max(stats["total_jobs"], 1) * 100, 1
                ),
            },
            "preemptions": stats["preempted"],
            "idle_steps": stats["idle_steps"],
            "charge_steps": stats["charge_steps"],
            "power": {
                "avg_battery_pct": round(
                    sum(avg_batteries) / max(len(avg_batteries), 1) * 100, 1
                ),
            },
            "thermal": {
                "avg_temp_c": round(
                    sum(avg_temps) / max(len(avg_temps), 1), 1
                ),
                "max_temp_c": round(max(max_temps) if max_temps else 0, 1),
            },
        }

    # ------------------------------------------------------------------
    # HTTP server
    # ------------------------------------------------------------------

    def start(self, blocking: bool = True):
        """Start the REST API server."""
        server_ref = self  # capture for handler

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                # Quieter logging
                pass

            def _send_json(self, data: dict, status: int = 200):
                body = json.dumps(data, indent=2).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                parsed = urlparse(self.path)
                path = parsed.path.rstrip("/")

                try:
                    if path == "" or path == "/status":
                        self._send_json(server_ref.get_status())
                    elif path == "/satellites":
                        self._send_json(server_ref.get_satellites())
                    elif path == "/jobs":
                        self._send_json(server_ref.get_jobs())
                    elif path == "/contacts":
                        self._send_json(server_ref.get_contacts())
                    elif path == "/metrics":
                        self._send_json(server_ref.get_metrics())
                    else:
                        self._send_json({"error": "Not found", "endpoints": [
                            "GET /status", "GET /satellites", "GET /jobs",
                            "POST /jobs", "GET /contacts", "GET /metrics",
                        ]}, 404)
                except Exception as e:
                    self._send_json({"error": str(e)}, 500)

            def do_POST(self):
                parsed = urlparse(self.path)
                path = parsed.path.rstrip("/")

                try:
                    if path == "/jobs":
                        content_len = int(self.headers.get("Content-Length", 0))
                        if content_len == 0:
                            self._send_json(
                                {"error": "Missing JSON body. Send {\"workload\": \"<type>\"}",
                                 "available_workloads": list(WORKLOAD_CATALOG.keys())},
                                400,
                            )
                            return

                        raw = self.rfile.read(content_len)
                        body = json.loads(raw.decode("utf-8"))
                        workload = body.get("workload")
                        if not workload:
                            self._send_json(
                                {"error": "Missing 'workload' field",
                                 "available_workloads": list(WORKLOAD_CATALOG.keys())},
                                400,
                            )
                            return

                        result = server_ref.submit_job(workload)
                        status = 201 if "submitted" in result else 400
                        self._send_json(result, status)
                    else:
                        self._send_json({"error": "Not found"}, 404)
                except json.JSONDecodeError:
                    self._send_json({"error": "Invalid JSON"}, 400)
                except Exception as e:
                    self._send_json({"error": str(e),
                                     "traceback": traceback.format_exc()}, 500)

        self._server = HTTPServer((self.host, self.port), Handler)
        print(f"Orbital Compute API running on http://{self.host}:{self.port}")
        print(f"  GET  /status      — constellation status")
        print(f"  GET  /satellites  — satellite details")
        print(f"  GET  /jobs        — all jobs")
        print(f"  POST /jobs        — submit job  {{\"workload\": \"image_classification\"}}")
        print(f"  GET  /contacts    — ground station contacts")
        print(f"  GET  /metrics     — fleet metrics")

        if blocking:
            try:
                self._server.serve_forever()
            except KeyboardInterrupt:
                print("\nShutting down API server.")
                self._server.shutdown()
        else:
            thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            thread.start()
            return thread

    def stop(self):
        if self._server:
            self._server.shutdown()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _job_to_dict(job: ComputeJob) -> dict:
    return {
        "job_id": job.job_id,
        "name": job.name,
        "status": job.status.value,
        "power_watts": round(job.power_watts, 1),
        "duration_seconds": round(job.duration_seconds, 1),
        "progress_seconds": round(job.progress_seconds, 1),
        "remaining_seconds": round(job.remaining_seconds, 1),
        "priority": job.priority,
        "job_type": job.job_type.value,
        "deadline": job.deadline.isoformat() if job.deadline else None,
        "data_downlink_mb": job.data_downlink_mb,
        "checkpointable": job.checkpointable,
        "assigned_satellite": job.assigned_satellite,
    }


# ---------------------------------------------------------------------------
# Self-test (run sim, start API, test endpoints, shut down)
# ---------------------------------------------------------------------------

def _self_test():
    import urllib.request
    import time

    print("=" * 60)
    print("  ORBITAL COMPUTE — API SELF-TEST")
    print("=" * 60)

    # Run a quick sim
    from .workloads import WorkloadGenerator
    cfg = SimulationConfig(n_satellites=3, sim_duration_hours=2.0, n_jobs=0)
    sim = Simulation(cfg)
    sim.setup()

    # Use realistic workloads instead of default random ones
    gen = WorkloadGenerator(seed=99)
    jobs = gen.generate_batch(15, cfg.start_time, duration_hours=2.0)
    sim.scheduler.submit_jobs(jobs)

    sim.run()

    # Start API in background
    port = 18932  # unlikely to conflict
    server = SimulationServer(sim, port=port)
    server.start(blocking=False)
    time.sleep(0.3)

    base = f"http://127.0.0.1:{port}"
    errors = []

    def get(endpoint):
        url = f"{base}{endpoint}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                return resp.status, data
        except Exception as e:
            return None, str(e)

    def post(endpoint, body):
        url = f"{base}{endpoint}"
        try:
            payload = json.dumps(body).encode()
            req = urllib.request.Request(url, data=payload, method="POST",
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                return resp.status, data
        except urllib.error.HTTPError as e:
            data = json.loads(e.read().decode())
            return e.code, data
        except Exception as e:
            return None, str(e)

    # Test GET endpoints
    for ep in ["/status", "/satellites", "/jobs", "/contacts", "/metrics"]:
        status, data = get(ep)
        if status == 200 and isinstance(data, (dict, list)):
            print(f"  GET {ep:15s}  [{status}] OK")
        else:
            errors.append(f"GET {ep}: status={status}")
            print(f"  GET {ep:15s}  FAIL ({status})")

    # Test POST /jobs
    status, data = post("/jobs", {"workload": "sar_processing"})
    if status == 201 and "submitted" in data:
        print(f"  POST /jobs           [{status}] OK — {data['submitted']['job_id']}")
    else:
        errors.append(f"POST /jobs: status={status}, data={data}")
        print(f"  POST /jobs           FAIL ({status})")

    # Test POST /jobs with bad workload
    status, data = post("/jobs", {"workload": "nonexistent"})
    if status == 400 and "error" in data:
        print(f"  POST /jobs (bad)     [{status}] OK — error handled")
    else:
        errors.append(f"POST /jobs bad: status={status}")
        print(f"  POST /jobs (bad)     FAIL ({status})")

    # Test 404
    status, data = get("/nonexistent")
    if status == 404:
        print(f"  GET /nonexistent     [{status}] OK — 404 handled")
    else:
        errors.append(f"GET /nonexistent: status={status}")
        print(f"  GET /nonexistent     FAIL ({status})")

    server.stop()

    print(f"\n{'=' * 60}")
    if errors:
        print(f"  FAIL — {len(errors)} error(s)")
        for e in errors:
            print(f"    {e}")
    else:
        print("  PASS — all API endpoints OK")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    _self_test()
