#!/usr/bin/env python3
"""Web dashboard for Orbital Compute Simulator.

Serves a single-page dashboard showing simulation results:
- Constellation map with satellite positions
- Power/thermal timeline charts
- Job scheduling Gantt chart
- Ground station contacts

Usage:
    python dashboard.py                    # Run sim + serve dashboard
    python dashboard.py --port 8080        # Custom port
    python dashboard.py --results sim.json # Load existing results
"""

import argparse
import http.server
import json
import os
import sys
import threading
import webbrowser
from datetime import datetime, timezone

from orbital_compute.simulator import Simulation, SimulationConfig


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Orbital Compute Simulator</title>
<style>
:root { --bg: #0a0e17; --card: #141b2d; --border: #1e2a45; --text: #e0e6f0;
        --dim: #6b7a99; --accent: #4fc3f7; --green: #66bb6a; --orange: #ffa726;
        --red: #ef5350; --purple: #ab47bc; }
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: var(--bg); color: var(--text); font-family: 'SF Mono', 'Fira Code', monospace;
       font-size: 13px; padding: 20px; }
h1 { font-size: 18px; color: var(--accent); margin-bottom: 4px; }
h2 { font-size: 14px; color: var(--dim); margin-bottom: 16px; font-weight: normal; }
h3 { font-size: 13px; color: var(--accent); margin-bottom: 8px; padding-bottom: 4px;
     border-bottom: 1px solid var(--border); }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
.grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 16px; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
.card-full { grid-column: 1 / -1; }
.stat { text-align: center; padding: 12px; }
.stat .value { font-size: 28px; font-weight: bold; color: var(--accent); }
.stat .label { font-size: 11px; color: var(--dim); margin-top: 4px; }
table { width: 100%; border-collapse: collapse; }
th { text-align: left; color: var(--dim); font-weight: normal; padding: 6px 8px;
     border-bottom: 1px solid var(--border); font-size: 11px; text-transform: uppercase; }
td { padding: 6px 8px; border-bottom: 1px solid var(--border); }
.bar { height: 8px; border-radius: 4px; background: var(--border); position: relative; }
.bar-fill { height: 100%; border-radius: 4px; position: absolute; left: 0; top: 0; }
.green { color: var(--green); }
.orange { color: var(--orange); }
.red { color: var(--red); }
canvas { width: 100%; height: 200px; }
.timeline { position: relative; height: 40px; background: var(--bg); border-radius: 4px;
            margin: 4px 0; overflow: hidden; }
.timeline-block { position: absolute; height: 100%; border-radius: 2px; }
.timeline-eclipse { background: rgba(239,83,80,0.3); }
.timeline-compute { background: rgba(79,195,247,0.5); }
.timeline-contact { background: rgba(102,187,106,0.5); }
.legend { display: flex; gap: 16px; margin-top: 8px; font-size: 11px; color: var(--dim); }
.legend-dot { width: 10px; height: 10px; border-radius: 2px; display: inline-block;
              margin-right: 4px; vertical-align: middle; }
.contact-list { max-height: 200px; overflow-y: auto; }
.contact-item { display: flex; justify-content: space-between; padding: 4px 0;
                border-bottom: 1px solid var(--border); font-size: 12px; }
</style>
</head>
<body>
<h1>ORBITAL COMPUTE SIMULATOR</h1>
<h2>Schedule compute jobs across satellite constellations</h2>

<div class="grid-3" id="stats"></div>
<div class="grid" id="main-panels"></div>
<div class="card card-full" id="timeline-panel"></div>
<div class="grid" id="detail-panels" style="margin-top:16px"></div>

<script>
const DATA = __DATA__;

function render() {
    const d = DATA;
    const s = d.scheduler;

    // Stats cards
    document.getElementById('stats').innerHTML = `
        <div class="card stat">
            <div class="value">${d.config.n_satellites}</div>
            <div class="label">SATELLITES</div>
        </div>
        <div class="card stat">
            <div class="value">${s.completed}/${s.total_jobs}</div>
            <div class="label">JOBS COMPLETED</div>
        </div>
        <div class="card stat">
            <div class="value">${d.fleet_utilization_pct}%</div>
            <div class="label">FLEET UTILIZATION</div>
        </div>
        <div class="card stat">
            <div class="value">${d.total_compute_hours}h</div>
            <div class="label">COMPUTE DELIVERED</div>
        </div>
        <div class="card stat">
            <div class="value">${d.preemption_events}</div>
            <div class="label">PREEMPTIONS</div>
        </div>
        <div class="card stat">
            <div class="value">${d.config.sim_hours}h</div>
            <div class="label">SIM DURATION</div>
        </div>
    `;

    // Satellite table
    const satRows = Object.entries(d.satellite_details).map(([name, det]) => {
        const compColor = det.compute_pct > 50 ? 'green' : det.compute_pct > 20 ? 'orange' : 'red';
        const tempColor = det.max_temp_c > 75 ? 'red' : det.max_temp_c > 60 ? 'orange' : 'green';
        const battColor = det.min_battery_pct < 25 ? 'red' : det.min_battery_pct < 40 ? 'orange' : 'green';
        return `<tr>
            <td>${name}</td>
            <td><span class="${compColor}">${det.compute_pct}%</span></td>
            <td>${det.eclipse_pct}%</td>
            <td><span class="${battColor}">${det.avg_battery_pct}%</span></td>
            <td><span class="${tempColor}">${det.avg_temp_c}°C</span></td>
            <td>${det.max_temp_c}°C</td>
            <td>${det.contact_windows || 0}</td>
            <td>${det.total_contact_minutes || 0}m</td>
        </tr>`;
    }).join('');

    // Jobs table
    const jobRows = d.completed_jobs.slice(0, 15).map(j => `<tr>
        <td>${j.job_id}</td>
        <td>${j.satellite}</td>
        <td>${(j.duration_s / 60).toFixed(1)}m</td>
        <td>${j.power_w}W</td>
    </tr>`).join('');

    document.getElementById('main-panels').innerHTML = `
        <div class="card">
            <h3>SATELLITE FLEET</h3>
            <table>
                <tr><th>Sat</th><th>Compute</th><th>Eclipse</th><th>Battery</th>
                    <th>Avg Temp</th><th>Max Temp</th><th>Contacts</th><th>Contact Time</th></tr>
                ${satRows}
            </table>
        </div>
        <div class="card">
            <h3>COMPLETED JOBS</h3>
            <table>
                <tr><th>Job</th><th>Satellite</th><th>Duration</th><th>Power</th></tr>
                ${jobRows}
            </table>
            ${d.completed_jobs.length > 15 ? '<div style="color:var(--dim);margin-top:8px">... and ' + (d.completed_jobs.length - 15) + ' more</div>' : ''}
        </div>
    `;

    // Timeline visualization
    renderTimelines(d);

    // Ground stations + power charts
    renderDetails(d);
}

function renderTimelines(d) {
    const panel = document.getElementById('timeline-panel');
    let html = '<h3>SATELLITE TIMELINES</h3>';

    const totalMinutes = d.config.sim_hours * 60;

    Object.entries(d.telemetry || {}).forEach(([name, tel]) => {
        html += `<div style="display:flex;align-items:center;gap:8px">
            <span style="width:60px;font-size:11px;color:var(--dim)">${name}</span>
            <div class="timeline" style="flex:1">`;

        // Eclipse blocks
        (tel.eclipse_windows || []).forEach(w => {
            const left = (w.start_min / totalMinutes * 100);
            const width = (w.duration_min / totalMinutes * 100);
            html += `<div class="timeline-block timeline-eclipse"
                         style="left:${left}%;width:${Math.max(width, 0.3)}%"
                         title="Eclipse: ${w.duration_min.toFixed(0)}min"></div>`;
        });

        // Compute blocks
        (tel.compute_windows || []).forEach(w => {
            const left = (w.start_min / totalMinutes * 100);
            const width = (w.duration_min / totalMinutes * 100);
            html += `<div class="timeline-block timeline-compute"
                         style="left:${left}%;width:${Math.max(width, 0.3)}%"
                         title="Computing: ${w.duration_min.toFixed(0)}min"></div>`;
        });

        // Contact blocks
        (tel.contact_windows || []).forEach(w => {
            const left = (w.start_min / totalMinutes * 100);
            const width = (w.duration_min / totalMinutes * 100);
            html += `<div class="timeline-block timeline-contact"
                         style="left:${left}%;width:${Math.max(width, 0.5)}%"
                         title="${w.station}: ${w.duration_min.toFixed(0)}min"></div>`;
        });

        html += `</div></div>`;
    });

    html += `<div class="legend">
        <span><span class="legend-dot" style="background:rgba(239,83,80,0.5)"></span> Eclipse</span>
        <span><span class="legend-dot" style="background:rgba(79,195,247,0.5)"></span> Computing</span>
        <span><span class="legend-dot" style="background:rgba(102,187,106,0.5)"></span> Ground Contact</span>
    </div>`;

    panel.innerHTML = html;
}

function renderDetails(d) {
    const panel = document.getElementById('detail-panels');

    // Ground station contacts
    let contactHtml = '<h3>GROUND STATION CONTACTS</h3><div class="contact-list">';
    (d.ground_contacts || []).slice(0, 30).forEach(c => {
        contactHtml += `<div class="contact-item">
            <span>${c.satellite} → ${c.station}</span>
            <span>${c.start_time} (${c.duration_min.toFixed(1)}m, max elev ${c.max_elev}°)</span>
        </div>`;
    });
    contactHtml += '</div>';

    // Power summary
    let powerHtml = '<h3>POWER BUDGET</h3>';
    powerHtml += `<table>
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>Solar Panel Output</td><td>${d.power_config?.solar_watts || 2000}W</td></tr>
        <tr><td>Battery Capacity</td><td>${d.power_config?.battery_wh || 5000}Wh</td></tr>
        <tr><td>Housekeeping Load</td><td>${d.power_config?.housekeeping_watts || 150}W</td></tr>
        <tr><td>Avg Eclipse Duration</td><td>${d.avg_eclipse_minutes?.toFixed(1) || '~35'}min</td></tr>
        <tr><td>Orbital Period</td><td>~95min (LEO 550km)</td></tr>
        <tr><td>Max Sustainable Heat</td><td>${d.max_sustainable_heat_w?.toFixed(0) || 'N/A'}W</td></tr>
    </table>`;

    panel.innerHTML = `
        <div class="card">${contactHtml}</div>
        <div class="card">${powerHtml}</div>
    `;
}

render();
</script>
</body>
</html>"""


def build_dashboard_data(sim):
    """Build the data object for the dashboard, including timeline telemetry."""
    results = sim.results.copy()

    # Build timeline data from telemetry
    telemetry = {}
    start_time = sim.config.start_time

    for node in sim.nodes:
        eclipse_windows = []
        compute_windows = []

        # Detect eclipse windows from telemetry
        in_eclipse = False
        eclipse_start_min = 0
        in_compute = False
        compute_start_min = 0

        for h in node.power_history:
            t = datetime.fromisoformat(h["time"])
            minute = (t - start_time).total_seconds() / 60

            if h.get("in_eclipse", h["solar_w"] == 0) and not in_eclipse:
                eclipse_start_min = minute
                in_eclipse = True
            elif not h.get("in_eclipse", h["solar_w"] == 0) and in_eclipse:
                eclipse_windows.append({"start_min": eclipse_start_min,
                                        "duration_min": minute - eclipse_start_min})
                in_eclipse = False

            if h.get("computing", False) and not in_compute:
                compute_start_min = minute
                in_compute = True
            elif not h.get("computing", False) and in_compute:
                compute_windows.append({"start_min": compute_start_min,
                                        "duration_min": minute - compute_start_min})
                in_compute = False

        # Close open windows
        total_min = sim.config.sim_duration_hours * 60
        if in_eclipse:
            eclipse_windows.append({"start_min": eclipse_start_min,
                                    "duration_min": total_min - eclipse_start_min})
        if in_compute:
            compute_windows.append({"start_min": compute_start_min,
                                    "duration_min": total_min - compute_start_min})

        # Contact windows
        contact_vis = []
        for cw in node.contact_windows:
            start_min = (cw.start - start_time).total_seconds() / 60
            dur_min = cw.duration_seconds / 60
            contact_vis.append({"start_min": start_min, "duration_min": dur_min,
                                "station": cw.station_name})

        telemetry[node.name] = {
            "eclipse_windows": eclipse_windows,
            "compute_windows": compute_windows,
            "contact_windows": contact_vis,
        }

    results["telemetry"] = telemetry

    # Ground contacts list
    all_contacts = []
    for node in sim.nodes:
        for cw in node.contact_windows:
            all_contacts.append({
                "satellite": node.name,
                "station": cw.station_name,
                "start_time": cw.start.strftime("%H:%M"),
                "duration_min": cw.duration_seconds / 60,
                "max_elev": cw.max_elevation_deg,
            })
    all_contacts.sort(key=lambda c: c["start_time"])
    results["ground_contacts"] = all_contacts

    # Power config
    results["power_config"] = {
        "solar_watts": sim.config.solar_panel_watts,
        "battery_wh": sim.config.battery_capacity_wh,
        "housekeeping_watts": sim.config.housekeeping_watts,
    }

    return results


def serve_dashboard(html_content, port=3000):
    """Serve the dashboard HTML."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html_content.encode())

        def log_message(self, format, *args):
            pass  # Quiet

    server = http.server.HTTPServer(("", port), Handler)
    print(f"\n  Dashboard: http://localhost:{port}")
    print("  Press Ctrl+C to stop\n")

    # Open browser
    threading.Timer(0.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Dashboard stopped.")
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Orbital Compute Dashboard")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--results", type=str, help="Load existing results JSON")
    parser.add_argument("--sats", type=int, default=6)
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--jobs", type=int, default=20)
    parser.add_argument("--no-serve", action="store_true", help="Generate HTML file only")
    args = parser.parse_args()

    if args.results:
        with open(args.results) as f:
            data = json.load(f)
    else:
        config = SimulationConfig(
            n_satellites=args.sats,
            sim_duration_hours=args.hours,
            n_jobs=args.jobs,
        )
        sim = Simulation(config)
        sim.setup()
        sim.run()
        sim.print_report()
        data = build_dashboard_data(sim)

    # Inject data into HTML
    html = DASHBOARD_HTML.replace("__DATA__", json.dumps(data))

    if args.no_serve:
        with open("dashboard.html", "w") as f:
            f.write(html)
        print("  Dashboard saved to dashboard.html")
    else:
        serve_dashboard(html, args.port)


if __name__ == "__main__":
    main()
