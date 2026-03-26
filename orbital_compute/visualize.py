from __future__ import annotations

"""3D orbit visualization — generates a self-contained HTML file with Three.js.

No additional Python dependencies required. Loads Three.js from CDN.
"""

import json
import math
import os
import webbrowser
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np

from .orbit import Satellite, EARTH_RADIUS_KM, sun_position_eci, is_in_eclipse, eci_to_lla
from .ground_stations import GroundStation, DEFAULT_GROUND_STATIONS
from .isl import has_line_of_sight, MAX_LINK_RANGE_KM


# Scale factor: render Earth radius as 1.0 unit in Three.js
SCALE = 1.0 / EARTH_RADIUS_KM


def _propagate_satellites(
    satellites: List[Satellite],
    start: datetime,
    hours: float,
    step_seconds: float = 60.0,
) -> dict:
    """Propagate all satellites and build JSON-serializable data."""
    n_steps = int(hours * 3600 / step_seconds) + 1
    timestamps = []
    sat_data = []

    for i, sat in enumerate(satellites):
        positions = []
        for step_i in range(n_steps):
            dt = start + timedelta(seconds=step_i * step_seconds)
            if i == 0:
                timestamps.append(dt.isoformat())
            try:
                pos = sat.position_at(dt)
                positions.append({
                    "x": round(pos.x_km * SCALE, 6),
                    "y": round(pos.z_km * SCALE, 6),  # Three.js Y = up = ECI Z
                    "z": round(pos.y_km * SCALE, 6),  # Three.js Z = ECI Y
                    "lat": round(pos.lat_deg, 2),
                    "lon": round(pos.lon_deg, 2),
                    "alt": round(pos.altitude_km, 1),
                    "eclipse": bool(pos.in_eclipse),
                })
            except Exception:
                positions.append(None)

        sat_data.append({
            "name": sat.name,
            "positions": positions,
        })

    return {"timestamps": timestamps, "satellites": sat_data}


def _propagate_orbit_track(sat: Satellite, dt: datetime, n_points: int = 120) -> list:
    """Propagate one full orbit for the track line."""
    # Orbital period ~ 2*pi*sqrt(a^3/mu), approximate from mean motion
    # For ~550km LEO, period ~ 95 min
    period_seconds = 95.6 * 60  # Good enough for Starlink-like orbits
    points = []
    for i in range(n_points + 1):
        t = dt + timedelta(seconds=i * period_seconds / n_points)
        try:
            pos = sat.position_at(t)
            points.append({
                "x": round(pos.x_km * SCALE, 6),
                "y": round(pos.z_km * SCALE, 6),
                "z": round(pos.y_km * SCALE, 6),
            })
        except Exception:
            pass
    return points


def _ground_station_data(stations: List[GroundStation]) -> list:
    """Convert ground stations to 3D positions on Earth surface."""
    result = []
    for gs in stations:
        lat = math.radians(gs.lat_deg)
        lon = math.radians(gs.lon_deg)
        r = 1.005  # Slightly above surface
        x = r * math.cos(lat) * math.cos(lon)
        y = r * math.sin(lat)
        z = r * math.cos(lat) * math.sin(lon)
        result.append({
            "name": gs.name,
            "x": round(x, 6),
            "y": round(y, 6),
            "z": round(z, 6),
            "lat": gs.lat_deg,
            "lon": gs.lon_deg,
        })
    return result


def _compute_isl_links(
    satellites: List[Satellite],
    start: datetime,
    hours: float,
    step_seconds: float = 60.0,
) -> list:
    """Compute ISL links at each timestep."""
    n_steps = int(hours * 3600 / step_seconds) + 1
    all_links = []  # One list per timestep

    for step_i in range(n_steps):
        dt = start + timedelta(seconds=step_i * step_seconds)
        # Get ECI positions
        positions = {}
        for sat in satellites:
            try:
                pos = sat.position_at(dt)
                positions[sat.name] = np.array([pos.x_km, pos.y_km, pos.z_km])
            except Exception:
                pass

        step_links = []
        names = list(positions.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                p1 = positions[names[i]]
                p2 = positions[names[j]]
                dist = float(np.linalg.norm(p2 - p1))
                if dist <= MAX_LINK_RANGE_KM and has_line_of_sight(p1, p2):
                    step_links.append([i, j])

        all_links.append(step_links)

    return all_links


def generate_3d_visualization(
    satellites: List[Satellite],
    ground_stations: Optional[List[GroundStation]] = None,
    hours: float = 2.0,
    step_seconds: float = 60.0,
    start_time: Optional[datetime] = None,
    output: str = "orbit_3d.html",
    open_browser: bool = False,
) -> str:
    """Generate a self-contained HTML file with 3D orbit visualization.

    Parameters
    ----------
    satellites : list of Satellite
        Satellites to visualize.
    ground_stations : list of GroundStation, optional
        Ground stations to show. Defaults to DEFAULT_GROUND_STATIONS.
    hours : float
        Duration to visualize in hours.
    step_seconds : float
        Time step for propagation.
    start_time : datetime, optional
        Start time. Defaults to 2026-03-26 00:00 UTC.
    output : str
        Output HTML file path.
    open_browser : bool
        Whether to open the file in the default browser.

    Returns
    -------
    str
        Path to the generated HTML file.
    """
    if ground_stations is None:
        ground_stations = DEFAULT_GROUND_STATIONS
    if start_time is None:
        start_time = datetime(2026, 3, 26, 0, 0, 0, tzinfo=timezone.utc)

    print(f"Propagating {len(satellites)} satellites for {hours}h...")
    sat_data = _propagate_satellites(satellites, start_time, hours, step_seconds)

    print("Computing orbit tracks...")
    orbit_tracks = []
    for sat in satellites:
        track = _propagate_orbit_track(sat, start_time)
        orbit_tracks.append(track)

    print("Computing ISL links...")
    isl_links = _compute_isl_links(satellites, start_time, hours, step_seconds)

    gs_data = _ground_station_data(ground_stations)

    # Build the JSON payload
    viz_data = {
        "timestamps": sat_data["timestamps"],
        "satellites": sat_data["satellites"],
        "orbitTracks": orbit_tracks,
        "groundStations": gs_data,
        "islLinks": isl_links,
        "earthRadius": 1.0,
        "satScale": SCALE,
    }

    data_json = json.dumps(viz_data, separators=(",", ":"))

    html = _build_html(data_json)

    output_path = os.path.abspath(output)
    with open(output_path, "w") as f:
        f.write(html)

    print(f"Saved 3D visualization to {output_path}")

    if open_browser:
        webbrowser.open("file://" + output_path)

    return output_path


def _build_html(data_json: str) -> str:
    """Build the complete HTML file with embedded Three.js visualization."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Orbital Compute Simulator — 3D Visualization</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #000; color: #eee; font-family: 'Segoe UI', sans-serif; overflow: hidden; }}
  #canvas-container {{ width: 100vw; height: 100vh; }}
  #controls {{
    position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%);
    background: rgba(0,0,0,0.75); padding: 12px 24px; border-radius: 8px;
    display: flex; align-items: center; gap: 16px; z-index: 10;
    border: 1px solid #333;
  }}
  #controls label {{ font-size: 13px; white-space: nowrap; }}
  #time-slider {{ width: 400px; cursor: pointer; }}
  #time-display {{ font-size: 13px; min-width: 180px; text-align: center; font-variant-numeric: tabular-nums; }}
  #play-btn {{
    background: #2a6; color: #fff; border: none; padding: 6px 14px;
    border-radius: 4px; cursor: pointer; font-size: 13px;
  }}
  #play-btn:hover {{ background: #3b7; }}
  #legend {{
    position: absolute; top: 20px; left: 20px;
    background: rgba(0,0,0,0.75); padding: 14px 18px; border-radius: 8px;
    font-size: 13px; z-index: 10; border: 1px solid #333;
  }}
  #legend h3 {{ margin-bottom: 8px; font-size: 14px; color: #aaa; }}
  .legend-item {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
  #info-panel {{
    position: absolute; top: 20px; right: 20px;
    background: rgba(0,0,0,0.85); padding: 16px 20px; border-radius: 8px;
    font-size: 13px; z-index: 10; min-width: 220px; display: none;
    border: 1px solid #444;
  }}
  #info-panel h3 {{ color: #6cf; margin-bottom: 10px; font-size: 15px; }}
  #info-panel .row {{ display: flex; justify-content: space-between; margin: 3px 0; }}
  #info-panel .row .label {{ color: #888; }}
  #info-panel .row .value {{ color: #eee; font-variant-numeric: tabular-nums; }}
  #title {{
    position: absolute; top: 20px; left: 50%; transform: translateX(-50%);
    font-size: 16px; color: #888; z-index: 10; letter-spacing: 1px;
  }}
</style>
</head>
<body>
<div id="canvas-container"></div>
<div id="title">ORBITAL COMPUTE SIMULATOR</div>
<div id="legend">
  <h3>Satellite Status</h3>
  <div class="legend-item"><span class="legend-dot" style="background:#4f4;"></span> Sunlit + Computing</div>
  <div class="legend-item"><span class="legend-dot" style="background:#ff0;"></span> Sunlit + Idle</div>
  <div class="legend-item"><span class="legend-dot" style="background:#f44;"></span> Eclipse</div>
  <div class="legend-item"><span class="legend-dot" style="background:#48f;"></span> Eclipse + Computing</div>
  <div class="legend-item"><span class="legend-dot" style="background:#fff;"></span> Ground Station</div>
  <div class="legend-item"><span class="legend-dot" style="background:#555;"></span> Orbit Track</div>
  <div class="legend-item" style="margin-top:6px;"><span style="width:20px;height:2px;background:#284;display:inline-block;"></span> ISL Link</div>
</div>
<div id="info-panel">
  <h3 id="info-name">—</h3>
  <div class="row"><span class="label">Lat</span><span class="value" id="info-lat">—</span></div>
  <div class="row"><span class="label">Lon</span><span class="value" id="info-lon">—</span></div>
  <div class="row"><span class="label">Alt</span><span class="value" id="info-alt">—</span></div>
  <div class="row"><span class="label">Eclipse</span><span class="value" id="info-eclipse">—</span></div>
  <div class="row"><span class="label">Status</span><span class="value" id="info-status">—</span></div>
</div>
<div id="controls">
  <button id="play-btn">Play</button>
  <label>Time:</label>
  <input type="range" id="time-slider" min="0" max="100" value="0">
  <div id="time-display">—</div>
</div>

<script type="importmap">
{{
  "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }}
}}
</script>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const DATA = {data_json};

// ---- Scene setup ----
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set(0, 1.5, 3.5);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 1.2;
controls.maxDistance = 20;

// ---- Lighting ----
scene.add(new THREE.AmbientLight(0x404050, 1.0));
const sunLight = new THREE.DirectionalLight(0xffffff, 1.5);
sunLight.position.set(5, 3, 5);
scene.add(sunLight);

// ---- Stars ----
const starGeo = new THREE.BufferGeometry();
const starVerts = new Float32Array(3000);
for (let i = 0; i < 3000; i++) {{
  starVerts[i] = (Math.random() - 0.5) * 80;
}}
starGeo.setAttribute('position', new THREE.BufferAttribute(starVerts, 3));
const starMat = new THREE.PointsMaterial({{ color: 0xffffff, size: 0.05 }});
scene.add(new THREE.Points(starGeo, starMat));

// ---- Earth ----
const earthGroup = new THREE.Group();
scene.add(earthGroup);

// Blue sphere
const earthGeo = new THREE.SphereGeometry(1.0, 64, 48);
const earthMat = new THREE.MeshPhongMaterial({{
  color: 0x1a3a6a,
  emissive: 0x050510,
  specular: 0x222244,
  shininess: 15,
}});
const earth = new THREE.Mesh(earthGeo, earthMat);
earthGroup.add(earth);

// Grid lines (lat/lon)
const gridMat = new THREE.LineBasicMaterial({{ color: 0x2a4a7a, transparent: true, opacity: 0.3 }});
// Latitude lines
for (let lat = -60; lat <= 60; lat += 30) {{
  const points = [];
  const r = Math.cos(lat * Math.PI / 180) * 1.002;
  const y = Math.sin(lat * Math.PI / 180) * 1.002;
  for (let i = 0; i <= 72; i++) {{
    const lon = (i / 72) * Math.PI * 2;
    points.push(new THREE.Vector3(r * Math.cos(lon), y, r * Math.sin(lon)));
  }}
  const geo = new THREE.BufferGeometry().setFromPoints(points);
  earthGroup.add(new THREE.Line(geo, gridMat));
}}
// Longitude lines
for (let lon = 0; lon < 360; lon += 30) {{
  const points = [];
  const lonRad = lon * Math.PI / 180;
  for (let i = 0; i <= 48; i++) {{
    const lat = (i / 48) * Math.PI - Math.PI / 2;
    const r = Math.cos(lat) * 1.002;
    const y = Math.sin(lat) * 1.002;
    points.push(new THREE.Vector3(r * Math.cos(lonRad), y, r * Math.sin(lonRad)));
  }}
  const geo = new THREE.BufferGeometry().setFromPoints(points);
  earthGroup.add(new THREE.Line(geo, gridMat));
}}

// Simple continent outlines (rough equator-area land patches)
const landMat = new THREE.MeshPhongMaterial({{ color: 0x2a7a3a, emissive: 0x0a200a }});
function addLandPatch(latDeg, lonDeg, sizeDeg) {{
  const lat = latDeg * Math.PI / 180;
  const lon = lonDeg * Math.PI / 180;
  const r = 1.003;
  const patchGeo = new THREE.CircleGeometry(sizeDeg * Math.PI / 180, 8);
  const patch = new THREE.Mesh(patchGeo, landMat);
  patch.position.set(
    r * Math.cos(lat) * Math.cos(lon),
    r * Math.sin(lat),
    r * Math.cos(lat) * Math.sin(lon)
  );
  patch.lookAt(0, 0, 0);
  patch.rotateY(Math.PI);
  earthGroup.add(patch);
}}
// Rough continent centers
const continents = [
  [48, 10, 0.35], [10, 20, 0.45], [-15, 25, 0.3],   // Europe, W.Africa, S.Africa
  [35, 100, 0.5], [20, 78, 0.35], [55, 90, 0.4],     // China, India, Russia
  [40, -100, 0.5], [0, -60, 0.5], [-15, -50, 0.45],  // N.America, C.America, S.America
  [-25, 135, 0.4], [65, -20, 0.15],                    // Australia, Iceland
];
continents.forEach(c => addLandPatch(c[0], c[1], c[2]));

// ---- Ground Stations ----
const gsMat = new THREE.MeshBasicMaterial({{ color: 0xffffff }});
const gsGeo = new THREE.SphereGeometry(0.012, 8, 8);
const gsLabels = [];
DATA.groundStations.forEach(gs => {{
  const mesh = new THREE.Mesh(gsGeo, gsMat);
  mesh.position.set(gs.x, gs.y, gs.z);
  earthGroup.add(mesh);

  // Label sprite
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.font = '24px sans-serif';
  ctx.fillStyle = '#aaaaaa';
  ctx.fillText(gs.name, 4, 40);
  const tex = new THREE.CanvasTexture(canvas);
  const spriteMat = new THREE.SpriteMaterial({{ map: tex, transparent: true, depthTest: false }});
  const sprite = new THREE.Sprite(spriteMat);
  sprite.scale.set(0.3, 0.075, 1);
  sprite.position.set(gs.x * 1.06, gs.y * 1.06 + 0.03, gs.z * 1.06);
  earthGroup.add(sprite);
}});

// ---- Orbit Tracks ----
const trackMat = new THREE.LineBasicMaterial({{ color: 0x555555, transparent: true, opacity: 0.4 }});
DATA.orbitTracks.forEach(track => {{
  if (!track || track.length < 2) return;
  const points = track.map(p => new THREE.Vector3(p.x, p.y, p.z));
  const geo = new THREE.BufferGeometry().setFromPoints(points);
  scene.add(new THREE.Line(geo, trackMat));
}});

// ---- Satellite dots ----
const satMeshes = [];
const satGeo = new THREE.SphereGeometry(0.018, 12, 12);
const satColors = {{
  sunlit_compute: new THREE.Color(0x44ff44),
  sunlit_idle: new THREE.Color(0xffff00),
  eclipse: new THREE.Color(0xff4444),
  eclipse_compute: new THREE.Color(0x4488ff),
}};

DATA.satellites.forEach((sat, i) => {{
  const mat = new THREE.MeshBasicMaterial({{ color: 0xffff00 }});
  const mesh = new THREE.Mesh(satGeo, mat);
  mesh.userData = {{ satIndex: i, name: sat.name }};
  scene.add(mesh);
  satMeshes.push(mesh);
}});

// ---- ISL link lines ----
let islLineGroup = new THREE.Group();
scene.add(islLineGroup);
const islMat = new THREE.LineBasicMaterial({{ color: 0x227744, transparent: true, opacity: 0.5 }});

// ---- State ----
let currentStep = 0;
const maxStep = DATA.timestamps.length - 1;
let playing = false;
let playSpeed = 1;
let lastFrameTime = 0;
let selectedSat = -1;

const slider = document.getElementById('time-slider');
const timeDisplay = document.getElementById('time-display');
const playBtn = document.getElementById('play-btn');
const infoPanel = document.getElementById('info-panel');

slider.max = maxStep;

function updateStep(step) {{
  currentStep = Math.max(0, Math.min(step, maxStep));
  slider.value = currentStep;

  // Update time display
  const ts = DATA.timestamps[currentStep];
  if (ts) {{
    const d = new Date(ts);
    timeDisplay.textContent = d.toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
  }}

  // Update satellite positions and colors
  DATA.satellites.forEach((sat, i) => {{
    const pos = sat.positions[currentStep];
    if (!pos) return;
    const mesh = satMeshes[i];
    mesh.position.set(pos.x, pos.y, pos.z);

    // Color based on status — simple heuristic: every other sat is "computing"
    const computing = (i % 3 !== 0);  // 2/3 sats computing
    let color;
    if (pos.eclipse) {{
      color = computing ? satColors.eclipse_compute : satColors.eclipse;
    }} else {{
      color = computing ? satColors.sunlit_compute : satColors.sunlit_idle;
    }}
    mesh.material.color.copy(color);
  }});

  // Update ISL links
  while (islLineGroup.children.length > 0) {{
    const child = islLineGroup.children[0];
    islLineGroup.remove(child);
    child.geometry.dispose();
  }}
  const links = DATA.islLinks[currentStep] || [];
  links.forEach(link => {{
    const [i, j] = link;
    const p1 = DATA.satellites[i].positions[currentStep];
    const p2 = DATA.satellites[j].positions[currentStep];
    if (!p1 || !p2) return;
    const points = [
      new THREE.Vector3(p1.x, p1.y, p1.z),
      new THREE.Vector3(p2.x, p2.y, p2.z),
    ];
    const geo = new THREE.BufferGeometry().setFromPoints(points);
    islLineGroup.add(new THREE.Line(geo, islMat));
  }});

  // Update info panel if a satellite is selected
  if (selectedSat >= 0) {{
    updateInfoPanel(selectedSat);
  }}
}}

function updateInfoPanel(satIdx) {{
  const sat = DATA.satellites[satIdx];
  const pos = sat.positions[currentStep];
  if (!pos) return;

  infoPanel.style.display = 'block';
  document.getElementById('info-name').textContent = sat.name;
  document.getElementById('info-lat').textContent = pos.lat.toFixed(2) + '\u00b0';
  document.getElementById('info-lon').textContent = pos.lon.toFixed(2) + '\u00b0';
  document.getElementById('info-alt').textContent = pos.alt.toFixed(1) + ' km';
  document.getElementById('info-eclipse').textContent = pos.eclipse ? 'Yes (shadow)' : 'No (sunlit)';

  const computing = (satIdx % 3 !== 0);
  if (pos.eclipse) {{
    document.getElementById('info-status').textContent = computing ? 'Eclipse + Battery Compute' : 'Eclipse + Idle';
  }} else {{
    document.getElementById('info-status').textContent = computing ? 'Sunlit + Computing' : 'Sunlit + Idle';
  }}
}}

// ---- Controls ----
slider.addEventListener('input', () => {{ updateStep(parseInt(slider.value)); }});

playBtn.addEventListener('click', () => {{
  playing = !playing;
  playBtn.textContent = playing ? 'Pause' : 'Play';
}});

// ---- Raycaster for clicking satellites ----
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
raycaster.params.Points = {{ threshold: 0.05 }};

renderer.domElement.addEventListener('click', (event) => {{
  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(satMeshes);
  if (intersects.length > 0) {{
    selectedSat = intersects[0].object.userData.satIndex;
    updateInfoPanel(selectedSat);
  }} else {{
    selectedSat = -1;
    infoPanel.style.display = 'none';
  }}
}});

// ---- Resize ----
window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});

// ---- Animate ----
let accum = 0;
function animate(time) {{
  requestAnimationFrame(animate);

  const dt = time - lastFrameTime;
  lastFrameTime = time;

  if (playing) {{
    accum += dt;
    if (accum > 100) {{  // ~10 steps per second
      accum = 0;
      if (currentStep < maxStep) {{
        updateStep(currentStep + 1);
      }} else {{
        playing = false;
        playBtn.textContent = 'Play';
      }}
    }}
  }}

  // Slow Earth rotation for visual effect
  earthGroup.rotation.y += 0.0002;

  controls.update();
  renderer.render(scene, camera);
}}

// ---- Init ----
updateStep(0);
requestAnimationFrame(animate);
</script>
</body>
</html>"""

