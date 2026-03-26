// === 3D Globe Visualization with Three.js ===
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const EARTH_RADIUS = 1;
const SAT_ALTITUDE = 0.086;
const DEG2RAD = Math.PI / 180;

let scene, camera, renderer, controls;
let earth, satellites = [], orbitLines = [], islLines = [];
let groundStationMarkers = [];
let simTime = 0, playing = true, speedMultiplier = 1, numSats = 12, selectedSat = null;

const GROUND_STATIONS = [
  { name: 'Svalbard', lat: 78.2, lon: 15.6 },
  { name: 'McMurdo', lat: -77.8, lon: 166.7 },
  { name: 'Singapore', lat: 1.3, lon: 103.8 },
  { name: 'Hawaii', lat: 19.8, lon: -155.5 },
  { name: 'Santiago', lat: -33.4, lon: -70.6 }
];

function latLonToVec3(lat, lon, radius) {
  const phi = (90 - lat) * DEG2RAD, theta = (lon + 180) * DEG2RAD;
  return new THREE.Vector3(
    -radius * Math.sin(phi) * Math.cos(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.sin(theta)
  );
}

function createEarth() {
  const geo = new THREE.SphereGeometry(EARTH_RADIUS, 64, 64);
  const mat = new THREE.MeshPhongMaterial({ color: 0x1a3a5c, emissive: 0x0a1628, specular: 0x333333, shininess: 25, transparent: true, opacity: 0.95 });
  earth = new THREE.Mesh(geo, mat);
  scene.add(earth);

  const atmosGeo = new THREE.SphereGeometry(EARTH_RADIUS * 1.02, 64, 64);
  const atmosMat = new THREE.MeshPhongMaterial({ color: 0x4fc3f7, transparent: true, opacity: 0.08, side: THREE.BackSide });
  scene.add(new THREE.Mesh(atmosGeo, atmosMat));

  const gridMat = new THREE.LineBasicMaterial({ color: 0x1e4a6e, transparent: true, opacity: 0.3 });
  for (let lat = -60; lat <= 60; lat += 30) {
    const pts = []; for (let lon = 0; lon <= 360; lon += 3) pts.push(latLonToVec3(lat, lon, EARTH_RADIUS * 1.001));
    scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), gridMat));
  }
  for (let lon = 0; lon < 360; lon += 30) {
    const pts = []; for (let lat = -90; lat <= 90; lat += 3) pts.push(latLonToVec3(lat, lon, EARTH_RADIUS * 1.001));
    scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), gridMat));
  }

  const eqMat = new THREE.LineBasicMaterial({ color: 0x2a6a8e, transparent: true, opacity: 0.5 });
  const eqPts = []; for (let lon = 0; lon <= 360; lon += 2) eqPts.push(latLonToVec3(0, lon, EARTH_RADIUS * 1.002));
  scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(eqPts), eqMat));

  const cMat = new THREE.LineBasicMaterial({ color: 0x3a8a5c, transparent: true, opacity: 0.5 });
  const paths = [
    [[50,-130],[55,-120],[60,-140],[70,-160],[70,-140],[65,-60],[45,-65],[30,-80],[25,-100],[30,-115],[40,-125],[50,-130]],
    [[10,-75],[5,-77],[-5,-80],[-15,-75],[-25,-65],[-35,-58],[-55,-68],[-50,-75],[-20,-40],[-5,-35],[5,-60],[10,-75]],
    [[35,-10],[40,0],[50,5],[55,10],[60,25],[70,30],[55,35],[40,25],[30,35],[10,45],[0,40],[-20,35],[-35,20],[5,0],[35,-10]],
    [[30,35],[40,45],[50,60],[55,70],[60,80],[65,100],[55,130],[45,140],[35,135],[25,100],[10,80],[25,55],[30,35]],
    [[-15,130],[-25,115],[-35,118],[-35,140],[-38,148],[-30,153],[-20,150],[-12,135],[-15,130]]
  ];
  paths.forEach(path => {
    const pts = path.map(([lat, lon]) => latLonToVec3(lat, lon, EARTH_RADIUS * 1.003));
    pts.push(pts[0]);
    scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), cMat));
  });
}

function createGroundStations() {
  groundStationMarkers.forEach(m => scene.remove(m)); groundStationMarkers = [];
  const geo = new THREE.SphereGeometry(0.015, 8, 8);
  const mat = new THREE.MeshBasicMaterial({ color: 0xffa726 });
  GROUND_STATIONS.forEach(gs => {
    const marker = new THREE.Mesh(geo, mat);
    marker.position.copy(latLonToVec3(gs.lat, gs.lon, EARTH_RADIUS * 1.005));
    marker.userData = { type: 'groundStation', ...gs };
    scene.add(marker); groundStationMarkers.push(marker);

    const ringPts = [];
    for (let a = 0; a <= 360; a += 5) {
      const lat2 = gs.lat + 15 * Math.cos(a * DEG2RAD);
      const lon2 = gs.lon + 15 * Math.sin(a * DEG2RAD) / Math.cos(gs.lat * DEG2RAD);
      ringPts.push(latLonToVec3(lat2, lon2, EARTH_RADIUS * 1.003));
    }
    scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(ringPts),
      new THREE.LineBasicMaterial({ color: 0xffa726, transparent: true, opacity: 0.2 })));
  });
}

function createSatellites(count) {
  satellites.forEach(s => { scene.remove(s.mesh); scene.remove(s.trail); });
  orbitLines.forEach(l => scene.remove(l));
  satellites = []; orbitLines = []; numSats = count;

  const geo = new THREE.SphereGeometry(0.02, 8, 8);
  const numPlanes = Math.max(1, Math.ceil(count / 4));
  const satsPerPlane = Math.ceil(count / numPlanes);

  for (let i = 0; i < count; i++) {
    const planeIdx = Math.floor(i / satsPerPlane);
    const idxInPlane = i % satsPerPlane;
    const raan = (planeIdx / numPlanes) * 360;
    const phase0 = (idxInPlane / satsPerPlane) * 360;
    const inclination = 53;
    const color = new THREE.Color().setHSL(0.55 + planeIdx * 0.15, 0.8, 0.6);
    const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color }));
    mesh.userData = { type: 'satellite', id: i, name: `SAT-${String(i + 1).padStart(3, '0')}`, raan, phase0, inclination };
    scene.add(mesh);

    const trailGeo = new THREE.BufferGeometry();
    trailGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(300 * 3), 3));
    trailGeo.setDrawRange(0, 0);
    const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.3 }));
    scene.add(trail);
    satellites.push({ mesh, trail, raan, phase0, inclination, trailPoints: [], color });

    const orbitPts = [];
    for (let a = 0; a <= 360; a += 2) {
      const r = EARTH_RADIUS + SAT_ALTITUDE, aRad = a * DEG2RAD;
      const raanRad = raan * DEG2RAD, inclRad = inclination * DEG2RAD;
      const xOrb = r * Math.cos(aRad), yOrb = r * Math.sin(aRad);
      orbitPts.push(new THREE.Vector3(
        Math.cos(raanRad) * xOrb - Math.sin(raanRad) * Math.cos(inclRad) * yOrb,
        Math.sin(inclRad) * yOrb,
        Math.sin(raanRad) * xOrb + Math.cos(raanRad) * Math.cos(inclRad) * yOrb
      ));
    }
    const orbitLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(orbitPts),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.12 }));
    scene.add(orbitLine); orbitLines.push(orbitLine);
  }
}

function updateSatPositions(time) {
  const period = 5700, r = EARTH_RADIUS + SAT_ALTITUDE;
  islLines.forEach(l => scene.remove(l)); islLines = [];

  satellites.forEach(sat => {
    const n = 2 * Math.PI / period;
    const ta = sat.phase0 * DEG2RAD + n * time;
    const raanRad = sat.raan * DEG2RAD, inclRad = sat.inclination * DEG2RAD;
    const xOrb = r * Math.cos(ta), yOrb = r * Math.sin(ta);
    const x = Math.cos(raanRad) * xOrb - Math.sin(raanRad) * Math.cos(inclRad) * yOrb;
    const y = Math.sin(inclRad) * yOrb;
    const z = Math.sin(raanRad) * xOrb + Math.cos(raanRad) * Math.cos(inclRad) * yOrb;
    sat.mesh.position.set(x, y, z);

    sat.trailPoints.push(new THREE.Vector3(x, y, z));
    if (sat.trailPoints.length > 100) sat.trailPoints.shift();
    const arr = sat.trail.geometry.attributes.position.array;
    sat.trailPoints.forEach((p, j) => { arr[j * 3] = p.x; arr[j * 3 + 1] = p.y; arr[j * 3 + 2] = p.z; });
    sat.trail.geometry.attributes.position.needsUpdate = true;
    sat.trail.geometry.setDrawRange(0, sat.trailPoints.length);
  });

  const islMat = new THREE.LineBasicMaterial({ color: 0x4fc3f7, transparent: true, opacity: 0.15 });
  for (let i = 0; i < satellites.length; i++) {
    for (let j = i + 1; j < satellites.length; j++) {
      if (satellites[i].mesh.position.distanceTo(satellites[j].mesh.position) < 0.5) {
        const geo = new THREE.BufferGeometry().setFromPoints([satellites[i].mesh.position.clone(), satellites[j].mesh.position.clone()]);
        const line = new THREE.Line(geo, islMat);
        scene.add(line); islLines.push(line);
      }
    }
  }
}

function updateInfoPanel() {
  const el = document.getElementById('sat-info');
  if (!el) return;
  if (selectedSat !== null && satellites[selectedSat]) {
    const pos = satellites[selectedSat].mesh.position;
    const r = pos.length();
    const lat = (Math.asin(pos.y / r) / DEG2RAD).toFixed(1);
    const lon = (Math.atan2(pos.z, pos.x) / DEG2RAD).toFixed(1);
    el.innerHTML = `<div style="font-weight:700;color:#4fc3f7;margin-bottom:0.5rem">SAT-${String(selectedSat + 1).padStart(3, '0')}</div>
      <div class="text-sm"><span class="text-muted">Lat:</span> ${lat}</div>
      <div class="text-sm"><span class="text-muted">Lon:</span> ${lon}</div>
      <div class="text-sm"><span class="text-muted">Alt:</span> 550 km</div>
      <div class="text-sm"><span class="text-muted">RAAN:</span> ${satellites[selectedSat].raan.toFixed(0)}</div>
      <div class="text-sm"><span class="text-muted">Incl:</span> ${satellites[selectedSat].inclination}</div>`;
  } else {
    el.innerHTML = `<div class="text-muted text-sm">Click a satellite for details</div>
      <div class="text-sm mt-1"><span class="text-muted">Satellites:</span> ${numSats}</div>
      <div class="text-sm"><span class="text-muted">Altitude:</span> 550 km</div>
      <div class="text-sm"><span class="text-muted">Inclination:</span> 53 deg</div>`;
  }
}

function onClick(event) {
  const container = document.getElementById('globe-container');
  const rect = container.getBoundingClientRect();
  const mouse = new THREE.Vector2(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1
  );
  const raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(satellites.map(s => s.mesh));
  if (intersects.length > 0) {
    selectedSat = intersects[0].object.userData.id;
    satellites.forEach((s, i) => s.mesh.scale.setScalar(i === selectedSat ? 2 : 1));
  } else {
    selectedSat = null;
    satellites.forEach(s => s.mesh.scale.setScalar(1));
  }
  updateInfoPanel();
}

function animate() {
  requestAnimationFrame(animate);
  if (playing) simTime += 0.016 * speedMultiplier * 60;
  updateSatPositions(simTime);
  if (earth) earth.rotation.y = simTime * 0.0001;
  controls.update();
  renderer.render(scene, camera);
  const timeEl = document.getElementById('time-display');
  if (timeEl) timeEl.textContent = `T+${(simTime / 3600).toFixed(1)}h`;
  updateInfoPanel();
}

export function init() {
  const container = document.getElementById('globe-container');
  if (!container) return;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0e17);
  camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.01, 100);
  camera.position.set(0, 1, 3);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.05;
  controls.minDistance = 1.5; controls.maxDistance = 8;

  scene.add(new THREE.AmbientLight(0x404060, 0.5));
  const sun = new THREE.DirectionalLight(0xffffff, 1.2);
  sun.position.set(5, 3, 5); scene.add(sun);

  const starGeo = new THREE.BufferGeometry();
  const starPos = new Float32Array(3000 * 3);
  for (let i = 0; i < 3000; i++) {
    const r = 20 + Math.random() * 30;
    const theta = Math.random() * Math.PI * 2, phi = Math.acos(2 * Math.random() - 1);
    starPos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
    starPos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
    starPos[i * 3 + 2] = r * Math.cos(phi);
  }
  starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
  scene.add(new THREE.Points(starGeo, new THREE.PointsMaterial({ color: 0xffffff, size: 0.05 })));

  createEarth(); createGroundStations(); createSatellites(numSats);

  window.addEventListener('resize', () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });
  container.addEventListener('click', onClick);

  const playBtn = document.getElementById('play-btn');
  if (playBtn) playBtn.addEventListener('click', () => { playing = !playing; playBtn.textContent = playing ? 'Pause' : 'Play'; });

  const speedSlider = document.getElementById('speed-slider');
  if (speedSlider) speedSlider.addEventListener('input', e => {
    speedMultiplier = parseFloat(e.target.value);
    const l = document.getElementById('speed-label'); if (l) l.textContent = speedMultiplier.toFixed(1) + 'x';
  });

  const satSlider = document.getElementById('globe-sat-count');
  if (satSlider) satSlider.addEventListener('input', e => {
    const count = parseInt(e.target.value);
    const l = document.getElementById('globe-sat-label'); if (l) l.textContent = count;
    createSatellites(count);
  });

  const resetBtn = document.getElementById('reset-btn');
  if (resetBtn) resetBtn.addEventListener('click', () => { simTime = 0; satellites.forEach(s => { s.trailPoints = []; }); });

  animate();
}
