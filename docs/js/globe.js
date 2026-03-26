// === 3D Globe Visualization with Three.js ===
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const EARTH_RADIUS = 1;
const SAT_ALTITUDE = 0.086; // ~550km / 6371km
const DEG2RAD = Math.PI / 180;

let scene, camera, renderer, controls;
let earth, satellites = [], orbitLines = [], islLines = [];
let groundStationMarkers = [];
let simTime = 0;
let playing = true;
let speedMultiplier = 1;
let numSats = 12;
let selectedSat = null;

const GROUND_STATIONS = [
  { name: 'Svalbard', lat: 78.2, lon: 15.6 },
  { name: 'McMurdo', lat: -77.8, lon: 166.7 },
  { name: 'Singapore', lat: 1.3, lon: 103.8 },
  { name: 'Hawaii', lat: 19.8, lon: -155.5 },
  { name: 'Santiago', lat: -33.4, lon: -70.6 }
];

function latLonToVec3(lat, lon, radius) {
  const phi = (90 - lat) * DEG2RAD;
  const theta = (lon + 180) * DEG2RAD;
  return new THREE.Vector3(
    -radius * Math.sin(phi) * Math.cos(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.sin(theta)
  );
}

function createEarth() {
  // Earth sphere
  const geometry = new THREE.SphereGeometry(EARTH_RADIUS, 64, 64);
  const material = new THREE.MeshPhongMaterial({
    color: 0x1a3a5c,
    emissive: 0x0a1628,
    specular: 0x333333,
    shininess: 25,
    transparent: true,
    opacity: 0.95
  });
  earth = new THREE.Mesh(geometry, material);
  scene.add(earth);

  // Atmosphere glow
  const atmosGeom = new THREE.SphereGeometry(EARTH_RADIUS * 1.02, 64, 64);
  const atmosMat = new THREE.MeshPhongMaterial({
    color: 0x4fc3f7,
    transparent: true,
    opacity: 0.08,
    side: THREE.BackSide
  });
  const atmosphere = new THREE.Mesh(atmosGeom, atmosMat);
  scene.add(atmosphere);

  // Lat/Lon grid
  const gridMat = new THREE.LineBasicMaterial({ color: 0x1e4a6e, transparent: true, opacity: 0.3 });

  // Latitude lines
  for (let lat = -60; lat <= 60; lat += 30) {
    const points = [];
    for (let lon = 0; lon <= 360; lon += 3) {
      points.push(latLonToVec3(lat, lon, EARTH_RADIUS * 1.001));
    }
    const geom = new THREE.BufferGeometry().setFromPoints(points);
    scene.add(new THREE.Line(geom, gridMat));
  }

  // Longitude lines
  for (let lon = 0; lon < 360; lon += 30) {
    const points = [];
    for (let lat = -90; lat <= 90; lat += 3) {
      points.push(latLonToVec3(lat, lon, EARTH_RADIUS * 1.001));
    }
    const geom = new THREE.BufferGeometry().setFromPoints(points);
    scene.add(new THREE.Line(geom, gridMat));
  }

  // Equator (brighter)
  const eqMat = new THREE.LineBasicMaterial({ color: 0x2a6a8e, transparent: true, opacity: 0.5 });
  const eqPoints = [];
  for (let lon = 0; lon <= 360; lon += 2) {
    eqPoints.push(latLonToVec3(0, lon, EARTH_RADIUS * 1.002));
  }
  scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(eqPoints), eqMat));

  // Simple continent outlines (very approximate major landmasses)
  const continentMat = new THREE.LineBasicMaterial({ color: 0x3a8a5c, transparent: true, opacity: 0.5 });
  const continentPaths = [
    // North America (rough outline)
    [[50,-130],[55,-120],[60,-140],[70,-160],[70,-140],[65,-60],[45,-65],[30,-80],[25,-100],[30,-115],[35,-120],[40,-125],[48,-125],[50,-130]],
    // South America
    [[10,-75],[5,-77],[-5,-80],[-15,-75],[-25,-65],[-35,-58],[-50,-70],[-55,-68],[-50,-75],[-40,-73],[-20,-40],[-5,-35],[5,-60],[10,-75]],
    // Europe/Africa
    [[35,-10],[40,0],[45,5],[50,5],[55,10],[60,25],[70,30],[55,35],[45,30],[40,25],[35,35],[30,35],[10,45],[0,40],[-10,40],[-20,35],[-35,20],[-35,17],[5,0],[10,-15],[35,-10]],
    // Asia (very rough)
    [[30,35],[40,45],[45,50],[50,60],[55,70],[60,80],[65,100],[55,130],[45,140],[35,135],[30,120],[25,100],[20,90],[10,80],[25,55],[30,35]],
    // Australia
    [[-15,130],[-20,115],[-25,115],[-35,118],[-35,140],[-38,148],[-30,153],[-20,150],[-12,135],[-15,130]]
  ];

  continentPaths.forEach(path => {
    const points = path.map(([lat, lon]) => latLonToVec3(lat, lon, EARTH_RADIUS * 1.003));
    points.push(points[0]); // close loop
    scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), continentMat));
  });
}

function createGroundStations() {
  groundStationMarkers.forEach(m => scene.remove(m));
  groundStationMarkers = [];

  const markerGeom = new THREE.SphereGeometry(0.015, 8, 8);
  const markerMat = new THREE.MeshBasicMaterial({ color: 0xffa726 });

  GROUND_STATIONS.forEach(gs => {
    const marker = new THREE.Mesh(markerGeom, markerMat);
    const pos = latLonToVec3(gs.lat, gs.lon, EARTH_RADIUS * 1.005);
    marker.position.copy(pos);
    marker.userData = { type: 'groundStation', ...gs };
    scene.add(marker);
    groundStationMarkers.push(marker);

    // Range ring
    const ringPoints = [];
    const rangeDeg = 15;
    for (let angle = 0; angle <= 360; angle += 5) {
      const lat2 = gs.lat + rangeDeg * Math.cos(angle * DEG2RAD);
      const lon2 = gs.lon + rangeDeg * Math.sin(angle * DEG2RAD) / Math.cos(gs.lat * DEG2RAD);
      ringPoints.push(latLonToVec3(lat2, lon2, EARTH_RADIUS * 1.003));
    }
    const ringGeom = new THREE.BufferGeometry().setFromPoints(ringPoints);
    const ringMat = new THREE.LineBasicMaterial({ color: 0xffa726, transparent: true, opacity: 0.2 });
    scene.add(new THREE.Line(ringGeom, ringMat));
  });
}

function createSatellites(count) {
  // Remove old
  satellites.forEach(s => { scene.remove(s.mesh); scene.remove(s.trail); });
  orbitLines.forEach(l => scene.remove(l));
  satellites = [];
  orbitLines = [];
  numSats = count;

  const satGeom = new THREE.SphereGeometry(0.02, 8, 8);
  const numPlanes = Math.max(1, Math.ceil(count / 4));
  const satsPerPlane = Math.ceil(count / numPlanes);

  for (let i = 0; i < count; i++) {
    const planeIdx = Math.floor(i / satsPerPlane);
    const idxInPlane = i % satsPerPlane;
    const raan = (planeIdx / numPlanes) * 360;
    const phase0 = (idxInPlane / satsPerPlane) * 360;
    const inclination = 53;

    const color = new THREE.Color().setHSL(0.55 + planeIdx * 0.15, 0.8, 0.6);
    const mat = new THREE.MeshBasicMaterial({ color });
    const mesh = new THREE.Mesh(satGeom, mat);
    mesh.userData = { type: 'satellite', id: i, name: `SAT-${String(i + 1).padStart(3, '0')}`, raan, phase0, inclination };
    scene.add(mesh);

    // Trail
    const trailGeom = new THREE.BufferGeometry();
    const trailPositions = new Float32Array(300 * 3);
    trailGeom.setAttribute('position', new THREE.BufferAttribute(trailPositions, 3));
    trailGeom.setDrawRange(0, 0);
    const trailMat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.3 });
    const trail = new THREE.Line(trailGeom, trailMat);
    scene.add(trail);

    satellites.push({ mesh, trail, raan, phase0, inclination, trailPoints: [], color });

    // Orbit line
    const orbitPoints = [];
    for (let angle = 0; angle <= 360; angle += 2) {
      const r = EARTH_RADIUS + SAT_ALTITUDE;
      const aRad = angle * DEG2RAD;
      const raanRad = raan * DEG2RAD;
      const inclRad = inclination * DEG2RAD;

      const xOrb = r * Math.cos(aRad);
      const yOrb = r * Math.sin(aRad);

      const x = Math.cos(raanRad) * xOrb - Math.sin(raanRad) * Math.cos(inclRad) * yOrb;
      const y = Math.sin(inclRad) * yOrb;
      const z = Math.sin(raanRad) * xOrb + Math.cos(raanRad) * Math.cos(inclRad) * yOrb;

      orbitPoints.push(new THREE.Vector3(x, y, z));
    }
    const orbitGeom = new THREE.BufferGeometry().setFromPoints(orbitPoints);
    const orbitMat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.12 });
    const orbitLine = new THREE.Line(orbitGeom, orbitMat);
    scene.add(orbitLine);
    orbitLines.push(orbitLine);
  }
}

function updateSatPositions(time) {
  const period = 5700; // seconds, ~95 min orbit period in sim
  const r = EARTH_RADIUS + SAT_ALTITUDE;

  // Remove old ISL lines
  islLines.forEach(l => scene.remove(l));
  islLines = [];

  satellites.forEach((sat, i) => {
    const n = 2 * Math.PI / period;
    const trueAnomaly = sat.phase0 * DEG2RAD + n * time;
    const raanRad = sat.raan * DEG2RAD;
    const inclRad = sat.inclination * DEG2RAD;

    const xOrb = r * Math.cos(trueAnomaly);
    const yOrb = r * Math.sin(trueAnomaly);

    const x = Math.cos(raanRad) * xOrb - Math.sin(raanRad) * Math.cos(inclRad) * yOrb;
    const y = Math.sin(inclRad) * yOrb;
    const z = Math.sin(raanRad) * xOrb + Math.cos(raanRad) * Math.cos(inclRad) * yOrb;

    sat.mesh.position.set(x, y, z);

    // Update trail
    sat.trailPoints.push(new THREE.Vector3(x, y, z));
    if (sat.trailPoints.length > 100) sat.trailPoints.shift();

    const posArr = sat.trail.geometry.attributes.position.array;
    sat.trailPoints.forEach((p, j) => {
      posArr[j * 3] = p.x;
      posArr[j * 3 + 1] = p.y;
      posArr[j * 3 + 2] = p.z;
    });
    sat.trail.geometry.attributes.position.needsUpdate = true;
    sat.trail.geometry.setDrawRange(0, sat.trailPoints.length);
  });

  // ISL links: connect nearby satellites
  const islMat = new THREE.LineBasicMaterial({ color: 0x4fc3f7, transparent: true, opacity: 0.15 });
  const maxDist = 0.5;

  for (let i = 0; i < satellites.length; i++) {
    for (let j = i + 1; j < satellites.length; j++) {
      const dist = satellites[i].mesh.position.distanceTo(satellites[j].mesh.position);
      if (dist < maxDist) {
        const points = [satellites[i].mesh.position.clone(), satellites[j].mesh.position.clone()];
        const geom = new THREE.BufferGeometry().setFromPoints(points);
        const line = new THREE.Line(geom, islMat);
        scene.add(line);
        islLines.push(line);
      }
    }
  }
}

function updateInfoPanel() {
  const infoEl = document.getElementById('sat-info');
  if (!infoEl) return;

  if (selectedSat !== null && satellites[selectedSat]) {
    const sat = satellites[selectedSat];
    const pos = sat.mesh.position;
    // Approximate lat/lon from 3D position
    const r = pos.length();
    const lat = Math.asin(pos.y / r) / DEG2RAD;
    const lon = Math.atan2(pos.z, pos.x) / DEG2RAD;

    infoEl.innerHTML = `
      <div style="font-weight:700;color:#4fc3f7;margin-bottom:0.5rem">SAT-${String(selectedSat + 1).padStart(3, '0')}</div>
      <div class="text-sm"><span class="text-muted">Lat:</span> ${lat.toFixed(1)}</div>
      <div class="text-sm"><span class="text-muted">Lon:</span> ${lon.toFixed(1)}</div>
      <div class="text-sm"><span class="text-muted">Alt:</span> 550 km</div>
      <div class="text-sm"><span class="text-muted">RAAN:</span> ${sat.raan.toFixed(0)}</div>
      <div class="text-sm"><span class="text-muted">Incl:</span> ${sat.inclination}</div>
    `;
  } else {
    infoEl.innerHTML = `
      <div class="text-muted text-sm">Click a satellite for details</div>
      <div class="text-sm mt-1"><span class="text-muted">Satellites:</span> ${numSats}</div>
      <div class="text-sm"><span class="text-muted">Altitude:</span> 550 km</div>
      <div class="text-sm"><span class="text-muted">Inclination:</span> 53 deg</div>
    `;
  }
}

function onResize() {
  const container = document.getElementById('globe-container');
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
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

  const satMeshes = satellites.map(s => s.mesh);
  const intersects = raycaster.intersectObjects(satMeshes);

  if (intersects.length > 0) {
    const hitMesh = intersects[0].object;
    selectedSat = hitMesh.userData.id;

    // Highlight selected
    satellites.forEach((s, i) => {
      s.mesh.scale.setScalar(i === selectedSat ? 2 : 1);
    });
  } else {
    selectedSat = null;
    satellites.forEach(s => s.mesh.scale.setScalar(1));
  }

  updateInfoPanel();
}

function animate() {
  requestAnimationFrame(animate);

  if (playing) {
    simTime += 0.016 * speedMultiplier * 60; // ~60 sim seconds per real second at 1x
  }

  updateSatPositions(simTime);

  // Slow Earth rotation
  if (earth) earth.rotation.y = simTime * 0.0001;

  controls.update();
  renderer.render(scene, camera);

  // Update time display
  const timeEl = document.getElementById('time-display');
  if (timeEl) {
    const hrs = (simTime / 3600).toFixed(1);
    timeEl.textContent = `T+${hrs}h`;
  }

  updateInfoPanel();
}

export function init() {
  const container = document.getElementById('globe-container');
  if (!container) return;

  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0e17);

  // Camera
  camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.01, 100);
  camera.position.set(0, 1, 3);

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  // Controls
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.minDistance = 1.5;
  controls.maxDistance = 8;

  // Lighting
  const ambientLight = new THREE.AmbientLight(0x404060, 0.5);
  scene.add(ambientLight);

  const sunLight = new THREE.DirectionalLight(0xffffff, 1.2);
  sunLight.position.set(5, 3, 5);
  scene.add(sunLight);

  // Starfield
  const starGeom = new THREE.BufferGeometry();
  const starPositions = new Float32Array(3000 * 3);
  for (let i = 0; i < 3000; i++) {
    const r = 20 + Math.random() * 30;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    starPositions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
    starPositions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
    starPositions[i * 3 + 2] = r * Math.cos(phi);
  }
  starGeom.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
  const starMat = new THREE.PointsMaterial({ color: 0xffffff, size: 0.05 });
  scene.add(new THREE.Points(starGeom, starMat));

  // Build scene
  createEarth();
  createGroundStations();
  createSatellites(numSats);

  // Event listeners
  window.addEventListener('resize', onResize);
  container.addEventListener('click', onClick);

  // UI controls
  const playBtn = document.getElementById('play-btn');
  if (playBtn) {
    playBtn.addEventListener('click', () => {
      playing = !playing;
      playBtn.textContent = playing ? 'Pause' : 'Play';
    });
  }

  const speedSlider = document.getElementById('speed-slider');
  if (speedSlider) {
    speedSlider.addEventListener('input', (e) => {
      speedMultiplier = parseFloat(e.target.value);
      const label = document.getElementById('speed-label');
      if (label) label.textContent = speedMultiplier.toFixed(1) + 'x';
    });
  }

  const satSlider = document.getElementById('globe-sat-count');
  if (satSlider) {
    satSlider.addEventListener('input', (e) => {
      const count = parseInt(e.target.value);
      const label = document.getElementById('globe-sat-label');
      if (label) label.textContent = count;
      createSatellites(count);
    });
  }

  const resetBtn = document.getElementById('reset-btn');
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      simTime = 0;
      satellites.forEach(s => { s.trailPoints = []; });
    });
  }

  animate();
}
