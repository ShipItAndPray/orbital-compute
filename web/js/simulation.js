// === Orbital Compute Simulator — Core Simulation Engine ===

const DEG2RAD = Math.PI / 180;
const EARTH_RADIUS_KM = 6371;
const MU_EARTH = 398600.4418; // km^3/s^2

export function orbitalPeriod(altitudeKm) {
  const a = EARTH_RADIUS_KM + altitudeKm;
  return 2 * Math.PI * Math.sqrt(Math.pow(a, 3) / MU_EARTH);
}

export function eclipseFraction(altitudeKm) {
  const a = EARTH_RADIUS_KM + altitudeKm;
  const rho = Math.asin(EARTH_RADIUS_KM / a);
  // beta=0 gives worst-case (maximum) eclipse
  const eclipseAngle = 2 * rho;
  return eclipseAngle / (2 * Math.PI);
}

export function satPosition(altitudeKm, inclDeg, raanDeg, timeSec, phase0Deg = 0) {
  const a = EARTH_RADIUS_KM + altitudeKm;
  const period = orbitalPeriod(altitudeKm);
  const n = 2 * Math.PI / period;
  const trueAnomaly = (phase0Deg * DEG2RAD) + n * timeSec;
  const inclRad = inclDeg * DEG2RAD;
  const raanRad = raanDeg * DEG2RAD;
  const xOrb = a * Math.cos(trueAnomaly);
  const yOrb = a * Math.sin(trueAnomaly);
  const cosR = Math.cos(raanRad), sinR = Math.sin(raanRad);
  const cosI = Math.cos(inclRad), sinI = Math.sin(inclRad);
  const x = cosR * xOrb - sinR * cosI * yOrb;
  const y = sinR * xOrb + cosR * cosI * yOrb;
  const z = sinI * yOrb;
  const earthRotRate = 2 * Math.PI / 86400;
  const r = Math.sqrt(x * x + y * y + z * z);
  const lat = Math.asin(z / r) / DEG2RAD;
  const lon = ((Math.atan2(y, x) / DEG2RAD) - (timeSec * earthRotRate / DEG2RAD) + 360) % 360;
  const lonNorm = lon > 180 ? lon - 360 : lon;
  return { lat, lon: lonNorm, alt: altitudeKm, x, y, z };
}

export function thermalModel(solarWatts, computeWatts, inEclipse) {
  const baseTemp = inEclipse ? -40 : 20;
  const heatGen = computeWatts * 0.7;
  const tempRise = heatGen / (0.85 * 5);
  return Math.round(baseTemp + tempRise);
}

const GROUND_STATIONS = [
  { name: 'Svalbard', lat: 78.2, lon: 15.6 },
  { name: 'McMurdo', lat: -77.8, lon: 166.7 },
  { name: 'Singapore', lat: 1.3, lon: 103.8 },
  { name: 'Hawaii', lat: 19.8, lon: -155.5 },
  { name: 'Santiago', lat: -33.4, lon: -70.6 }
];

export function getGroundStations() { return GROUND_STATIONS; }

export function inContact(satLat, satLon, gsLat, gsLon, altitudeKm) {
  const dLat = (satLat - gsLat) * DEG2RAD;
  const dLon = (satLon - gsLon) * DEG2RAD;
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(satLat * DEG2RAD) * Math.cos(gsLat * DEG2RAD) * Math.sin(dLon / 2) ** 2;
  const dist = 2 * Math.asin(Math.sqrt(a)) * EARTH_RADIUS_KM;
  const maxRange = Math.sqrt(2 * EARTH_RADIUS_KM * altitudeKm + altitudeKm * altitudeKm);
  return dist < maxRange;
}

export function scheduleJobs(numJobs, satellites, simHours) {
  const jobs = [];
  for (let i = 0; i < numJobs; i++) {
    jobs.push({
      id: i, submitTime: Math.random() * simHours * 3600,
      durationSec: 120 + Math.random() * 600, gpuCount: Math.ceil(Math.random() * 4),
      status: 'pending', assignedSat: null, startTime: null, endTime: null,
      priority: Math.random()
    });
  }
  jobs.sort((a, b) => a.submitTime - b.submitTime);
  const satBusyUntil = new Array(satellites.length).fill(0);
  for (const job of jobs) {
    let bestSat = -1, bestTime = Infinity;
    for (let s = 0; s < satellites.length; s++) {
      const earliest = Math.max(job.submitTime, satBusyUntil[s]);
      if (earliest < bestTime) { bestTime = earliest; bestSat = s; }
    }
    if (bestSat >= 0) {
      job.assignedSat = bestSat; job.startTime = bestTime;
      job.endTime = bestTime + job.durationSec; job.status = 'completed';
      satBusyUntil[bestSat] = job.endTime;
    }
  }
  return jobs;
}

export function runSimulation(params = {}) {
  const {
    numSatellites = 12, simHours = 12, numJobs = 80,
    solarWatts = 2000, batteryWh = 5000,
    altitudeKm = 550, inclination = 53, computeWatts = 800
  } = params;

  const period = orbitalPeriod(altitudeKm);
  const eclFrac = eclipseFraction(altitudeKm);
  const totalSteps = Math.ceil((simHours * 3600) / (period / 60));
  const dt = (simHours * 3600) / totalSteps;

  const satellites = [];
  for (let i = 0; i < numSatellites; i++) {
    const planeIndex = Math.floor(i / Math.ceil(numSatellites / 3));
    const phaseInPlane = (i % Math.ceil(numSatellites / 3)) * (360 / Math.ceil(numSatellites / 3));
    satellites.push({
      id: i, name: `SAT-${String(i + 1).padStart(3, '0')}`,
      altitude: altitudeKm, inclination, raan: planeIndex * 120,
      phase0: phaseInPlane, gpuCount: 8, states: []
    });
  }

  for (let t = 0; t < totalSteps; t++) {
    const timeSec = t * dt;
    for (const sat of satellites) {
      const pos = satPosition(sat.altitude, sat.inclination, sat.raan, timeSec, sat.phase0);
      const orbitFrac = (timeSec % period) / period;
      const isEclipse = orbitFrac >= (0.5 - eclFrac / 2) && orbitFrac <= (0.5 + eclFrac / 2);
      const solarIn = isEclipse ? 0 : solarWatts;
      const prevBattery = t > 0 ? sat.states[t - 1].battery : batteryWh;
      const netPower = solarIn - computeWatts;
      const battery = Math.max(0, Math.min(batteryWh, prevBattery + (netPower * dt / 3600)));
      const canCompute = battery > batteryWh * 0.1;
      const temp = thermalModel(solarWatts, canCompute ? computeWatts : 0, isEclipse);
      const contacts = GROUND_STATIONS.filter(gs => inContact(pos.lat, pos.lon, gs.lat, gs.lon, sat.altitude));
      sat.states.push({
        step: t, timeSec, timeHours: timeSec / 3600,
        lat: pos.lat, lon: pos.lon, x: pos.x, y: pos.y, z: pos.z,
        inEclipse: isEclipse, solarIn, battery,
        batteryPct: (battery / batteryWh) * 100,
        canCompute, computing: canCompute, tempC: temp,
        groundContacts: contacts.map(gs => gs.name)
      });
    }
  }

  const jobs = scheduleJobs(numJobs, satellites, simHours);
  const completedJobs = jobs.filter(j => j.status === 'completed');
  const avgLatency = completedJobs.length > 0
    ? completedJobs.reduce((sum, j) => sum + (j.startTime - j.submitTime), 0) / completedJobs.length : 0;
  const totalComputeSteps = satellites.reduce((s, sat) => s + sat.states.filter(st => st.computing).length, 0);
  const totalStepsAll = satellites.reduce((s, sat) => s + sat.states.length, 0);
  const totalEclipseSteps = satellites.reduce((s, sat) => s + sat.states.filter(st => st.inEclipse).length, 0);

  const stats = {
    numSatellites, simHours, numJobs,
    completedJobs: completedJobs.length,
    failedJobs: numJobs - completedJobs.length,
    fleetUtilization: ((totalComputeSteps / totalStepsAll) * 100).toFixed(1),
    avgLatencySec: avgLatency.toFixed(1),
    avgBatteryPct: (satellites.reduce((s, sat) =>
      s + sat.states.reduce((a, st) => a + st.batteryPct, 0) / sat.states.length, 0) / numSatellites).toFixed(1),
    eclipsePercent: ((totalEclipseSteps / totalStepsAll) * 100).toFixed(1),
    orbitalPeriodMin: (period / 60).toFixed(1),
    eclipseFraction: (eclFrac * 100).toFixed(1),
    totalTimesteps: totalSteps,
    altitudeKm, inclination
  };

  const satSummaries = satellites.map(sat => {
    const cs = sat.states.filter(s => s.computing).length;
    const es = sat.states.filter(s => s.inEclipse).length;
    return {
      id: sat.id, name: sat.name,
      utilization: ((cs / sat.states.length) * 100).toFixed(1),
      eclipsePercent: ((es / sat.states.length) * 100).toFixed(1),
      avgBattery: (sat.states.reduce((s, st) => s + st.batteryPct, 0) / sat.states.length).toFixed(1),
      minBattery: Math.min(...sat.states.map(s => s.batteryPct)).toFixed(1),
      avgTemp: (sat.states.reduce((s, st) => s + st.tempC, 0) / sat.states.length).toFixed(1),
      jobsCompleted: jobs.filter(j => j.assignedSat === sat.id).length,
      contactWindows: sat.states.filter(s => s.groundContacts.length > 0).length
    };
  });

  return {
    params: { numSatellites, simHours, numJobs, solarWatts, batteryWh, altitudeKm, inclination, computeWatts },
    stats, satSummaries,
    satellites: satellites.map(sat => ({
      id: sat.id, name: sat.name, altitude: sat.altitude, inclination: sat.inclination,
      raan: sat.raan, phase0: sat.phase0,
      states: sat.states.filter((_, i) => i % 3 === 0)
    })),
    jobs: jobs.slice(0, 200),
    groundStations: GROUND_STATIONS
  };
}
