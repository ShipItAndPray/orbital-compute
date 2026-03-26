// === Orbital Compute Simulator — Core Simulation Engine ===

const DEG2RAD = Math.PI / 180;
const EARTH_RADIUS_KM = 6371;
const MU_EARTH = 398600.4418; // km^3/s^2
const SOLAR_CONSTANT = 1361; // W/m^2

/**
 * Compute orbital period from altitude (circular orbit).
 * @param {number} altitudeKm - Altitude in km
 * @returns {number} Period in seconds
 */
export function orbitalPeriod(altitudeKm) {
  const a = EARTH_RADIUS_KM + altitudeKm;
  return 2 * Math.PI * Math.sqrt(Math.pow(a, 3) / MU_EARTH);
}

/**
 * Compute eclipse fraction for a circular orbit.
 * Simplified: depends on altitude and beta angle (sun angle to orbit plane).
 * @param {number} altitudeKm
 * @param {number} betaDeg - Sun angle to orbit plane in degrees
 * @returns {number} Fraction of orbit in eclipse [0,1]
 */
export function eclipseFraction(altitudeKm, betaDeg = 0) {
  const a = EARTH_RADIUS_KM + altitudeKm;
  const rho = Math.asin(EARTH_RADIUS_KM / a);
  const betaRad = Math.abs(betaDeg) * DEG2RAD;
  if (betaRad > (Math.PI / 2 - rho)) return 0; // No eclipse
  const eclipseAngle = 2 * Math.acos(Math.cos(rho) / Math.cos(betaRad));
  return eclipseAngle / (2 * Math.PI);
}

/**
 * Compute satellite position on a circular orbit at a given time.
 * @param {number} altitudeKm
 * @param {number} inclDeg - Inclination in degrees
 * @param {number} raanDeg - Right ascension of ascending node in degrees
 * @param {number} timeSec - Time since epoch in seconds
 * @param {number} phase0Deg - Initial phase in degrees
 * @returns {{lat: number, lon: number, alt: number, x: number, y: number, z: number}}
 */
export function satPosition(altitudeKm, inclDeg, raanDeg, timeSec, phase0Deg = 0) {
  const a = EARTH_RADIUS_KM + altitudeKm;
  const period = orbitalPeriod(altitudeKm);
  const n = 2 * Math.PI / period;
  const trueAnomaly = (phase0Deg * DEG2RAD) + n * timeSec;

  const inclRad = inclDeg * DEG2RAD;
  const raanRad = raanDeg * DEG2RAD;

  // Position in orbital plane
  const xOrb = a * Math.cos(trueAnomaly);
  const yOrb = a * Math.sin(trueAnomaly);

  // Rotate to ECI
  const cosR = Math.cos(raanRad), sinR = Math.sin(raanRad);
  const cosI = Math.cos(inclRad), sinI = Math.sin(inclRad);

  const x = cosR * xOrb - sinR * cosI * yOrb;
  const y = sinR * xOrb + cosR * cosI * yOrb;
  const z = sinI * yOrb;

  // To lat/lon (simplified, ignoring Earth rotation for visualization)
  const earthRotRate = 2 * Math.PI / 86400; // rad/s
  const r = Math.sqrt(x * x + y * y + z * z);
  const lat = Math.asin(z / r) / DEG2RAD;
  const lon = ((Math.atan2(y, x) / DEG2RAD) - (timeSec * earthRotRate / DEG2RAD) + 360) % 360;
  const lonNorm = lon > 180 ? lon - 360 : lon;

  return { lat, lon: lonNorm, alt: altitudeKm, x, y, z };
}

/**
 * Power model: compute battery state over time.
 */
export function powerModel(solarWatts, batteryWh, eclipseFrac, periodSec, computeWatts, numSteps = 60) {
  const dt = periodSec / numSteps;
  const states = [];
  let battery = batteryWh; // Start full

  for (let i = 0; i < numSteps; i++) {
    const orbitFrac = i / numSteps;
    const inEclipse = orbitFrac >= (1 - eclipseFrac); // Eclipse at end of orbit

    const solarIn = inEclipse ? 0 : solarWatts;
    const load = computeWatts;
    const netPower = solarIn - load;
    battery = Math.max(0, Math.min(batteryWh, battery + (netPower * dt / 3600)));

    const canCompute = battery > batteryWh * 0.1; // Need >10% battery

    states.push({
      step: i,
      time: i * dt,
      orbitFrac,
      inEclipse,
      solarIn,
      battery,
      batteryPct: (battery / batteryWh) * 100,
      canCompute,
      computing: canCompute
    });
  }
  return states;
}

/**
 * Thermal model: simplified steady-state.
 */
export function thermalModel(solarWatts, computeWatts, inEclipse) {
  const baseTemp = inEclipse ? -40 : 20; // Celsius
  const heatGen = computeWatts * 0.7; // 70% of compute power becomes heat
  const radiatorEff = 0.85;
  const tempRise = heatGen / (radiatorEff * 5); // Simplified
  return Math.round(baseTemp + tempRise);
}

/**
 * Greedy job scheduler.
 * Assigns jobs to satellites based on availability and power.
 */
export function scheduleJobs(numJobs, satellites, simHours) {
  const jobs = [];
  for (let i = 0; i < numJobs; i++) {
    jobs.push({
      id: i,
      submitTime: Math.random() * simHours * 3600,
      durationSec: 120 + Math.random() * 600, // 2-12 minutes
      gpuCount: Math.ceil(Math.random() * 4),
      status: 'pending',
      assignedSat: null,
      startTime: null,
      endTime: null,
      priority: Math.random()
    });
  }
  jobs.sort((a, b) => a.submitTime - b.submitTime);

  // Greedy assignment
  const satBusyUntil = new Array(satellites.length).fill(0);

  for (const job of jobs) {
    let bestSat = -1;
    let bestTime = Infinity;

    for (let s = 0; s < satellites.length; s++) {
      const earliest = Math.max(job.submitTime, satBusyUntil[s]);
      if (earliest < bestTime) {
        bestTime = earliest;
        bestSat = s;
      }
    }

    if (bestSat >= 0) {
      job.assignedSat = bestSat;
      job.startTime = bestTime;
      job.endTime = bestTime + job.durationSec;
      job.status = 'completed';
      satBusyUntil[bestSat] = job.endTime;
    }
  }

  return jobs;
}

/**
 * Ground station contact windows.
 */
const GROUND_STATIONS = [
  { name: 'Svalbard', lat: 78.2, lon: 15.6 },
  { name: 'McMurdo', lat: -77.8, lon: 166.7 },
  { name: 'Singapore', lat: 1.3, lon: 103.8 },
  { name: 'Hawaii', lat: 19.8, lon: -155.5 },
  { name: 'Santiago', lat: -33.4, lon: -70.6 }
];

export function getGroundStations() {
  return GROUND_STATIONS;
}

/**
 * Check if satellite is in contact with a ground station.
 * @param {number} satLat
 * @param {number} satLon
 * @param {number} gsLat
 * @param {number} gsLon
 * @param {number} altitudeKm
 * @returns {boolean}
 */
export function inContact(satLat, satLon, gsLat, gsLon, altitudeKm) {
  // Simple great-circle distance check
  const dLat = (satLat - gsLat) * DEG2RAD;
  const dLon = (satLon - gsLon) * DEG2RAD;
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(satLat * DEG2RAD) * Math.cos(gsLat * DEG2RAD) * Math.sin(dLon / 2) ** 2;
  const dist = 2 * Math.asin(Math.sqrt(a)) * EARTH_RADIUS_KM;
  const maxRange = Math.sqrt(2 * EARTH_RADIUS_KM * altitudeKm + altitudeKm * altitudeKm);
  return dist < maxRange;
}

/**
 * Run full simulation.
 */
export function runSimulation(params = {}) {
  const {
    numSatellites = 12,
    simHours = 12,
    numJobs = 80,
    solarWatts = 2000,
    batteryWh = 5000,
    altitudeKm = 550,
    inclination = 53,
    computeWatts = 800
  } = params;

  const period = orbitalPeriod(altitudeKm);
  const eclFrac = eclipseFraction(altitudeKm, 23.5);
  const totalSteps = Math.ceil((simHours * 3600) / (period / 60));
  const dt = (simHours * 3600) / totalSteps;

  // Initialize satellites in evenly-spaced orbital planes
  const satellites = [];
  for (let i = 0; i < numSatellites; i++) {
    const planeIndex = Math.floor(i / Math.ceil(numSatellites / 3));
    const phaseInPlane = (i % Math.ceil(numSatellites / 3)) * (360 / Math.ceil(numSatellites / 3));
    satellites.push({
      id: i,
      name: `SAT-${String(i + 1).padStart(3, '0')}`,
      altitude: altitudeKm,
      inclination,
      raan: planeIndex * 120,
      phase0: phaseInPlane,
      gpuCount: 8,
      states: []
    });
  }

  // Simulate each timestep
  for (let t = 0; t < totalSteps; t++) {
    const timeSec = t * dt;

    for (const sat of satellites) {
      const pos = satPosition(sat.altitude, sat.inclination, sat.raan, timeSec, sat.phase0);
      const orbitFrac = (timeSec % period) / period;
      const isEclipse = orbitFrac >= (1 - eclFrac);
      const solarIn = isEclipse ? 0 : solarWatts;

      // Battery model
      const prevBattery = t > 0 ? sat.states[t - 1].battery : batteryWh;
      const load = computeWatts;
      const netPower = solarIn - load;
      const battery = Math.max(0, Math.min(batteryWh, prevBattery + (netPower * dt / 3600)));
      const canCompute = battery > batteryWh * 0.1;

      const temp = thermalModel(solarWatts, canCompute ? computeWatts : 0, isEclipse);

      // Ground station contacts
      const contacts = GROUND_STATIONS.filter(gs => inContact(pos.lat, pos.lon, gs.lat, gs.lon, sat.altitude));

      sat.states.push({
        step: t,
        timeSec,
        timeHours: timeSec / 3600,
        lat: pos.lat,
        lon: pos.lon,
        x: pos.x,
        y: pos.y,
        z: pos.z,
        inEclipse: isEclipse,
        solarIn,
        battery,
        batteryPct: (battery / batteryWh) * 100,
        canCompute,
        computing: canCompute,
        tempC: temp,
        groundContacts: contacts.map(gs => gs.name)
      });
    }
  }

  // Schedule jobs
  const jobs = scheduleJobs(numJobs, satellites, simHours);

  // Compute fleet statistics
  const completedJobs = jobs.filter(j => j.status === 'completed');
  const avgLatency = completedJobs.reduce((sum, j) => sum + (j.startTime - j.submitTime), 0) / completedJobs.length;

  const totalComputeSteps = satellites.reduce((sum, sat) => sum + sat.states.filter(s => s.computing).length, 0);
  const totalStepsAll = satellites.reduce((sum, sat) => sum + sat.states.length, 0);
  const fleetUtilization = totalComputeSteps / totalStepsAll;

  const avgBattery = satellites.reduce((sum, sat) => {
    const satAvg = sat.states.reduce((s, st) => s + st.batteryPct, 0) / sat.states.length;
    return sum + satAvg;
  }, 0) / numSatellites;

  const totalEclipseSteps = satellites.reduce((sum, sat) => sum + sat.states.filter(s => s.inEclipse).length, 0);
  const eclipsePercent = totalEclipseSteps / totalStepsAll;

  const stats = {
    numSatellites,
    simHours,
    numJobs,
    completedJobs: completedJobs.length,
    failedJobs: numJobs - completedJobs.length,
    fleetUtilization: (fleetUtilization * 100).toFixed(1),
    avgLatencySec: avgLatency.toFixed(1),
    avgBatteryPct: avgBattery.toFixed(1),
    eclipsePercent: (eclipsePercent * 100).toFixed(1),
    orbitalPeriodMin: (period / 60).toFixed(1),
    eclipseFraction: (eclFrac * 100).toFixed(1),
    totalTimesteps: totalSteps,
    altitudeKm,
    inclination
  };

  // Compute per-satellite summaries
  const satSummaries = satellites.map(sat => {
    const computeSteps = sat.states.filter(s => s.computing).length;
    const eclipseSteps = sat.states.filter(s => s.inEclipse).length;
    const avgBat = sat.states.reduce((s, st) => s + st.batteryPct, 0) / sat.states.length;
    const minBat = Math.min(...sat.states.map(s => s.batteryPct));
    const avgTemp = sat.states.reduce((s, st) => s + st.tempC, 0) / sat.states.length;
    const jobsAssigned = jobs.filter(j => j.assignedSat === sat.id).length;
    const contactCount = sat.states.filter(s => s.groundContacts.length > 0).length;

    return {
      id: sat.id,
      name: sat.name,
      utilization: ((computeSteps / sat.states.length) * 100).toFixed(1),
      eclipsePercent: ((eclipseSteps / sat.states.length) * 100).toFixed(1),
      avgBattery: avgBat.toFixed(1),
      minBattery: minBat.toFixed(1),
      avgTemp: avgTemp.toFixed(1),
      jobsCompleted: jobsAssigned,
      contactWindows: contactCount
    };
  });

  return {
    params: { numSatellites, simHours, numJobs, solarWatts, batteryWh, altitudeKm, inclination, computeWatts },
    stats,
    satellites: satellites.map(sat => ({
      id: sat.id,
      name: sat.name,
      altitude: sat.altitude,
      inclination: sat.inclination,
      raan: sat.raan,
      phase0: sat.phase0,
      // Downsample states to keep data manageable (every 3rd step)
      states: sat.states.filter((_, i) => i % 3 === 0)
    })),
    satSummaries,
    jobs: jobs.slice(0, 200), // Cap job list
    groundStations: GROUND_STATIONS
  };
}
