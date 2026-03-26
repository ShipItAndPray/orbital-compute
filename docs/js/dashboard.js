// === Dashboard — Charts and Data Display ===
import { runSimulation } from './simulation.js';

let simData = null;

// --- Transform pre-computed JSON (Python sim) into the internal format ---

function transformPrecomputedData(raw) {
  // If data already has 'stats' and 'satSummaries', it came from runSimulation() — use as-is
  if (raw.stats && raw.satSummaries) return raw;

  const config = raw.config || {};
  const scheduler = raw.scheduler || {};
  const numSats = config.n_satellites || (raw.satellites ? raw.satellites.length : 12);
  const simHours = config.sim_hours || 12;
  const numJobs = scheduler.total_jobs || config.n_jobs || 80;
  const completedCount = scheduler.completed || numJobs;

  // Build satellite states from the raw positions/power/thermal arrays
  const satellites = (raw.satellites || []).map((sat, idx) => {
    const positions = sat.positions || [];
    const power = sat.power_history || [];
    const thermal = sat.thermal_history || [];

    const states = positions.map((pos, i) => {
      const pw = power[i] || {};
      const th = thermal[i] || {};
      const timeSec = (pos.min || 0) * 60;
      return {
        step: i,
        timeSec,
        timeHours: timeSec / 3600,
        lat: pos.lat,
        lon: pos.lon,
        inEclipse: pos.eclipse || pw.in_eclipse || false,
        battery: (pw.battery_pct || 0) * (config.battery_wh || 5000),
        batteryPct: (pw.battery_pct || 0) * 100,
        canCompute: pw.computing || false,
        computing: pw.computing || false,
        tempC: th.temp_c || 0,
        groundContacts: []
      };
    });

    return {
      id: idx,
      name: sat.name || `SAT-${String(idx).padStart(3, '0')}`,
      altitude: 550,
      inclination: 53,
      raan: 0,
      phase0: 0,
      states
    };
  });

  // Build satSummaries from satellite_details dict or from computed states
  const satDetails = raw.satellite_details || {};
  const satSummaries = satellites.map(sat => {
    const detail = satDetails[sat.name] || {};
    const jobsForSat = (raw.completed_jobs || []).filter(j => j.satellite === sat.name).length;

    if (Object.keys(detail).length > 0) {
      return {
        id: sat.id,
        name: sat.name,
        utilization: (detail.compute_pct || 0).toFixed(1),
        eclipsePercent: (detail.eclipse_pct || 0).toFixed(1),
        avgBattery: (detail.avg_battery_pct || 0).toFixed(1),
        minBattery: (detail.min_battery_pct || 0).toFixed(1),
        avgTemp: (detail.avg_temp_c || 0).toFixed(1),
        jobsCompleted: jobsForSat,
        contactWindows: detail.contact_windows || 0
      };
    }
    // Fallback: compute from states
    const states = sat.states;
    const computeSteps = states.filter(s => s.computing).length;
    const eclipseSteps = states.filter(s => s.inEclipse).length;
    const avgBat = states.length > 0 ? states.reduce((s, st) => s + st.batteryPct, 0) / states.length : 0;
    const minBat = states.length > 0 ? Math.min(...states.map(s => s.batteryPct)) : 0;
    const avgTemp = states.length > 0 ? states.reduce((s, st) => s + st.tempC, 0) / states.length : 0;

    return {
      id: sat.id,
      name: sat.name,
      utilization: states.length > 0 ? ((computeSteps / states.length) * 100).toFixed(1) : '0.0',
      eclipsePercent: states.length > 0 ? ((eclipseSteps / states.length) * 100).toFixed(1) : '0.0',
      avgBattery: avgBat.toFixed(1),
      minBattery: minBat.toFixed(1),
      avgTemp: avgTemp.toFixed(1),
      jobsCompleted: jobsForSat,
      contactWindows: 0
    };
  });

  // Compute aggregate stats
  const avgBatteryPct = satSummaries.length > 0
    ? (satSummaries.reduce((s, ss) => s + parseFloat(ss.avgBattery), 0) / satSummaries.length).toFixed(1)
    : '0.0';
  const avgEclipse = satSummaries.length > 0
    ? (satSummaries.reduce((s, ss) => s + parseFloat(ss.eclipsePercent), 0) / satSummaries.length).toFixed(1)
    : '0.0';

  // Estimate orbital period from positions (time between entries * count ~ one orbit)
  const periodMin = 95.6; // Typical LEO ~550km

  // Compute avg latency from completed_jobs if durations available
  const completedJobs = raw.completed_jobs || [];
  const avgLatency = completedJobs.length > 0
    ? (completedJobs.reduce((s, j) => s + (j.duration_s || 0), 0) / completedJobs.length).toFixed(1)
    : '0.0';

  const totalSteps = satellites.length > 0 ? satellites[0].states.length : 0;

  const stats = {
    numSatellites: numSats,
    simHours,
    numJobs,
    completedJobs: completedCount,
    failedJobs: numJobs - completedCount,
    fleetUtilization: (raw.fleet_utilization_pct || 0).toFixed(1),
    avgLatencySec: avgLatency,
    avgBatteryPct,
    eclipsePercent: avgEclipse,
    orbitalPeriodMin: periodMin.toFixed(1),
    eclipseFraction: avgEclipse,
    totalTimesteps: totalSteps,
    altitudeKm: 550,
    inclination: 53
  };

  // Build jobs list in internal format (for cumulative chart)
  const jobs = completedJobs.map((j, i) => ({
    id: i,
    status: 'completed',
    assignedSat: satellites.findIndex(s => s.name === j.satellite),
    submitTime: 0,
    startTime: i * (simHours * 3600 / Math.max(completedJobs.length, 1)),
    endTime: (i + 1) * (simHours * 3600 / Math.max(completedJobs.length, 1)),
    durationSec: j.duration_s || 300
  }));

  return {
    params: config,
    stats,
    satellites,
    satSummaries,
    jobs,
    groundStations: raw.ground_stations || []
  };
}

// --- Simple Canvas Chart Library ---

function clearCanvas(canvas) {
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * dpr;
  canvas.height = canvas.clientHeight * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  return ctx;
}

function roundRectPolyfill(ctx, x, y, w, h, radii) {
  // Polyfill for browsers without ctx.roundRect
  const r = typeof radii === 'number' ? radii : (radii[0] || 0);
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y, x + w, y + r, r);
  ctx.lineTo(x + w, y + h);
  ctx.lineTo(x, y + h);
  ctx.lineTo(x, y + r);
  ctx.arcTo(x, y, x + r, y, r);
  ctx.closePath();
}

function drawBarChart(canvas, labels, values, { color = '#4fc3f7', maxVal = null, unit = '' } = {}) {
  const ctx = clearCanvas(canvas);
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const pad = { top: 20, right: 20, bottom: 40, left: 50 };
  const chartW = w - pad.left - pad.right;
  const chartH = h - pad.top - pad.bottom;

  const max = maxVal || Math.max(...values) * 1.1 || 1;
  const barW = Math.min(40, (chartW / labels.length) * 0.7);
  const gap = chartW / labels.length;

  // Grid lines
  ctx.strokeStyle = '#1e2a42';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (chartH * i / 4);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();

    ctx.fillStyle = '#5a6478';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'right';
    const val = max - (max * i / 4);
    ctx.fillText(val.toFixed(0) + unit, pad.left - 8, y + 4);
  }

  // Bars
  labels.forEach((label, i) => {
    const x = pad.left + i * gap + (gap - barW) / 2;
    const barH = (values[i] / max) * chartH;
    const y = pad.top + chartH - barH;

    ctx.fillStyle = color;
    ctx.globalAlpha = 0.8;
    ctx.beginPath();
    if (ctx.roundRect) {
      ctx.roundRect(x, y, barW, barH, [3, 3, 0, 0]);
    } else {
      roundRectPolyfill(ctx, x, y, barW, barH, [3, 3, 0, 0]);
    }
    ctx.fill();
    ctx.globalAlpha = 1;

    // Label
    ctx.fillStyle = '#8892a4';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(label, x + barW / 2, h - pad.bottom + 16);
  });
}

function drawLineChart(canvas, datasets, { xLabels = [], yUnit = '', maxY = null } = {}) {
  const ctx = clearCanvas(canvas);
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const pad = { top: 20, right: 20, bottom: 40, left: 50 };
  const chartW = w - pad.left - pad.right;
  const chartH = h - pad.top - pad.bottom;

  const allVals = datasets.flatMap(d => d.values);
  const max = maxY || Math.max(...allVals) * 1.1 || 1;
  const min = 0;

  // Grid
  ctx.strokeStyle = '#1e2a42';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (chartH * i / 4);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();

    ctx.fillStyle = '#5a6478';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(((max - min) * (1 - i / 4) + min).toFixed(0) + yUnit, pad.left - 8, y + 4);
  }

  // X labels
  if (xLabels.length > 0) {
    const step = Math.max(1, Math.floor(xLabels.length / 8));
    xLabels.forEach((label, i) => {
      if (i % step !== 0) return;
      const x = pad.left + (i / Math.max(xLabels.length - 1, 1)) * chartW;
      ctx.fillStyle = '#5a6478';
      ctx.font = '10px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(label, x, h - pad.bottom + 16);
    });
  }

  // Lines
  const colors = ['#4fc3f7', '#66bb6a', '#ffa726', '#ef5350', '#ab47bc', '#26a69a'];
  datasets.forEach((ds, di) => {
    if (!ds.values || ds.values.length === 0) return;
    ctx.strokeStyle = colors[di % colors.length];
    ctx.lineWidth = 2;
    ctx.beginPath();
    ds.values.forEach((v, i) => {
      const x = pad.left + (i / Math.max(ds.values.length - 1, 1)) * chartW;
      const y = pad.top + chartH - ((v - min) / (max - min)) * chartH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Legend
    ctx.fillStyle = colors[di % colors.length];
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(ds.label, pad.left + di * 100, pad.top - 6);
  });
}

function drawTimelineChart(canvas, satellites, simHours) {
  const ctx = clearCanvas(canvas);
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const pad = { top: 30, right: 20, bottom: 30, left: 80 };
  const chartW = w - pad.left - pad.right;
  const chartH = h - pad.top - pad.bottom;

  const numSats = Math.min(satellites.length, 12);
  const rowH = chartH / numSats;

  // Time axis
  for (let hr = 0; hr <= simHours; hr += Math.max(1, Math.floor(simHours / 8))) {
    const x = pad.left + (hr / simHours) * chartW;
    ctx.strokeStyle = '#1e2a42';
    ctx.beginPath();
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, h - pad.bottom);
    ctx.stroke();

    ctx.fillStyle = '#5a6478';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(hr + 'h', x, h - pad.bottom + 16);
  }

  // Title
  ctx.fillStyle = '#8892a4';
  ctx.font = '11px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Time', w / 2, h - 4);

  for (let s = 0; s < numSats; s++) {
    const sat = satellites[s];
    const y = pad.top + s * rowH;

    // Sat label
    ctx.fillStyle = '#8892a4';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(sat.name, pad.left - 8, y + rowH / 2 + 4);

    // Draw state blocks
    const states = sat.states;
    if (!states || states.length === 0) continue;

    const totalSec = simHours * 3600;
    for (let i = 0; i < states.length; i++) {
      const st = states[i];
      const nextSt = states[i + 1] || null;
      const x1 = pad.left + (st.timeSec / totalSec) * chartW;
      const x2 = nextSt
        ? pad.left + (nextSt.timeSec / totalSec) * chartW
        : x1 + Math.max(1, chartW / states.length);
      const bw = Math.max(1, x2 - x1);

      if (st.inEclipse) {
        ctx.fillStyle = 'rgba(239, 83, 80, 0.4)';
      } else if (st.computing) {
        ctx.fillStyle = 'rgba(79, 195, 247, 0.5)';
      } else {
        ctx.fillStyle = 'rgba(102, 187, 106, 0.2)';
      }

      ctx.fillRect(x1, y + 2, bw, rowH - 4);
    }
  }

  // Legend
  const legendY = pad.top - 16;
  const items = [
    { color: 'rgba(79, 195, 247, 0.5)', label: 'Computing' },
    { color: 'rgba(239, 83, 80, 0.4)', label: 'Eclipse' },
    { color: 'rgba(102, 187, 106, 0.2)', label: 'Idle/Sunlit' }
  ];
  let lx = pad.left;
  items.forEach(item => {
    ctx.fillStyle = item.color;
    ctx.fillRect(lx, legendY - 8, 14, 10);
    ctx.fillStyle = '#8892a4';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(item.label, lx + 18, legendY);
    lx += 100;
  });
}

// --- DOM Helpers ---

function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

function updateStatCard(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function colorForMetric(value, thresholds) {
  // thresholds = { good: 70, warn: 40 } — above good is green, above warn is orange, below is red
  const v = parseFloat(value);
  if (v >= thresholds.good) return 'var(--success)';
  if (v >= thresholds.warn) return 'var(--warning)';
  return 'var(--danger)';
}

function renderSatTable(summaries) {
  const tbody = $('#sat-table-body');
  if (!tbody) return;
  tbody.innerHTML = summaries.map(s => {
    const utilColor = colorForMetric(s.utilization, { good: 50, warn: 20 });
    const eclipseColor = colorForMetric(100 - parseFloat(s.eclipsePercent), { good: 60, warn: 40 });
    const battColor = colorForMetric(s.avgBattery, { good: 60, warn: 30 });
    const minBattColor = colorForMetric(s.minBattery, { good: 30, warn: 15 });
    const tempVal = parseFloat(s.avgTemp);
    const tempColor = tempVal > 40 ? 'var(--danger)' : tempVal > 20 ? 'var(--warning)' : 'var(--accent)';

    return `<tr>
      <td><strong>${s.name}</strong></td>
      <td style="color:${utilColor};font-weight:600">${s.utilization}%</td>
      <td style="color:${eclipseColor};font-weight:600">${s.eclipsePercent}%</td>
      <td style="color:${battColor};font-weight:600">${s.avgBattery}%</td>
      <td style="color:${minBattColor};font-weight:600">${s.minBattery}%</td>
      <td style="color:${tempColor};font-weight:600">${s.avgTemp}&deg;C</td>
      <td>${s.jobsCompleted}</td>
      <td>${s.contactWindows}</td>
    </tr>`;
  }).join('');
}

function showLoading(show) {
  let overlay = $('#loading-overlay');
  if (show) {
    if (!overlay) {
      overlay = document.createElement('div');
      overlay.id = 'loading-overlay';
      overlay.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(10,14,23,0.7);display:flex;align-items:center;justify-content:center;z-index:9999;backdrop-filter:blur(4px)';
      overlay.innerHTML = '<div style="text-align:center"><div class="spinner" style="width:48px;height:48px;border-width:4px;margin:0 auto 1rem"></div><div style="color:var(--accent);font-weight:600;font-size:1.1rem">Running Simulation...</div><div style="color:var(--text-secondary);font-size:0.85rem;margin-top:0.5rem">Computing orbital mechanics</div></div>';
      document.body.appendChild(overlay);
    }
    overlay.style.display = 'flex';
  } else if (overlay) {
    overlay.style.display = 'none';
  }
}

function renderDashboard(data) {
  const { stats, satellites, satSummaries, jobs } = data;

  // Stat cards
  updateStatCard('stat-sats', stats.numSatellites);
  updateStatCard('stat-util', stats.fleetUtilization + '%');
  updateStatCard('stat-jobs', `${stats.completedJobs}/${stats.numJobs}`);
  updateStatCard('stat-latency', stats.avgLatencySec + 's');
  updateStatCard('stat-battery', stats.avgBatteryPct + '%');
  updateStatCard('stat-eclipse', stats.eclipsePercent + '%');
  updateStatCard('stat-period', stats.orbitalPeriodMin + ' min');
  updateStatCard('stat-steps', stats.totalTimesteps);

  // Table
  renderSatTable(satSummaries);

  // Charts
  const utilCanvas = $('#chart-utilization');
  if (utilCanvas && satSummaries.length > 0) {
    drawBarChart(
      utilCanvas,
      satSummaries.map(s => s.name.replace('SAT-', '')),
      satSummaries.map(s => parseFloat(s.utilization)),
      { maxVal: 100, unit: '%' }
    );
  }

  const batteryCanvas = $('#chart-battery');
  if (batteryCanvas && satellites.length > 0) {
    const firstSat = satellites[0];
    const secondSat = satellites.length > 1 ? satellites[1] : null;
    if (firstSat.states && firstSat.states.length > 0) {
      const ds = [{ label: firstSat.name, values: firstSat.states.map(s => s.batteryPct) }];
      if (secondSat && secondSat.states && secondSat.states.length > 0) {
        ds.push({ label: secondSat.name, values: secondSat.states.map(s => s.batteryPct) });
      }
      const xLabels = firstSat.states.map(s => (s.timeHours != null ? s.timeHours : s.timeSec / 3600).toFixed(1) + 'h');
      drawLineChart(batteryCanvas, ds, { xLabels, yUnit: '%', maxY: 100 });
    }
  }

  const timelineCanvas = $('#chart-timeline');
  if (timelineCanvas && satellites.length > 0) {
    drawTimelineChart(timelineCanvas, satellites, stats.simHours);
  }

  // Job completion chart
  const jobCanvas = $('#chart-jobs');
  if (jobCanvas && jobs && jobs.length > 0) {
    const completedJobs = jobs.filter(j => j.status === 'completed').sort((a, b) => a.endTime - b.endTime);
    if (completedJobs.length > 0) {
      const cumulative = [];
      const labels = [];
      completedJobs.forEach((j, i) => {
        cumulative.push(i + 1);
        labels.push((j.endTime / 3600).toFixed(1) + 'h');
      });
      // Downsample for display
      const step = Math.max(1, Math.floor(cumulative.length / 60));
      const dsValues = cumulative.filter((_, i) => i % step === 0);
      const dsLabels = labels.filter((_, i) => i % step === 0);
      drawLineChart(jobCanvas, [{ label: 'Jobs Completed', values: dsValues }], { xLabels: dsLabels });
    }
  }
}

// --- Initialization ---

function bindControls() {
  // Sync slider values to display
  $$('.control-group input[type="range"]').forEach(slider => {
    const valueSpan = slider.parentElement.querySelector('.control-value');
    if (valueSpan) {
      valueSpan.textContent = slider.value;
      slider.addEventListener('input', () => { valueSpan.textContent = slider.value; });
    }
  });

  // Run button
  const runBtn = $('#run-sim-btn');
  if (runBtn) {
    runBtn.addEventListener('click', () => {
      runBtn.disabled = true;
      runBtn.textContent = '';
      const spinEl = document.createElement('span');
      spinEl.className = 'spinner';
      spinEl.style.cssText = 'width:16px;height:16px;border-width:2px;vertical-align:middle;margin-right:8px';
      runBtn.appendChild(spinEl);
      runBtn.appendChild(document.createTextNode('Running...'));
      showLoading(true);

      // Use setTimeout to let the UI update before heavy computation
      setTimeout(() => {
        try {
          const params = {
            numSatellites: parseInt($('#ctrl-sats')?.value || '12'),
            simHours: parseInt($('#ctrl-hours')?.value || '12'),
            numJobs: parseInt($('#ctrl-jobs')?.value || '80'),
            solarWatts: parseInt($('#ctrl-solar')?.value || '2000'),
            batteryWh: parseInt($('#ctrl-battery')?.value || '5000')
          };

          simData = runSimulation(params);
          renderDashboard(simData);
        } catch (err) {
          console.error('Simulation error:', err);
        } finally {
          showLoading(false);
          runBtn.disabled = false;
          runBtn.textContent = 'Run Simulation';
        }
      }, 100);
    });
  }
}

async function init() {
  bindControls();

  // Load pre-computed data first, then allow re-run
  try {
    const resp = await fetch('data/sim_data.json');
    if (!resp.ok) throw new Error('Failed to load sim_data.json');
    const raw = await resp.json();
    simData = transformPrecomputedData(raw);
    renderDashboard(simData);
  } catch (e) {
    console.log('No pre-computed data found, running live simulation...', e);
    try {
      simData = runSimulation();
      renderDashboard(simData);
    } catch (err) {
      console.error('Simulation failed:', err);
      const tbody = $('#sat-table-body');
      if (tbody) tbody.innerHTML = '<tr><td colspan="8" class="text-muted text-center">Simulation failed. Try clicking Run Simulation.</td></tr>';
    }
  }
}

document.addEventListener('DOMContentLoaded', init);
