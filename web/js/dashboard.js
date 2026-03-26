// === Dashboard — Charts and Data Display ===
import { runSimulation } from './simulation.js';

let simData = null;

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
    ctx.roundRect(x, y, barW, barH, [3, 3, 0, 0]);
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
      const x = pad.left + (i / (xLabels.length - 1)) * chartW;
      ctx.fillStyle = '#5a6478';
      ctx.font = '10px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(label, x, h - pad.bottom + 16);
    });
  }

  // Lines
  const colors = ['#4fc3f7', '#66bb6a', '#ffa726', '#ef5350', '#ab47bc', '#26a69a'];
  datasets.forEach((ds, di) => {
    ctx.strokeStyle = colors[di % colors.length];
    ctx.lineWidth = 2;
    ctx.beginPath();
    ds.values.forEach((v, i) => {
      const x = pad.left + (i / (ds.values.length - 1)) * chartW;
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
    for (let i = 0; i < states.length - 1; i++) {
      const st = states[i];
      const nextSt = states[i + 1] || st;
      const x1 = pad.left + (st.timeSec / totalSec) * chartW;
      const x2 = pad.left + (nextSt.timeSec / totalSec) * chartW;
      const bw = Math.max(1, x2 - x1);

      if (st.inEclipse) {
        ctx.fillStyle = 'rgba(239, 83, 80, 0.3)';
      } else if (st.computing) {
        ctx.fillStyle = 'rgba(79, 195, 247, 0.4)';
      } else {
        ctx.fillStyle = 'rgba(102, 187, 106, 0.2)';
      }

      ctx.fillRect(x1, y + 2, bw, rowH - 4);
    }
  }

  // Legend
  const legendY = pad.top - 16;
  const items = [
    { color: 'rgba(79, 195, 247, 0.4)', label: 'Computing' },
    { color: 'rgba(239, 83, 80, 0.3)', label: 'Eclipse' },
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

function renderSatTable(summaries) {
  const tbody = $('#sat-table-body');
  if (!tbody) return;
  tbody.innerHTML = summaries.map(s => `
    <tr>
      <td><strong>${s.name}</strong></td>
      <td>${s.utilization}%</td>
      <td>${s.eclipsePercent}%</td>
      <td>${s.avgBattery}%</td>
      <td>${s.minBattery}%</td>
      <td>${s.avgTemp} C</td>
      <td>${s.jobsCompleted}</td>
      <td>${s.contactWindows}</td>
    </tr>
  `).join('');
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
  if (utilCanvas) {
    drawBarChart(utilCanvas, satSummaries.map(s => s.name.replace('SAT-', '')), satSummaries.map(s => parseFloat(s.utilization)), { maxVal: 100, unit: '%' });
  }

  const batteryCanvas = $('#chart-battery');
  if (batteryCanvas && satellites.length > 0) {
    const firstSat = satellites[0];
    const secondSat = satellites.length > 1 ? satellites[1] : null;
    const ds = [{ label: firstSat.name, values: firstSat.states.map(s => s.batteryPct) }];
    if (secondSat) ds.push({ label: secondSat.name, values: secondSat.states.map(s => s.batteryPct) });
    const xLabels = firstSat.states.map(s => (s.timeHours || s.timeSec / 3600).toFixed(1) + 'h');
    drawLineChart(batteryCanvas, ds, { xLabels, yUnit: '%', maxY: 100 });
  }

  const timelineCanvas = $('#chart-timeline');
  if (timelineCanvas) {
    drawTimelineChart(timelineCanvas, satellites, stats.simHours);
  }

  // Job completion chart
  const jobCanvas = $('#chart-jobs');
  if (jobCanvas && jobs.length > 0) {
    const completedJobs = jobs.filter(j => j.status === 'completed').sort((a, b) => a.endTime - b.endTime);
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
      runBtn.innerHTML = '<span class="spinner"></span> Running...';

      requestAnimationFrame(() => {
        setTimeout(() => {
          const params = {
            numSatellites: parseInt($('#ctrl-sats')?.value || 12),
            simHours: parseInt($('#ctrl-hours')?.value || 12),
            numJobs: parseInt($('#ctrl-jobs')?.value || 80),
            solarWatts: parseInt($('#ctrl-solar')?.value || 2000),
            batteryWh: parseInt($('#ctrl-battery')?.value || 5000)
          };

          simData = runSimulation(params);
          renderDashboard(simData);

          runBtn.disabled = false;
          runBtn.innerHTML = 'Run Simulation';
        }, 50);
      });
    });
  }
}

async function init() {
  bindControls();

  // Load pre-computed data first, then allow re-run
  try {
    const resp = await fetch('data/sim_data.json');
    simData = await resp.json();
    renderDashboard(simData);
  } catch (e) {
    console.log('No pre-computed data, running simulation...');
    simData = runSimulation();
    renderDashboard(simData);
  }
}

document.addEventListener('DOMContentLoaded', init);
