// === Dashboard — Charts and Data Display ===
import { runSimulation } from './simulation.js';

let simData = null;

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
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const pad = { top: 20, right: 20, bottom: 40, left: 50 };
  const chartW = w - pad.left - pad.right;
  const chartH = h - pad.top - pad.bottom;
  const max = maxVal || Math.max(...values) * 1.1 || 1;
  const barW = Math.min(40, (chartW / labels.length) * 0.7);
  const gap = chartW / labels.length;

  ctx.strokeStyle = '#1e2a42'; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (chartH * i / 4);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
    ctx.fillStyle = '#5a6478'; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'right';
    ctx.fillText((max - (max * i / 4)).toFixed(0) + unit, pad.left - 8, y + 4);
  }

  labels.forEach((label, i) => {
    const x = pad.left + i * gap + (gap - barW) / 2;
    const barH = (values[i] / max) * chartH;
    const y = pad.top + chartH - barH;
    ctx.fillStyle = color; ctx.globalAlpha = 0.8;
    ctx.beginPath(); ctx.roundRect(x, y, barW, barH, [3, 3, 0, 0]); ctx.fill();
    ctx.globalAlpha = 1;
    ctx.fillStyle = '#8892a4'; ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(label, x + barW / 2, h - pad.bottom + 16);
  });
}

function drawLineChart(canvas, datasets, { xLabels = [], yUnit = '', maxY = null } = {}) {
  const ctx = clearCanvas(canvas);
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const pad = { top: 20, right: 20, bottom: 40, left: 50 };
  const chartW = w - pad.left - pad.right;
  const chartH = h - pad.top - pad.bottom;
  const max = maxY || Math.max(...datasets.flatMap(d => d.values)) * 1.1 || 1;

  ctx.strokeStyle = '#1e2a42'; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (chartH * i / 4);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
    ctx.fillStyle = '#5a6478'; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'right';
    ctx.fillText((max * (1 - i / 4)).toFixed(0) + yUnit, pad.left - 8, y + 4);
  }

  if (xLabels.length > 0) {
    const step = Math.max(1, Math.floor(xLabels.length / 8));
    xLabels.forEach((label, i) => {
      if (i % step !== 0) return;
      const x = pad.left + (i / (xLabels.length - 1)) * chartW;
      ctx.fillStyle = '#5a6478'; ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'center';
      ctx.fillText(label, x, h - pad.bottom + 16);
    });
  }

  const colors = ['#4fc3f7', '#66bb6a', '#ffa726', '#ef5350', '#ab47bc', '#26a69a'];
  datasets.forEach((ds, di) => {
    ctx.strokeStyle = colors[di % colors.length]; ctx.lineWidth = 2; ctx.beginPath();
    ds.values.forEach((v, i) => {
      const x = pad.left + (i / Math.max(1, ds.values.length - 1)) * chartW;
      const y = pad.top + chartH - (v / max) * chartH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = colors[di % colors.length]; ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'left'; ctx.fillText(ds.label, pad.left + di * 100, pad.top - 6);
  });
}

function drawTimelineChart(canvas, satellites, simHours) {
  const ctx = clearCanvas(canvas);
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const pad = { top: 30, right: 20, bottom: 30, left: 80 };
  const chartW = w - pad.left - pad.right;
  const chartH = h - pad.top - pad.bottom;
  const numSats = Math.min(satellites.length, 12);
  const rowH = chartH / numSats;

  for (let hr = 0; hr <= simHours; hr += Math.max(1, Math.floor(simHours / 8))) {
    const x = pad.left + (hr / simHours) * chartW;
    ctx.strokeStyle = '#1e2a42'; ctx.beginPath();
    ctx.moveTo(x, pad.top); ctx.lineTo(x, h - pad.bottom); ctx.stroke();
    ctx.fillStyle = '#5a6478'; ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(hr + 'h', x, h - pad.bottom + 16);
  }

  const items = [
    { color: 'rgba(79, 195, 247, 0.4)', label: 'Computing' },
    { color: 'rgba(239, 83, 80, 0.3)', label: 'Eclipse' },
    { color: 'rgba(102, 187, 106, 0.2)', label: 'Idle/Sunlit' }
  ];
  let lx = pad.left;
  items.forEach(item => {
    ctx.fillStyle = item.color; ctx.fillRect(lx, pad.top - 20, 14, 10);
    ctx.fillStyle = '#8892a4'; ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'left';
    ctx.fillText(item.label, lx + 18, pad.top - 12); lx += 100;
  });

  for (let s = 0; s < numSats; s++) {
    const sat = satellites[s];
    const y = pad.top + s * rowH;
    ctx.fillStyle = '#8892a4'; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'right';
    ctx.fillText(sat.name, pad.left - 8, y + rowH / 2 + 4);

    const states = sat.states;
    if (!states || states.length === 0) continue;
    const totalSec = simHours * 3600;
    for (let i = 0; i < states.length - 1; i++) {
      const st = states[i], nextSt = states[i + 1];
      const x1 = pad.left + (st.timeSec / totalSec) * chartW;
      const x2 = pad.left + (nextSt.timeSec / totalSec) * chartW;
      const bw = Math.max(1, x2 - x1);
      ctx.fillStyle = st.inEclipse ? 'rgba(239, 83, 80, 0.3)'
        : st.computing ? 'rgba(79, 195, 247, 0.4)' : 'rgba(102, 187, 106, 0.2)';
      ctx.fillRect(x1, y + 2, bw, rowH - 4);
    }
  }
}

function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

function renderDashboard(data) {
  const { stats, satellites, satSummaries, jobs } = data;

  const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  set('stat-sats', stats.numSatellites);
  set('stat-util', stats.fleetUtilization + '%');
  set('stat-jobs', `${stats.completedJobs}/${stats.numJobs}`);
  set('stat-latency', stats.avgLatencySec + 's');
  set('stat-battery', stats.avgBatteryPct + '%');
  set('stat-eclipse', stats.eclipsePercent + '%');
  set('stat-period', stats.orbitalPeriodMin + ' min');
  set('stat-steps', stats.totalTimesteps);

  const tbody = $('#sat-table-body');
  if (tbody) {
    tbody.innerHTML = satSummaries.map(s => `
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

  const utilCanvas = $('#chart-utilization');
  if (utilCanvas) {
    drawBarChart(utilCanvas,
      satSummaries.map(s => s.name.replace('SAT-', '')),
      satSummaries.map(s => parseFloat(s.utilization)),
      { maxVal: 100, unit: '%' });
  }

  const batteryCanvas = $('#chart-battery');
  if (batteryCanvas && satellites.length > 0) {
    const ds = [{ label: satellites[0].name, values: satellites[0].states.map(s => s.batteryPct) }];
    if (satellites.length > 1) ds.push({ label: satellites[1].name, values: satellites[1].states.map(s => s.batteryPct) });
    const xLabels = satellites[0].states.map(s => ((s.timeHours !== undefined ? s.timeHours : s.timeSec / 3600)).toFixed(1) + 'h');
    drawLineChart(batteryCanvas, ds, { xLabels, yUnit: '%', maxY: 100 });
  }

  const timelineCanvas = $('#chart-timeline');
  if (timelineCanvas) drawTimelineChart(timelineCanvas, satellites, stats.simHours);

  const jobCanvas = $('#chart-jobs');
  if (jobCanvas && jobs.length > 0) {
    const completed = jobs.filter(j => j.status === 'completed').sort((a, b) => a.endTime - b.endTime);
    const cum = [], labels = [];
    completed.forEach((j, i) => { cum.push(i + 1); labels.push((j.endTime / 3600).toFixed(1) + 'h'); });
    const step = Math.max(1, Math.floor(cum.length / 60));
    drawLineChart(jobCanvas,
      [{ label: 'Jobs Completed', values: cum.filter((_, i) => i % step === 0) }],
      { xLabels: labels.filter((_, i) => i % step === 0) });
  }
}

function bindControls() {
  $$('.control-group input[type="range"]').forEach(slider => {
    const span = slider.parentElement.querySelector('.control-value');
    if (span) { span.textContent = slider.value; slider.addEventListener('input', () => { span.textContent = slider.value; }); }
  });

  const runBtn = $('#run-sim-btn');
  if (runBtn) {
    runBtn.addEventListener('click', () => {
      runBtn.disabled = true; runBtn.innerHTML = '<span class="spinner"></span> Running...';
      setTimeout(() => {
        simData = runSimulation({
          numSatellites: parseInt($('#ctrl-sats')?.value || 12),
          simHours: parseInt($('#ctrl-hours')?.value || 12),
          numJobs: parseInt($('#ctrl-jobs')?.value || 80),
          solarWatts: parseInt($('#ctrl-solar')?.value || 2000),
          batteryWh: parseInt($('#ctrl-battery')?.value || 5000)
        });
        renderDashboard(simData);
        runBtn.disabled = false; runBtn.innerHTML = 'Run Simulation';
      }, 50);
    });
  }
}

async function init() {
  bindControls();
  try {
    const resp = await fetch('data/sim_data.json');
    simData = await resp.json();
    renderDashboard(simData);
  } catch (e) {
    console.log('No pre-computed data, running simulation...', e);
    simData = runSimulation();
    renderDashboard(simData);
  }
}

document.addEventListener('DOMContentLoaded', init);
