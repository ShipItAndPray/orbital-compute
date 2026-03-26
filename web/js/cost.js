// === Cost Calculator ===

const LAUNCH_VEHICLES = {
  'falcon9-rideshare': { name: 'Falcon 9 Rideshare', costPerKg: 2720, maxKg: 800 },
  'falcon9-dedicated': { name: 'Falcon 9 Dedicated', costPerKg: 4400, maxKg: 8300 },
  'electron': { name: 'Rocket Lab Electron', costPerKg: 25000, maxKg: 300 },
  'starship': { name: 'SpaceX Starship', costPerKg: 200, maxKg: 100000 }
};

const CLOUD_COMPARISON = {
  'aws-p4d': { name: 'AWS p4d.24xlarge', gpuHourCost: 4.10, gpuType: 'A100' },
  'aws-p5': { name: 'AWS p5.48xlarge', gpuHourCost: 12.00, gpuType: 'H100' },
  'colo-a100': { name: 'Colo A100', gpuHourCost: 1.50, gpuType: 'A100' },
  'colo-h100': { name: 'Colo H100', gpuHourCost: 2.50, gpuType: 'H100' }
};

function computeCost(params) {
  const {
    numSatellites,
    gpusPerSat,
    solarWatts,
    launchVehicle,
    utilization
  } = params;

  const vehicle = LAUNCH_VEHICLES[launchVehicle];

  // Satellite hardware costs
  const gpuCost = gpusPerSat * 25000; // $25K per GPU (A100-class)
  const solarPanelCost = solarWatts * 3; // $3/W for space-grade
  const batteryCost = 5000; // Fixed per sat
  const structureCost = 20000; // Bus, thermal, etc.
  const computeBoardCost = gpusPerSat * 5000; // Supporting hardware
  const commsCost = 15000; // ISL + ground link
  const satHardwareCost = gpuCost + solarPanelCost + batteryCost + structureCost + computeBoardCost + commsCost;

  // Satellite mass estimate
  const satMassKg = 50 + gpusPerSat * 5 + solarWatts * 0.005;

  // Launch cost
  const launchCostPerSat = satMassKg * vehicle.costPerKg;
  const satsPerLaunch = Math.floor(vehicle.maxKg / satMassKg);
  const numLaunches = Math.ceil(numSatellites / satsPerLaunch);

  // Total CAPEX
  const totalHardware = satHardwareCost * numSatellites;
  const totalLaunch = launchCostPerSat * numSatellites;
  const integrationCost = numSatellites * 30000; // Testing, integration
  const groundSegment = 2000000; // Ground stations, ops center
  const insurance = (totalHardware + totalLaunch) * 0.08;

  const totalCapex = totalHardware + totalLaunch + integrationCost + groundSegment + insurance;

  // Annual OPEX
  const operationsCrew = 800000; // 5-person ops team
  const groundStationOps = 300000;
  const bandwidth = numSatellites * 12000; // Per sat per year
  const softwareLicenses = 100000;
  const spare = totalCapex * 0.03; // 3% spares/replacement

  const annualOpex = operationsCrew + groundStationOps + bandwidth + softwareLicenses + spare;

  // Revenue model
  const totalGPUs = numSatellites * gpusPerSat;
  const hoursPerYear = 8760;
  const effectiveComputeHours = totalGPUs * hoursPerYear * (utilization / 100) * 0.65; // 65% sunlit compute factor
  const lifetimeYears = 5;
  const totalComputeHours = effectiveComputeHours * lifetimeYears;
  const costPerGPUHour = (totalCapex + annualOpex * lifetimeYears) / totalComputeHours;

  // Break-even
  const revenuePerHour = 3.00; // Target price
  const monthlyRevenue = totalGPUs * 730 * (utilization / 100) * 0.65 * revenuePerHour;
  const monthlyCost = annualOpex / 12;
  const breakEvenMonths = totalCapex / (monthlyRevenue - monthlyCost);

  // Sensitivity analysis
  const sensitivity = [];
  for (let util = 20; util <= 100; util += 10) {
    const effHours = totalGPUs * hoursPerYear * (util / 100) * 0.65;
    const totalHours = effHours * lifetimeYears;
    const cph = (totalCapex + annualOpex * lifetimeYears) / totalHours;
    const monthRev = totalGPUs * 730 * (util / 100) * 0.65 * revenuePerHour;
    const beMonths = totalCapex / Math.max(1, monthRev - monthlyCost);
    sensitivity.push({
      utilization: util,
      costPerGPUHour: cph,
      breakEvenMonths: Math.min(beMonths, 999)
    });
  }

  return {
    capex: {
      hardware: totalHardware,
      launch: totalLaunch,
      integration: integrationCost,
      groundSegment,
      insurance,
      total: totalCapex
    },
    opex: {
      operations: operationsCrew,
      groundStation: groundStationOps,
      bandwidth,
      software: softwareLicenses,
      spares: spare,
      annualTotal: annualOpex
    },
    perSat: {
      hardwareCost: satHardwareCost,
      launchCost: launchCostPerSat,
      massKg: satMassKg,
      totalCost: satHardwareCost + launchCostPerSat
    },
    fleet: {
      totalGPUs,
      numLaunches,
      satsPerLaunch,
      effectiveComputeHoursPerYear: effectiveComputeHours,
      lifetimeYears,
      costPerGPUHour,
      breakEvenMonths: Math.max(0, breakEvenMonths)
    },
    sensitivity,
    comparison: Object.entries(CLOUD_COMPARISON).map(([key, cloud]) => ({
      key,
      name: cloud.name,
      gpuHourCost: cloud.gpuHourCost,
      gpuType: cloud.gpuType,
      savingsVsOrbital: ((1 - costPerGPUHour / cloud.gpuHourCost) * 100).toFixed(1)
    }))
  };
}

function formatMoney(val) {
  if (val >= 1e9) return '$' + (val / 1e9).toFixed(1) + 'B';
  if (val >= 1e6) return '$' + (val / 1e6).toFixed(1) + 'M';
  if (val >= 1e3) return '$' + (val / 1e3).toFixed(0) + 'K';
  return '$' + val.toFixed(2);
}

function clearCanvas(canvas) {
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * dpr;
  canvas.height = canvas.clientHeight * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  return ctx;
}

function drawComparisonChart(canvas, orbitalCost, comparisons) {
  const ctx = clearCanvas(canvas);
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const pad = { top: 20, right: 60, bottom: 20, left: 140 };
  const chartW = w - pad.left - pad.right;
  const chartH = h - pad.top - pad.bottom;

  const items = [
    { name: 'Orbital Compute', cost: orbitalCost, color: '#4fc3f7' },
    ...comparisons.map(c => ({ name: c.name, cost: c.gpuHourCost, color: '#8892a4' }))
  ];

  const maxCost = Math.max(...items.map(i => i.cost)) * 1.15;
  const barH = Math.min(35, (chartH / items.length) * 0.75);
  const gap = chartH / items.length;

  items.forEach((item, i) => {
    const y = pad.top + i * gap + (gap - barH) / 2;
    const barW = (item.cost / maxCost) * chartW;

    // Label
    ctx.fillStyle = '#e0e6f0';
    ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(item.name, pad.left - 12, y + barH / 2 + 4);

    // Bar
    ctx.fillStyle = item.color;
    ctx.globalAlpha = 0.7;
    ctx.beginPath();
    ctx.roundRect(pad.left, y, barW, barH, [0, 4, 4, 0]);
    ctx.fill();
    ctx.globalAlpha = 1;

    // Value
    ctx.fillStyle = '#e0e6f0';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('$' + item.cost.toFixed(2) + '/hr', pad.left + barW + 8, y + barH / 2 + 4);
  });
}

function drawCapexPieChart(canvas, capex) {
  const ctx = clearCanvas(canvas);
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const cx = w * 0.35;
  const cy = h / 2;
  const r = Math.min(cx - 20, cy - 30);

  const slices = [
    { label: 'Hardware', value: capex.hardware, color: '#4fc3f7' },
    { label: 'Launch', value: capex.launch, color: '#66bb6a' },
    { label: 'Integration', value: capex.integration, color: '#ffa726' },
    { label: 'Ground Seg.', value: capex.groundSegment, color: '#ab47bc' },
    { label: 'Insurance', value: capex.insurance, color: '#ef5350' }
  ];

  const total = slices.reduce((s, sl) => s + sl.value, 0);
  let startAngle = -Math.PI / 2;

  slices.forEach((slice, i) => {
    const sliceAngle = (slice.value / total) * Math.PI * 2;

    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, r, startAngle, startAngle + sliceAngle);
    ctx.closePath();
    ctx.fillStyle = slice.color;
    ctx.globalAlpha = 0.8;
    ctx.fill();
    ctx.globalAlpha = 1;

    startAngle += sliceAngle;
  });

  // Inner circle (donut)
  ctx.beginPath();
  ctx.arc(cx, cy, r * 0.55, 0, Math.PI * 2);
  ctx.fillStyle = '#141b2d';
  ctx.fill();

  // Center text
  ctx.fillStyle = '#e0e6f0';
  ctx.font = 'bold 14px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(formatMoney(total), cx, cy - 4);
  ctx.font = '10px Inter, sans-serif';
  ctx.fillStyle = '#8892a4';
  ctx.fillText('Total CAPEX', cx, cy + 12);

  // Legend
  const lx = w * 0.65;
  let ly = 30;
  slices.forEach(slice => {
    ctx.fillStyle = slice.color;
    ctx.fillRect(lx, ly - 6, 12, 12);
    ctx.fillStyle = '#e0e6f0';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`${slice.label}: ${formatMoney(slice.value)}`, lx + 18, ly + 4);
    ly += 22;
  });
}

function renderResults(result) {
  const { capex, opex, perSat, fleet, sensitivity, comparison } = result;

  // Summary cards
  document.getElementById('cost-capex').textContent = formatMoney(capex.total);
  document.getElementById('cost-opex').textContent = formatMoney(opex.annualTotal) + '/yr';
  document.getElementById('cost-gpu-hour').textContent = '$' + fleet.costPerGPUHour.toFixed(2);
  document.getElementById('cost-breakeven').textContent = fleet.breakEvenMonths < 900 ? fleet.breakEvenMonths.toFixed(0) + ' mo' : 'N/A';
  document.getElementById('cost-total-gpus').textContent = fleet.totalGPUs;
  document.getElementById('cost-launches').textContent = fleet.numLaunches;
  document.getElementById('cost-sat-cost').textContent = formatMoney(perSat.totalCost);
  document.getElementById('cost-sat-mass').textContent = perSat.massKg.toFixed(0) + ' kg';

  // Charts
  const compCanvas = document.getElementById('chart-comparison');
  if (compCanvas) {
    drawComparisonChart(compCanvas, fleet.costPerGPUHour, comparison);
  }

  const capexCanvas = document.getElementById('chart-capex');
  if (capexCanvas) {
    drawCapexPieChart(capexCanvas, capex);
  }

  // Sensitivity table
  const tbody = document.getElementById('sensitivity-body');
  if (tbody) {
    tbody.innerHTML = sensitivity.map(row => `
      <tr>
        <td>${row.utilization}%</td>
        <td>$${row.costPerGPUHour.toFixed(2)}</td>
        <td>${row.breakEvenMonths < 900 ? row.breakEvenMonths.toFixed(0) + ' months' : 'Never'}</td>
        <td>${row.costPerGPUHour < 3.0 ? '<span class="badge badge-success">Viable</span>' : row.costPerGPUHour < 5.0 ? '<span class="badge badge-warning">Marginal</span>' : '<span class="badge badge-danger">Too High</span>'}</td>
      </tr>
    `).join('');
  }

  // CAPEX breakdown table
  const capexBody = document.getElementById('capex-breakdown-body');
  if (capexBody) {
    capexBody.innerHTML = `
      <tr><td>Satellite Hardware</td><td>${formatMoney(capex.hardware)}</td><td>${((capex.hardware / capex.total) * 100).toFixed(1)}%</td></tr>
      <tr><td>Launch Services</td><td>${formatMoney(capex.launch)}</td><td>${((capex.launch / capex.total) * 100).toFixed(1)}%</td></tr>
      <tr><td>Integration & Testing</td><td>${formatMoney(capex.integration)}</td><td>${((capex.integration / capex.total) * 100).toFixed(1)}%</td></tr>
      <tr><td>Ground Segment</td><td>${formatMoney(capex.groundSegment)}</td><td>${((capex.groundSegment / capex.total) * 100).toFixed(1)}%</td></tr>
      <tr><td>Insurance</td><td>${formatMoney(capex.insurance)}</td><td>${((capex.insurance / capex.total) * 100).toFixed(1)}%</td></tr>
      <tr style="font-weight:700;border-top:2px solid var(--border)"><td>Total CAPEX</td><td>${formatMoney(capex.total)}</td><td>100%</td></tr>
    `;
  }
}

function update() {
  const params = {
    numSatellites: parseInt(document.getElementById('cost-sats')?.value || 12),
    gpusPerSat: parseInt(document.getElementById('cost-gpus')?.value || 8),
    solarWatts: parseInt(document.getElementById('cost-solar')?.value || 2000),
    launchVehicle: document.getElementById('cost-vehicle')?.value || 'falcon9-rideshare',
    utilization: parseInt(document.getElementById('cost-util')?.value || 60)
  };

  const result = computeCost(params);
  renderResults(result);
}

function init() {
  // Bind sliders
  document.querySelectorAll('.control-group input[type="range"]').forEach(slider => {
    const valueSpan = slider.parentElement.querySelector('.control-value');
    if (valueSpan) {
      valueSpan.textContent = slider.value;
      slider.addEventListener('input', () => {
        valueSpan.textContent = slider.value;
        update();
      });
    }
  });

  // Bind select
  const vehicleSelect = document.getElementById('cost-vehicle');
  if (vehicleSelect) {
    vehicleSelect.addEventListener('change', update);
  }

  // Initial compute
  update();
}

document.addEventListener('DOMContentLoaded', init);
