# Show HN Submission

## Title (80 chars max)
Show HN: Orbital compute is 50x more expensive than AWS – here's the math (open source)

## URL
https://github.com/ShipItAndPray/orbital-compute

## Text
Companies are putting GPUs in orbit (Starcloud/YC has an H100 flying since Nov 2025, Axiom Space has data center nodes on ISS, Kepler has 40 Jetson Orins in a mesh).

I built an open-source simulator covering the full stack: orbital mechanics (SGP4), power (solar+battery+eclipse), thermal (Stefan-Boltzmann radiative cooling), inter-satellite links (optical mesh routing), radiation fault injection (SEU/SAA), data pipelines, cost modeling, and a K8s scheduler extender for orbit-aware pod placement.

Key finding: in-orbit processing saves 99% of downlink bandwidth (1 TB raw imagery becomes 10 GB of classifications). But compute in orbit is 24-50x more expensive than AWS. The value prop isn't $/GPU-hour — it's processing data where it's born.

24 Python modules, 200 tests on Python 3.9-3.12, interactive web app (runs in browser, no server):

- Constellation designer: "I need 1000 GPU-hours/day, $50M budget" → optimal Walker constellation
- Cost calculator: live CAPEX/OPEX/break-even analysis
- Data pipeline visualization: the one chart that explains why orbital compute exists
- 3D globe with orbits and ground stations

Live demo: https://shipitandpray.github.io/orbital-compute/

No open-source orbital compute scheduler existed before this (IBM KubeSat is archived, KubeSpace is paper-only).

---

## Reddit (r/NewSpace) Title
I built an open-source satellite compute constellation simulator (24 modules, 200 tests, interactive web demos)

## Tweet
Open-source satellite compute simulator: orbital mechanics → power → thermal → radiation → K8s scheduling.

Key finding: in-orbit processing saves 99% of downlink bandwidth. But it's 24-50x more expensive than AWS.

24 modules, 200 tests, interactive web app.

github.com/ShipItAndPray/orbital-compute

#space #satellite #opensource #python
