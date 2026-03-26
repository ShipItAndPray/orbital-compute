.PHONY: test demo sim dashboard cost pipeline designer reliability standards formats help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

test:  ## Run all 222+ tests
	python3 -m unittest discover tests -v

demo:  ## Full-stack demo (all subsystems)
	python3 demo.py

sim:  ## Quick simulation (6 sats, 6h, 20 jobs)
	python3 run_sim.py

dashboard:  ## Launch web dashboard
	python3 dashboard.py

cost:  ## Cost analysis
	python3 -m orbital_compute.cost_model

pipeline:  ## Data pipeline comparison (99% bandwidth savings)
	python3 -m orbital_compute.data_pipeline

designer:  ## Auto-design a constellation
	python3 -m orbital_compute.designer

reliability:  ## SLA & availability analysis
	python3 -m orbital_compute.reliability

standards:  ## ECSS compliance check
	python3 -m orbital_compute.standards

formats:  ## Industry format export (CCSDS, STK)
	python3 -m orbital_compute.formats

k8s:  ## K8s scheduler extender demo
	python3 -m orbital_compute.k8s_scheduler

serve:  ## Serve web app locally
	cd docs && python3 -m http.server 3000

install:  ## Install dependencies
	pip install -r requirements.txt
