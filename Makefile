UV ?= uv
UV_RUN := $(UV) run

EVAL_SET_ID ?= travel_planning_user_sim
MEMORY_SERVICE_URI ?= memory://

.PHONY: setup lock lint run run-specialists run-coordinator run-ag-ui deploy-all web web-memory eval-create eval-add-scenarios eval-run clean

setup:
	$(UV) sync --extra dev

lock:
	$(UV) lock

lint:
	$(UV_RUN) --extra dev ruff check .

run:
	@set -e; \
	trap 'for job in $$(jobs -p); do kill "$$job" 2>/dev/null || true; done' INT TERM EXIT; \
	$(MAKE) --no-print-directory run-specialists & \
	$(MAKE) --no-print-directory run-coordinator & \
	$(MAKE) --no-print-directory web & \
	echo "Specialists, coordinator, and ADK Web are starting."; \
	wait

run-specialists:
	@set -e; \
	trap 'kill 0' INT TERM EXIT; \
	PYTHONPATH=. $(UV_RUN) uvicorn agents.comfort.agent:app --host 0.0.0.0 --port 8101 & \
	PYTHONPATH=. $(UV_RUN) uvicorn agents.risk.agent:app --host 0.0.0.0 --port 8102 & \
	PYTHONPATH=. $(UV_RUN) uvicorn agents.experience.agent:app --host 0.0.0.0 --port 8103 & \
	echo "Specialist A2A agents are running on ports 8101-8103."; \
	wait

run-coordinator:
	PYTHONPATH=. $(UV_RUN) uvicorn agents.coordinator.agent:app --host 0.0.0.0 --port 8100

run-ag-ui:
	PYTHONPATH=. $(UV_RUN) uvicorn agents.coordinator.ag_ui_app:app --host 0.0.0.0 --port 8200

deploy-all:
	./scripts/deploy_all.sh

web:
	PYTHONPATH=. $(UV_RUN) adk web agents --port 8000

web-memory:
	TRAVEL_AGENT_USE_MEMORY=true PYTHONPATH=. $(UV_RUN) adk web agents --port 8000 --memory_service_uri="$(MEMORY_SERVICE_URI)"

eval-create:
	PYTHONPATH=. $(UV_RUN) adk eval_set create agents/coordinator $(EVAL_SET_ID)

eval-add-scenarios:
	PYTHONPATH=. $(UV_RUN) adk eval_set add_eval_case agents/coordinator $(EVAL_SET_ID) --scenarios_file evals/travel_scenarios.json --session_input_file evals/session_input.json

eval-run:
	PYTHONPATH=. $(UV_RUN) adk eval agents/coordinator $(EVAL_SET_ID) --config_file_path evals/eval_config.json --print_detailed_results

clean:
	rm -rf .venv .ruff_cache .agent-runtime-temp .agent-engine-temp build dist *.egg-info
