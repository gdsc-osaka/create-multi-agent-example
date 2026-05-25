from __future__ import annotations

from google.adk import Workflow
from google.adk.workflow import DEFAULT_ROUTE

from agents._common import build_a2a_app
from agents.coordinator.constants import ROUTE_RETRY
from agents.coordinator.nodes import (
    account_context_agent,
    diagnostics_agent,
    final_package_agent,
    incident_status_agent,
    knowledge_base_agent,
    parallel_investigation_join,
    synthesis_hypothesis_agent,
    ticket_history_agent,
    triage_planning_agent,
)
from agents.coordinator.steps import (
    build_final_package_input,
    build_retry_planning_input,
    build_synthesis_input,
    request_retry_clarification,
    route_investigation_plan,
)

support_resolution_workflow = Workflow(
    name="support_case_resolution_workflow",
    description="Runs parallel investigation through final package generation.",
    edges=[
        ("START", account_context_agent, parallel_investigation_join),
        ("START", ticket_history_agent, parallel_investigation_join),
        ("START", incident_status_agent, parallel_investigation_join),
        ("START", knowledge_base_agent, parallel_investigation_join),
        ("START", diagnostics_agent, parallel_investigation_join),
        (
            parallel_investigation_join,
            build_synthesis_input,
            synthesis_hypothesis_agent,
            build_final_package_input,
            final_package_agent,
        ),
    ],
)

root_agent = Workflow(
    name="support_coordinator_agent",
    description="Runs the generic Support Case Resolution Workflow over specialist A2A agents.",
    edges=[
        ("START", triage_planning_agent, route_investigation_plan),
        (
            route_investigation_plan,
            {
                ROUTE_RETRY: request_retry_clarification,
                DEFAULT_ROUTE: support_resolution_workflow,
            },
        ),
        (request_retry_clarification, build_retry_planning_input, triage_planning_agent),
    ],
)

app = build_a2a_app(root_agent, default_port=8100)
