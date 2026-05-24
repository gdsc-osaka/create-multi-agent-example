from __future__ import annotations

import pytest


def test_adk_agent_entrypoints_import_when_dependencies_exist() -> None:
    pytest.importorskip("google.adk")

    from agents.account_context.agent import root_agent as account_agent
    from agents.coordinator.agent import root_agent as coordinator_agent
    from agents.customer_communication.agent import root_agent as communication_agent
    from agents.escalation_policy.agent import root_agent as policy_agent
    from agents.incident_status.agent import root_agent as incident_agent
    from agents.knowledge_base.agent import root_agent as kb_agent
    from agents.ticket_history.agent import root_agent as ticket_agent

    assert ticket_agent.name == "ticket_history_agent"
    assert kb_agent.name == "knowledge_base_agent"
    assert account_agent.name == "account_context_agent"
    assert incident_agent.name == "incident_status_agent"
    assert policy_agent.name == "escalation_policy_agent"
    assert communication_agent.name == "customer_communication_agent"
    assert coordinator_agent.name == "support_coordinator_agent"
    assert [agent.name for agent in coordinator_agent.sub_agents] == [
        "support_information_gathering_agent",
        "support_brief_agent",
        "customer_communication_agent",
        "support_coordinator_final_response_agent",
    ]
    information_gathering_agent = coordinator_agent.sub_agents[0]
    assert [agent.name for agent in information_gathering_agent.sub_agents] == [
        "ticket_history_agent",
        "knowledge_base_agent",
        "account_context_agent",
        "incident_status_agent",
        "escalation_policy_agent",
    ]
    assert {agent.name for agent in information_gathering_agent.sub_agents} == {
        "ticket_history_agent",
        "knowledge_base_agent",
        "account_context_agent",
        "incident_status_agent",
        "escalation_policy_agent",
    }
    assert {agent.name for agent in coordinator_agent.sub_agents} == {
        "support_information_gathering_agent",
        "support_brief_agent",
        "customer_communication_agent",
        "support_coordinator_final_response_agent",
    }
