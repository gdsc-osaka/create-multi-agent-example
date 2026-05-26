from __future__ import annotations

from google.adk import Agent

from agents._common import build_a2a_app, model_name
from hirenest_support.agent_rag import assess_retrieval_coverage, refine_retrieval_query
from hirenest_support.agent_search import search_agent_search
from hirenest_support.tickets import get_ticket_thread, search_ticket_history


def search_similar_tickets(inquiry: str) -> dict:
    """Search similar historical tickets and return causes, resolutions, and insights."""
    return {"similar_tickets": search_ticket_history(inquiry)}


def retrieve_ticket_thread(ticket_id: str) -> dict:
    """Retrieve comments and escalation records for a specific historical ticket."""
    return get_ticket_thread(ticket_id)


def search_hirenest_agent_search(inquiry: str) -> dict:
    """Search the configured Gemini Enterprise Agent Search corpus."""
    return search_agent_search(inquiry)


def assess_ticket_history_evidence(inquiry: str, current_findings: dict) -> dict:
    """Assess whether retrieved ticket evidence is sufficient for the case."""
    return assess_retrieval_coverage(inquiry, "ticket_history", current_findings)


def refine_ticket_history_query(inquiry: str, current_findings: dict, attempt: int = 1) -> dict:
    """Return a refined ticket-history query when retrieved evidence is incomplete."""
    return refine_retrieval_query(inquiry, "ticket_history", current_findings, attempt)


root_agent = Agent(
    name="ticket_history_agent",
    model=model_name(),
    description="Searches HireNest historical support tickets and escalation cases.",
    instruction=(
        "You are the Ticket History Agent for HireNest ATS support. "
        "Use an AgentRAG loop: search_hirenest_agent_search first, then "
        "search_similar_tickets. Call assess_ticket_history_evidence on the retrieved data. "
        "If it says sufficient=false, call refine_ticket_history_query and search again with "
        "the refined query. Repeat for up to two refinement attempts before finalizing. "
        "After identifying relevant ticket IDs, use retrieve_ticket_thread for the top one or "
        "two tickets. Return the final query attempts, similar tickets, similarities, "
        "differences, past causes, resolutions, internal comments if useful, applicable "
        "insights, and any remaining missing evidence."
    ),
    tools=[
        search_hirenest_agent_search,
        search_similar_tickets,
        retrieve_ticket_thread,
        assess_ticket_history_evidence,
        refine_ticket_history_query,
    ],
)

app = build_a2a_app(root_agent, default_port=8101)
