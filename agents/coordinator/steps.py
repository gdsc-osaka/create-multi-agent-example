from __future__ import annotations

from typing import Any

from google.adk.agents.context import Context
from google.adk.events import RequestInput
from google.adk.events.event import Event
from google.adk.workflow import DEFAULT_ROUTE

from acmedesk_support.communication import generate_customer_response_package
from acmedesk_support.policy import recommend_escalation
from agents.coordinator.constants import (
    INVESTIGATION_AGENT_NAMES,
    ROUTE_RETRY,
    STATE_CLARIFICATION_QUESTIONS,
    STATE_CLARIFICATION_REQUEST,
    STATE_ESCALATION_POLICY,
    STATE_INVESTIGATION_PLAN,
    STATE_SYNTHESIS_BRIEF,
)
from agents.coordinator.schemas import InvestigationPlan


def _event_text(event: Event) -> str:
    if not event.content or not event.content.parts:
        return ""
    return "".join(part.text for part in event.content.parts if part.text).strip()


def _latest_text_by_author(ctx: Context, author: str) -> str:
    for event in reversed(ctx.session.events):
        if event.invocation_id != ctx.invocation_id or event.author != author:
            continue
        text = _event_text(event)
        if text:
            return text
    return ""


def _session_user_messages(ctx: Context) -> list[str]:
    messages: list[str] = []
    for event in ctx.session.events:
        if event.author != "user":
            continue
        text = _event_text(event)
        if text:
            messages.append(text)
    return messages


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return repr(value)


def _canonical_agent_name(name: str) -> str:
    for agent_name in INVESTIGATION_AGENT_NAMES:
        if agent_name in name:
            return agent_name
    return name


def _format_joined_findings(node_input: Any) -> str:
    if not isinstance(node_input, dict):
        return _stringify(node_input)

    sections: list[str] = []
    for name, value in node_input.items():
        sections.append(f"{_canonical_agent_name(name)} findings:\n{_stringify(value)}")
    return "\n\n".join(sections)


def _format_clarification_request(questions: list[str]) -> str:
    formatted_questions = "\n".join(f"- {question}" for question in questions)
    return (
        "I need a little more information before running the support investigation:\n"
        f"{formatted_questions}"
    )


def route_investigation_plan(ctx: Context, node_input: InvestigationPlan):
    ctx.state[STATE_INVESTIGATION_PLAN] = node_input.model_dump()
    if not node_input.ready_for_investigation:
        questions = [
            question.strip()
            for question in node_input.clarification_questions
            if question.strip()
        ]
        ctx.state[STATE_CLARIFICATION_QUESTIONS] = questions
        ctx.state[STATE_CLARIFICATION_REQUEST] = _format_clarification_request(questions)
        yield Event(route=ROUTE_RETRY)
        return

    yield Event(output=node_input, route=DEFAULT_ROUTE)


def request_retry_clarification(ctx: Context, node_input: Any):
    questions = ctx.state.get(STATE_CLARIFICATION_QUESTIONS, [])
    message = ctx.state.get(STATE_CLARIFICATION_REQUEST)
    if not message:
        message = _format_clarification_request([_stringify(node_input)] if node_input else [])

    yield RequestInput(
        message=message,
        payload={
            "investigation_plan": ctx.state.get(STATE_INVESTIGATION_PLAN),
            "clarification_questions": questions,
        },
        response_schema=str,
    )


def build_retry_planning_input(ctx: Context, node_input: Any) -> str:
    clarification_questions = ctx.state.get(STATE_CLARIFICATION_QUESTIONS, [])
    return "\n\n".join(
        [
            "Customer inquiry and session context:",
            "\n\n".join(_session_user_messages(ctx)),
            "Previous triage / investigation plan:",
            _stringify(ctx.state.get(STATE_INVESTIGATION_PLAN, "")),
            "Clarification questions asked:",
            "\n".join(f"- {question}" for question in clarification_questions),
            "Human clarification response:",
            _stringify(node_input),
            "Update the InvestigationPlan using the clarification response.",
        ]
    )


def build_synthesis_input(ctx: Context, node_input: Any) -> str:
    customer_messages = "\n\n".join(_session_user_messages(ctx))
    plan = _latest_text_by_author(ctx, "triage_planning_agent")
    if not plan:
        plan = _stringify(ctx.state.get(STATE_INVESTIGATION_PLAN, ""))
    findings = _format_joined_findings(node_input)
    return "\n\n".join(
        [
            "Customer inquiry and session context:",
            customer_messages,
            "Triage / Investigation Plan:",
            plan,
            "Parallel specialist findings:",
            findings,
        ]
    )


def _planned_value(ctx: Context, field_name: str, default: str = "Unknown") -> str:
    plan = ctx.state.get(STATE_INVESTIGATION_PLAN, {})
    if not isinstance(plan, dict):
        return default
    value = plan.get(field_name)
    return _stringify(value) or default


def _format_escalation_policy(policy: dict[str, Any]) -> str:
    sla = policy.get("sla") or {}
    return "\n\n".join(
        [
            "## Escalation Policy Check",
            "\n".join(
                [
                    f"- Recommended severity: {policy.get('severity', 'Unknown')}",
                    f"- Reasoning: {policy.get('severity_reason', 'Unknown')}",
                    f"- SLA response deadline: {sla.get('first_response', 'Unknown')}",
                    f"- SLA update frequency: {sla.get('update_frequency', 'Unknown')}",
                    f"- Escalate: {'Yes' if policy.get('should_escalate') else 'No'}",
                    f"- Recommended team: {policy.get('team', 'Unknown')}",
                    f"- Escalation reason: {policy.get('reason', 'Unknown')}",
                    f"- Attach: {', '.join(policy.get('attach', [])) or 'None'}",
                    "- Additional information needed: "
                    f"{', '.join(policy.get('additional_info_needed', [])) or 'None'}",
                    "- Customer-safe constraints: "
                    f"{'; '.join(policy.get('customer_safe_constraints', [])) or 'None'}",
                ]
            ),
        ]
    )


def _format_customer_communication(response_package: dict[str, Any]) -> str:
    disclosure = response_package.get("disclosure_check", {})
    return "\n\n".join(
        [
            "## Customer Communication Draft",
            "\n".join(
                [
                    f"Subject: {response_package.get('subject', 'Update on your support case')}",
                    "",
                    _stringify(response_package.get("customer_response", "")),
                ]
            ),
            "## Customer Communication Safety Check",
            "\n".join(
                [
                    "- Disclosure safe to send: "
                    f"{'Yes' if disclosure.get('safe_to_send') else 'No'}",
                    "- Omitted or softened internal details: "
                    f"{'Yes' if disclosure.get('omitted_or_softened') else 'No'}",
                    "- Requires human review: "
                    f"{'Yes' if response_package.get('requires_human_review') else 'No'}",
                    "- Human review reason: "
                    f"{', '.join(response_package.get('human_review_reason', [])) or 'None'}",
                    "- Assumptions: "
                    f"{'; '.join(response_package.get('assumptions', [])) or 'None'}",
                ]
            ),
        ]
    )


def build_final_package_input(ctx: Context, node_input: Any) -> str:
    customer_messages = "\n\n".join(_session_user_messages(ctx))
    synthesis = _stringify(node_input)
    policy = recommend_escalation("\n\n".join([customer_messages, synthesis]))
    ctx.state[STATE_SYNTHESIS_BRIEF] = synthesis
    ctx.state[STATE_ESCALATION_POLICY] = policy

    policy_text = _format_escalation_policy(policy)
    communication_source = "\n\n".join(
        [
            "# Coordinator Support Brief",
            "## Case Summary",
            "\n".join(
                [
                    f"- Inquiry summary: {customer_messages}",
                    f"- Impact scope: {_planned_value(ctx, 'business_impact')}",
                    f"- Problem category: {_planned_value(ctx, 'case_category')}",
                    f"- Initial urgency: {_planned_value(ctx, 'urgency')}",
                ]
            ),
            "## Synthesis / Hypothesis Update",
            synthesis,
            policy_text,
        ]
    )
    response_package = generate_customer_response_package(communication_source)

    return "\n\n".join(
        [
            "Output language requirement:",
            (
                "Write the final response in the same language as the Customer inquiry "
                "and session context. Apply this to section headings, summaries, the "
                "customer communication draft, clarification questions, and internal next "
                "steps. Translate or rewrite English intermediate text as needed. Keep "
                "product names, severity codes, SLA values, IDs, and team names unchanged."
            ),
            "Customer inquiry and session context:",
            customer_messages,
            "Synthesis / hypothesis update:",
            synthesis,
            "Escalation policy check:",
            policy_text,
            "Customer communication draft:",
            _format_customer_communication(response_package),
        ]
    )
