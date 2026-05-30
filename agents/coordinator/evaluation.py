from __future__ import annotations

import json
from typing import Any

from google.adk import Workflow
from google.adk.agents.context import Context
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.events.event import Event
from google.adk.workflow import FunctionNode, JoinNode

from agents._common import remote_agent_card_url, runtime_a2a_httpx_client
from agents.coordinator.candidates import (
    STATE_RESEARCH_REPORTS,
    STATE_TRAVEL_OPTIONS,
)
from agents.coordinator.evaluation_models import (
    EvaluationReport,
    EvaluationReports,
    OptionEvaluation,
)
from agents.coordinator.intake import STATE_TRAVEL_REQUEST
from agents.coordinator.utils import text

__all__ = [
    "EvaluationReport",
    "EvaluationReports",
    "OptionEvaluation",
]

ROUTE_DISCUSS = "discuss"
ROUTE_EVALUATION_COMPLETE = "evaluation_complete"

STATE_REVISED_EVALUATIONS = "revised_evaluations"
STATE_EVALUATION_DISCUSSION_ROUNDS = "evaluation_discussion_rounds"
STATE_EVALUATION_DISCUSSION_HISTORY = "evaluation_discussion_history"

EVALUATION_AGENT_MODEL = "gemini-2.5-flash"


_remote_a2a_httpx_client = runtime_a2a_httpx_client()

comfort_agent = RemoteA2aAgent(
    name="comfort_agent",
    agent_card=remote_agent_card_url("COMFORT_A2A_URL", "http://localhost:8101"),
    httpx_client=_remote_a2a_httpx_client,
    description="移動負荷、休憩、宿泊快適性、疲労しにくさで候補を評価する。",
    output_schema=EvaluationReport,
    use_legacy=False,
)

risk_agent = RemoteA2aAgent(
    name="risk_agent",
    agent_card=remote_agent_card_url("RISK_A2A_URL", "http://localhost:8102"),
    httpx_client=_remote_a2a_httpx_client,
    description="休業、混雑、天候、予約困難、交通遅延、不確実性で候補を評価する。",
    output_schema=EvaluationReport,
    use_legacy=False,
)

experience_agent = RemoteA2aAgent(
    name="experience_agent",
    agent_card=remote_agent_card_url("EXPERIENCE_A2A_URL", "http://localhost:8103"),
    httpx_client=_remote_a2a_httpx_client,
    description="非日常性、記憶に残る体験、嗜好一致で候補を評価する。",
    output_schema=EvaluationReport,
    use_legacy=False,
)


def initialize_evaluation_discussion(ctx: Context, node_input: Any) -> Any:
    ctx.state[STATE_EVALUATION_DISCUSSION_ROUNDS] = 0
    ctx.state[STATE_EVALUATION_DISCUSSION_HISTORY] = []
    ctx.state[STATE_REVISED_EVALUATIONS] = {"reports": []}
    return node_input


initialize_evaluation_discussion_node = FunctionNode(
    name="initialize_evaluation_discussion",
    func=initialize_evaluation_discussion,
    parameter_binding="state",
)


def parse_evaluation_report(value: Any) -> EvaluationReport | None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("```json"):
            stripped = stripped.removeprefix("```json").removesuffix("```").strip()
        elif stripped.startswith("```"):
            stripped = stripped.removeprefix("```").removesuffix("```").strip()
        if not stripped:
            return None
        value = json.loads(stripped)

    if isinstance(value, EvaluationReport):
        return value
    if isinstance(value, dict):
        return EvaluationReport.model_validate(value)
    return None


def parse_evaluation_reports(value: Any) -> list[EvaluationReport]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("```json"):
            stripped = stripped.removeprefix("```json").removesuffix("```").strip()
        elif stripped.startswith("```"):
            stripped = stripped.removeprefix("```").removesuffix("```").strip()
        if not stripped:
            return []
        value = json.loads(stripped)

    if isinstance(value, EvaluationReports):
        return value.reports
    if isinstance(value, dict):
        return EvaluationReports.model_validate(value).reports
    return [EvaluationReport.model_validate(item) for item in value or []]


def evaluation_discussion_is_complete(ctx: Context) -> bool:
    rounds = int(ctx.state.get(STATE_EVALUATION_DISCUSSION_ROUNDS) or 0)
    if rounds < 2:
        return False

    option_ids = {
        item.get("option_id")
        for item in ctx.state.get(STATE_TRAVEL_OPTIONS, [])
        if isinstance(item, dict) and item.get("option_id")
    }
    reports = parse_evaluation_reports(ctx.state.get(STATE_REVISED_EVALUATIONS))
    reports_by_agent = {report.agent_name: report for report in reports}
    required_agents = {"comfort_agent", "risk_agent", "experience_agent"}
    if not required_agents.issubset(reports_by_agent):
        return False

    for agent_name in required_agents:
        covered = {
            evaluation.option_id
            for evaluation in reports_by_agent[agent_name].option_evaluations
        }
        if option_ids and not option_ids.issubset(covered):
            return False
    return True


def route_evaluation_discussion(ctx: Context, node_input: Any):
    route = ROUTE_EVALUATION_COMPLETE if evaluation_discussion_is_complete(ctx) else ROUTE_DISCUSS
    yield Event(route=route, output=node_input)


def build_evaluation_instruction(ctx: Context | ReadonlyContext, node_input: Any = None) -> str:
    discussion_round = int(ctx.state.get(STATE_EVALUATION_DISCUSSION_ROUNDS) or 0) + 1
    return "\n\n".join(
        [
            f"これは評価議論ラウンド {discussion_round} です。",
            "TravelRequest、TravelOptions、ResearchReports を根拠に全候補を評価してください。",
            "前回までの評価がある場合は、他観点との矛盾、見落とし、不足を踏まえて更新してください。",
            "あなた自身の観点だけを担当し、agent_name には自分の agent 名を入れてください。",
            "最終出力は EvaluationReport の JSON オブジェクトだけです。",
            "EvaluationReport は agent_name、preferred_option_id、option_evaluations を含みます。",
            "option_evaluations は全候補分の option_id、score、comment、concerns を含む配列です。",
            "Markdown ではなく JSON だけを返してください。",
            "TravelRequest:",
            text(ctx.state.get(STATE_TRAVEL_REQUEST)),
            "TravelOptions:",
            text(ctx.state.get(STATE_TRAVEL_OPTIONS)),
            "ResearchReports keyed by option_id:",
            text(ctx.state.get(STATE_RESEARCH_REPORTS)),
            "Previous EvaluationReports:",
            text(ctx.state.get(STATE_REVISED_EVALUATIONS)),
            "Previous discussion history:",
            text(ctx.state.get(STATE_EVALUATION_DISCUSSION_HISTORY)),
        ]
    )


build_evaluation_input_node = FunctionNode(
    name="build_evaluation_input",
    func=build_evaluation_instruction,
    parameter_binding="state",
)

evaluation_join = JoinNode(name="evaluation_join")


def collect_evaluation_round(ctx: Context, node_input: dict[str, Any]) -> EvaluationReports:
    existing_reports = {
        report.agent_name: report
        for report in parse_evaluation_reports(ctx.state.get(STATE_REVISED_EVALUATIONS))
    }
    round_reports: list[EvaluationReport] = []
    for agent_name in ("comfort_agent", "risk_agent", "experience_agent"):
        report = parse_evaluation_report(node_input.get(agent_name))
        if report is None:
            continue
        existing_reports[report.agent_name] = report
        round_reports.append(report)

    reports = EvaluationReports(reports=list(existing_reports.values()))
    ctx.state[STATE_REVISED_EVALUATIONS] = reports.model_dump()
    ctx.state[STATE_EVALUATION_DISCUSSION_ROUNDS] = (
        int(ctx.state.get(STATE_EVALUATION_DISCUSSION_ROUNDS) or 0) + 1
    )
    history = list(ctx.state.get(STATE_EVALUATION_DISCUSSION_HISTORY) or [])
    history.append(
        {
            "round": ctx.state[STATE_EVALUATION_DISCUSSION_ROUNDS],
            "reports": [report.model_dump() for report in round_reports],
        }
    )
    ctx.state[STATE_EVALUATION_DISCUSSION_HISTORY] = history
    return reports


collect_evaluation_round_node = FunctionNode(
    name="collect_evaluation_round",
    func=collect_evaluation_round,
    parameter_binding="state",
)


def finalize_evaluation_discussion(ctx: Context, node_input: Any) -> EvaluationReports:
    return EvaluationReports(
        reports=parse_evaluation_reports(ctx.state.get(STATE_REVISED_EVALUATIONS))
    )


finalize_evaluation_discussion_node = FunctionNode(
    name="finalize_evaluation_discussion",
    func=finalize_evaluation_discussion,
    parameter_binding="state",
)


evaluation_agent = Workflow(
    name="multi_agent_evaluation",
    description=(
        "Runs specialist evaluation agents in parallel until at least two complete "
        "discussion rounds are available, then returns evaluation reports."
    ),
    edges=[
        (
            "START",
            initialize_evaluation_discussion_node,
            route_evaluation_discussion,
            {
                ROUTE_DISCUSS: build_evaluation_input_node,
                ROUTE_EVALUATION_COMPLETE: finalize_evaluation_discussion_node,
            },
        ),
        (
            build_evaluation_input_node,
            (comfort_agent, risk_agent, experience_agent),
            evaluation_join,
            collect_evaluation_round_node,
            route_evaluation_discussion,
        ),
    ],
)
