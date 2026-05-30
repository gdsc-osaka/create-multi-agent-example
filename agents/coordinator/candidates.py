from __future__ import annotations

import asyncio
from typing import Any

from google.adk import Agent, Context
from google.adk.tools import google_search
from google.adk.workflow import node
from pydantic import BaseModel, Field

from agents._common import model_name
from agents.coordinator.intake import STATE_TRAVEL_REQUEST, text

STATE_TRAVEL_OPTIONS = "travel_options"
STATE_RESEARCH_REPORTS = "research_reports"


class TravelOption(BaseModel):
    option_id: str = Field(description="候補を一意に識別するID。例: option_1。")
    title: str = Field(description="ユーザーに見せられる短い候補名。")
    destination: str = Field(description="主な目的地またはエリア。")
    concept: str = Field(description="旅行方針。")
    research_focus: list[str] = Field(description="調査で確認すべき観点。")
    fit_hypothesis: str = Field(description="この候補が合いそうな理由の仮説。")


class TravelOptions(BaseModel):
    options: list[TravelOption] = Field(min_length=3, max_length=5)


class ResearchReport(BaseModel):
    option_id: str
    destination_summary: str
    access: str
    estimated_cost: str
    lodging_area: str
    recommended_spots: list[str]
    food_options: list[str]
    risks: list[str]
    weather_or_season_notes: list[str]
    source_notes: list[str]
    suitability_reason: str


def dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [dump(item) for item in value]
    if isinstance(value, dict):
        return {key: dump(item) for key, item in value.items()}
    return value


strategist_agent = Agent(
    name="strategist",
    model=model_name(),
    description="旅行方針と候補地を3から5案作る。",
    output_schema=TravelOptions,
    instruction=(
        "TravelRequest をもとに、1泊2日の国内旅行候補を3から5案作ってください。"
        "詳細旅程ではなく、旅行方針、候補地、調査観点を作ります。"
        "option_id は option_1, option_2 のように安定した値にしてください。"
    ),
    mode="single_turn",
)

research_agent = Agent(
    name="research_agent",
    model=model_name(),
    description="候補ごとの旅行リサーチを行う。",
    tools=[google_search],
    instruction=(
        "あなたは国内旅行リサーチ担当です。入力に含まれる TravelOption について、"
        "google_search を使ってアクセス、費用感、宿泊エリア、観光地、食事、"
        "リスク、季節性を調べてください。"
        "option_id は入力の値を必ず維持します。"
        "source_notes に相当する確認元名や根拠メモも自然文で含めてください。"
        "重要: google_search と structured output は同時に使えないため、"
        "あなたは構造化JSONではなく調査メモを返します。"
    ),
    mode="single_turn",
)

research_report_formatter = Agent(
    name="research_report_formatter",
    model=model_name(),
    description="検索済みリサーチメモをResearchReportへ構造化する。",
    output_schema=ResearchReport,
    instruction=(
        "入力には TravelRequest、TravelOption、google_search 済みの調査メモが含まれます。"
        "調査メモを ResearchReport に構造化してください。"
        "新しい事実を追加で断定せず、不明な項目は「要確認」と書いてください。"
        "option_id は TravelOption の値を必ず維持してください。"
    ),
    mode="single_turn",
)


def store_travel_options(ctx: Context, node_input: TravelOptions) -> TravelOptions:
    ctx.state[STATE_TRAVEL_OPTIONS] = [option.model_dump() for option in node_input.options]
    return node_input


def build_research_input(request: Any, option: Any) -> str:
    return "\n\n".join(
        [
            "TravelRequest:",
            text(request),
            "TravelOption:",
            text(option),
            "Google Search Grounding / google_search を使って、最新のアクセス、費用感、"
            "宿泊エリア、観光地、食事、リスク、季節性を調査してください。"
            " option_id は必ず維持してください。",
        ]
    )


def build_research_report_input(request: Any, option: Any, research_memo: Any) -> str:
    return "\n\n".join(
        [
            "TravelRequest:",
            text(request),
            "TravelOption:",
            text(option),
            "Research memo produced by research_agent with google_search:",
            text(research_memo),
            "上記だけを根拠に ResearchReport を返してください。",
        ]
    )


def collect_research_reports(ctx: Context, node_input: Any) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    if isinstance(node_input, dict):
        values = node_input.values()
    elif isinstance(node_input, list):
        values = node_input
    else:
        values = [node_input]
    for value in values:
        data = dump(value)
        is_report = isinstance(data, dict) and data.get("option_id")
        if is_report:
            reports[data["option_id"]] = data
    ctx.state[STATE_RESEARCH_REPORTS] = reports
    return reports


@node(name="research_candidate", rerun_on_resume=True)
async def research_candidate(ctx: Context, node_input: dict[str, Any]) -> ResearchReport:
    request = ctx.state.get(STATE_TRAVEL_REQUEST, {})
    research_memo = await ctx.run_node(research_agent, build_research_input(request, node_input))
    return await ctx.run_node(
        research_report_formatter,
        build_research_report_input(request, node_input, research_memo),
    )


@node(name="travel_research_workflow", rerun_on_resume=True)
async def travel_research_workflow(ctx: Context, node_input: Any) -> dict[str, dict[str, Any]]:
    options = ctx.state.get(STATE_TRAVEL_OPTIONS)
    if not options and isinstance(node_input, TravelOptions):
        options = [option.model_dump() for option in node_input.options]
        ctx.state[STATE_TRAVEL_OPTIONS] = options

    tasks = [ctx.run_node(research_candidate, option) for option in options or []]
    reports = await asyncio.gather(*tasks)
    return collect_research_reports(ctx, reports)
