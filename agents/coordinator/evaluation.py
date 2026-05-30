from __future__ import annotations

from typing import Any

from google.adk import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from pydantic import BaseModel, Field, model_validator

from agents._common import model_name, runtime_a2a_httpx_client, specialist_card_url
from agents.coordinator.candidates import (
    STATE_RESEARCH_REPORTS,
    STATE_TRAVEL_OPTIONS,
)
from agents.coordinator.intake import STATE_TRAVEL_REQUEST, text

STATE_REVISED_EVALUATIONS = "revised_evaluations"


class OptionEvaluation(BaseModel):
    option_id: str
    score: int = Field(ge=1, le=10)
    comment: str
    concerns: list[str] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    agent_name: str
    preferred_option_id: str
    option_evaluations: list[OptionEvaluation]

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_option_maps(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "option_evaluations" in data:
            return data

        scores = data.get("scores_by_option")
        comments = data.get("comments_by_option", {})
        concerns = data.get("concerns_by_option", {})
        if not isinstance(scores, dict):
            return data

        migrated = dict(data)
        migrated["option_evaluations"] = [
            {
                "option_id": option_id,
                "score": score,
                "comment": comments.get(option_id, ""),
                "concerns": concerns.get(option_id, []),
            }
            for option_id, score in scores.items()
        ]
        migrated.pop("scores_by_option", None)
        migrated.pop("comments_by_option", None)
        migrated.pop("concerns_by_option", None)
        return migrated

    def score_for(self, option_id: str) -> int | None:
        for evaluation in self.option_evaluations:
            if evaluation.option_id == option_id:
                return evaluation.score
        return None


class EvaluationReports(BaseModel):
    reports: list[EvaluationReport]


_remote_a2a_httpx_client = runtime_a2a_httpx_client()

comfort_agent = RemoteA2aAgent(
    name="comfort_agent",
    agent_card=specialist_card_url("COMFORT_A2A_URL", "http://localhost:8101"),
    httpx_client=_remote_a2a_httpx_client,
    description="移動負荷、休憩、宿泊快適性、疲労しにくさで候補を評価する。",
    output_schema=EvaluationReport,
    use_legacy=False,
)

risk_agent = RemoteA2aAgent(
    name="risk_agent",
    agent_card=specialist_card_url("RISK_A2A_URL", "http://localhost:8102"),
    httpx_client=_remote_a2a_httpx_client,
    description="休業、混雑、天候、予約困難、交通遅延、不確実性で候補を評価する。",
    output_schema=EvaluationReport,
    use_legacy=False,
)

experience_agent = RemoteA2aAgent(
    name="experience_agent",
    agent_card=specialist_card_url("EXPERIENCE_A2A_URL", "http://localhost:8103"),
    httpx_client=_remote_a2a_httpx_client,
    description="非日常性、記憶に残る体験、嗜好一致で候補を評価する。",
    output_schema=EvaluationReport,
    use_legacy=False,
)


def build_evaluation_instruction(ctx: ReadonlyContext) -> str:
    return "\n\n".join(
        [
            "あなたは旅行候補評価の coordinator です。",
            "TravelRequest、TravelOptions、ResearchReports を根拠に全候補を評価してください。",
            "comfort_agent、risk_agent、experience_agent には必要な観点評価を依頼できます。",
            "各 subagent は single_turn なので、依頼文には TravelRequest、TravelOptions、"
            "ResearchReports、期待する EvaluationReport 形式を含めてください。",
            "費用、費用対効果、隠れコストの budget_agent 評価はあなた自身が作成してください。",
            "subagent の評価が不足、矛盾、曖昧な場合だけ追加で依頼してください。",
            "追加依頼の必要回数はあなたが判断してください。",
            "最終出力は EvaluationReports だけです。reports には budget_agent、comfort_agent、"
            "risk_agent、experience_agent の EvaluationReport を入れてください。",
            "各 EvaluationReport は agent_name、preferred_option_id、"
            "option_evaluations を含みます。",
            "option_evaluations は全候補分の option_id、score、comment、concerns を含む配列です。",
            "最終回答は Markdown ではなく、EvaluationReports の JSON オブジェクトだけに"
            "してください。",
            "TravelRequest:",
            text(ctx.state.get(STATE_TRAVEL_REQUEST)),
            "TravelOptions:",
            text(ctx.state.get(STATE_TRAVEL_OPTIONS)),
            "ResearchReports keyed by option_id:",
            text(ctx.state.get(STATE_RESEARCH_REPORTS)),
        ]
    )


evaluation_agent = Agent(
    name="multi_agent_evaluation",
    model=model_name(),
    description="Collaboratively evaluates travel candidates with remote specialist agents.",
    instruction=build_evaluation_instruction,
    output_key=STATE_REVISED_EVALUATIONS,
    sub_agents=[comfort_agent, risk_agent, experience_agent],
    mode="single_turn",
)
