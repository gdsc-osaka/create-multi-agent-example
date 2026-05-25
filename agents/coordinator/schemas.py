from __future__ import annotations

from pydantic import BaseModel, Field


class SpecialistDirective(BaseModel):
    agent_name: str = Field(description="Specialist agent this directive is for.")
    focus: str = Field(description="What this specialist should investigate.")
    priority: str = Field(description="high, medium, or low.")


class InvestigationPlan(BaseModel):
    case_category: str = Field(description="The best current case category.")
    urgency: str = Field(description="Initial urgency or severity estimate.")
    business_impact: str = Field(description="Known or inferred business impact.")
    ready_for_investigation: bool = Field(
        description="Whether enough information exists to run parallel investigation."
    )
    clarification_questions: list[str] = Field(
        description="Questions to ask before investigation if important information is missing."
    )
    initial_hypotheses: list[str] = Field(description="Current working hypotheses.")
    specialist_directives: list[SpecialistDirective] = Field(
        description="Directives for Account Context, Ticket History, Incident Status, "
        "Knowledge Base, and Diagnostics."
    )
