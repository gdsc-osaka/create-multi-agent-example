from __future__ import annotations

from hirenest_support.brief import build_escalation_brief

REQUIRED_SECTIONS = [
    "## Case Summary",
    "## Customer Account Context",
    "## Similar Historical Tickets",
    "## Relevant Knowledge Base / Runbook References",
    "## Incident Correlation",
    "## Severity Recommendation",
    "## Escalation Decision",
    "## Draft Customer Response",
    "## Customer Response Package",
    "## Internal Escalation Note",
]


def test_case_a_invitation_delivery_brief() -> None:
    brief = build_escalation_brief(
        "Apex Robotics reports that interview invitation emails are not delivered to all "
        "candidates for a hiring event. This is a Premier support customer."
    )

    for section in REQUIRED_SECTIONS:
        assert section in brief
    assert "Apex Robotics" in brief
    assert "TCK-4101" in brief
    assert "Messaging Platform" in brief
    assert "Recommended severity: SEV2" in brief
    assert "INC-2026-0524-INVITE-DELIVERY" in brief


def test_case_b_calendar_availability_brief() -> None:
    brief = build_escalation_brief(
        "BlueWave Health says Google Calendar availability is not showing any free slots "
        "for panel interviewers. Recruiting is blocked."
    )

    assert "BlueWave Health" in brief
    assert "TCK-4207" in brief
    assert "Scheduling Integrations" in brief
    assert "Recommended severity: SEV2" in brief
    assert "free/busy" in brief


def test_case_c_careers_site_form_brief() -> None:
    brief = build_escalation_brief(
        "ClearPath Logistics reports that the application form is missing on public job pages. "
        "Candidates can see job descriptions but cannot apply."
    )

    assert "ClearPath Logistics" in brief
    assert "TCK-4314" in brief
    assert "Careers Site Platform" in brief
    assert "Recommended severity: SEV2" in brief
    assert "INC-2026-0523-CAREERS-FORM" in brief
