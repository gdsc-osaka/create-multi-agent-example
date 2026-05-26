from __future__ import annotations

import argparse

from hirenest_support import build_escalation_brief

SAMPLE_CASES = {
    "case-a": (
        "Apex Robotics reports that interview invitation emails are not delivered to all "
        "candidates for a hiring event. This is a Premier support customer. "
        "Check prior cases, known incidents, SLA, first response, and escalation plan."
    ),
    "case-b": (
        "BlueWave Health says Google Calendar availability is not showing any free slots "
        "for panel interviewers. Recruiting is blocked. Check historical tickets and "
        "contract information, then draft a response."
    ),
    "case-c": (
        "ClearPath Logistics reports that the application form is missing on public job pages. "
        "Candidates can see job descriptions but cannot apply, and they want to know whether "
        "this is a known incident or a customer-specific configuration problem."
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic HireNest sample case.")
    parser.add_argument("case", choices=sorted(SAMPLE_CASES))
    args = parser.parse_args()
    print(build_escalation_brief(SAMPLE_CASES[args.case]))


if __name__ == "__main__":
    main()
