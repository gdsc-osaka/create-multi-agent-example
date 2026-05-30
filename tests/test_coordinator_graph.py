from types import SimpleNamespace

from agents.coordinator.agent import candidate_workflow, root_agent
from agents.coordinator.recommendation import (
    ROUTE_SELECTED,
    STATE_COORDINATOR_RECOMMENDATION,
    STATE_SELECTED_OPTION_ID,
    CoordinatorRecommendation,
    RankedOption,
    request_user_selection,
    route_user_selection,
)


def edge_set(workflow):
    return {
        (edge.from_node.name, edge.route, edge.to_node.name)
        for edge in workflow.graph.edges
    }


def test_clarification_route_waits_before_reanalysis():
    edges = edge_set(root_agent)

    assert ("route_after_analysis", "clarify", "request_clarification") in edges
    assert ("request_clarification", None, "build_reanalysis_input") in edges
    assert ("build_reanalysis_input", None, "analyst") in edges

    assert ("route_after_analysis", "clarify", "build_reanalysis_input") not in edges
    assert ("route_after_analysis", "clarify", "analyst") not in edges


def test_replan_route_builds_replan_input_before_analysis():
    edges = edge_set(candidate_workflow)

    assert ("route_user_selection", "replan", "build_replan_input") in edges
    assert ("build_replan_input", None, "analyst") in edges

    assert ("route_user_selection", "replan", "analyst") not in edges


def sample_recommendation():
    return CoordinatorRecommendation(
        ranked_options=[
            RankedOption(
                option_id="option_1",
                rank=1,
                title="xxx",
                reason="first",
                cautions=[],
            ),
            RankedOption(
                option_id="option_2",
                rank=2,
                title="yyy",
                reason="second",
                cautions=[],
            ),
            RankedOption(
                option_id="option_3",
                rank=3,
                title="zzz",
                reason="third",
                cautions=[],
            ),
        ],
        comparison_summary="summary",
        conflict_resolution="notes",
        user_message="message",
    )


def test_user_selection_request_accepts_numeric_input():
    request = next(request_user_selection(SimpleNamespace(state={}), sample_recommendation()))

    assert request.response_schema == str | int


def test_route_user_selection_accepts_integer_rank():
    recommendation = sample_recommendation()
    ctx = SimpleNamespace(
        state={STATE_COORDINATOR_RECOMMENDATION: recommendation.model_dump()}
    )

    events = list(route_user_selection(ctx, 2))

    assert events[0].actions.route == ROUTE_SELECTED
    assert events[0].output == "option_2"
    assert ctx.state[STATE_SELECTED_OPTION_ID] == "option_2"
