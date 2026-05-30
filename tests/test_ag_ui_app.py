from google.adk.events.event import Event as AdkEvent
from google.genai import types

from agents.coordinator.ag_ui_app import (
    adk_event_text,
    latest_user_message_text,
    parse_run_agent_input,
)


def test_parse_run_agent_input_fills_protocol_defaults():
    run_input = parse_run_agent_input(
        {
            "threadId": "thread-1",
            "runId": "run-1",
            "messages": [{"id": "message-1", "role": "user", "content": "東京から温泉"}],
        }
    )

    assert run_input.thread_id == "thread-1"
    assert run_input.run_id == "run-1"
    assert run_input.state == {}
    assert run_input.tools == []
    assert run_input.context == []
    assert run_input.forwarded_props == {}


def test_latest_user_message_text_uses_last_user_message():
    run_input = parse_run_agent_input(
        {
            "threadId": "thread-1",
            "runId": "run-1",
            "messages": [
                {"id": "message-1", "role": "user", "content": "古い希望"},
                {"id": "message-2", "role": "assistant", "content": "質問"},
                {"id": "message-3", "role": "user", "content": "新しい希望"},
            ],
        }
    )

    assert latest_user_message_text(run_input) == "新しい希望"


def test_adk_event_text_prefers_text_parts():
    event = AdkEvent(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text="候補を作成しました。")],
        ),
        output={"ignored": True},
    )

    assert adk_event_text(event) == "候補を作成しました。"
