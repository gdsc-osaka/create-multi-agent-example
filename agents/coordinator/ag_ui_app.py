from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from ag_ui.core import (
    CustomEvent,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateSnapshotEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
)
from ag_ui.encoder import EventEncoder
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from google.adk.artifacts import InMemoryArtifactService
from google.adk.auth.credential_service.in_memory_credential_service import (
    InMemoryCredentialService,
)
from google.adk.events import RequestInput
from google.adk.events.event import Event as AdkEvent
from google.adk.memory import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel

from agents.coordinator.agent import root_agent

APP_NAME = root_agent.name or "dynamic_travel_planning_agent"
DEFAULT_USER_ID = "ag-ui-user"

session_service = InMemorySessionService()
runner = Runner(
    app_name=APP_NAME,
    agent=root_agent,
    artifact_service=InMemoryArtifactService(),
    session_service=session_service,
    memory_service=InMemoryMemoryService(),
    credential_service=InMemoryCredentialService(),
)

app = FastAPI(
    title="Dynamic Travel Planning Agent AG-UI",
    description="AG-UI endpoint for the ADK travel planning coordinator.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
async def health() -> dict[str, Any]:
    return {
        "name": APP_NAME,
        "protocol": "AG-UI",
        "endpoint": "/ag-ui",
    }


@app.post("/ag-ui")
async def run_ag_ui(request: Request) -> StreamingResponse:
    payload = await request.json()
    run_input = parse_run_agent_input(payload)
    accept = request.headers.get("accept")
    encoder = EventEncoder(accept=accept)

    return StreamingResponse(
        encode_events(run_input, encoder),
        media_type=encoder.get_content_type(),
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def parse_run_agent_input(payload: dict[str, Any]) -> RunAgentInput:
    normalized = {
        "threadId": payload.get("threadId") or payload.get("thread_id") or str(uuid4()),
        "runId": payload.get("runId") or payload.get("run_id") or str(uuid4()),
        "parentRunId": payload.get("parentRunId") or payload.get("parent_run_id"),
        "state": payload.get("state") or {},
        "messages": payload.get("messages") or [],
        "tools": payload.get("tools") or [],
        "context": payload.get("context") or [],
        "forwardedProps": payload.get("forwardedProps") or payload.get("forwarded_props") or {},
    }
    return RunAgentInput.model_validate(normalized)


async def encode_events(run_input: RunAgentInput, encoder: EventEncoder) -> AsyncIterator[str]:
    try:
        async for event in stream_ag_ui_events(run_input):
            yield encoder.encode(event)
    except Exception as exc:
        yield encoder.encode(
            RunErrorEvent(
                message=str(exc),
                code=exc.__class__.__name__,
            )
        )


async def stream_ag_ui_events(run_input: RunAgentInput) -> AsyncIterator[Any]:
    thread_id = run_input.thread_id
    run_id = run_input.run_id
    user_id = user_id_from_input(run_input)

    yield RunStartedEvent(thread_id=thread_id, run_id=run_id, input=run_input)
    await ensure_session(user_id=user_id, session_id=thread_id, state=run_input.state)
    yield StateSnapshotEvent(snapshot=jsonable(run_input.state))

    user_text = latest_user_message_text(run_input)
    if not user_text:
        yield RunErrorEvent(
            message="AG-UI request does not include a user message.",
            code="NO_INPUT",
        )
        return

    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_text)],
    )
    final_output: Any = None

    async for adk_event in runner.run_async(
        user_id=user_id,
        session_id=thread_id,
        invocation_id=run_id,
        new_message=message,
    ):
        async for ag_ui_event in adk_event_to_ag_ui(adk_event):
            if isinstance(ag_ui_event, (TextMessageContentEvent, CustomEvent)):
                final_output = getattr(ag_ui_event, "delta", None) or getattr(
                    ag_ui_event, "value", None
                )
            yield ag_ui_event

    session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=thread_id,
    )
    yield StateSnapshotEvent(snapshot=jsonable(getattr(session, "state", {}) if session else {}))
    yield RunFinishedEvent(thread_id=thread_id, run_id=run_id, result=jsonable(final_output))


async def ensure_session(*, user_id: str, session_id: str, state: Any) -> None:
    session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )
    if session:
        return

    await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
        state=state if isinstance(state, dict) else {},
    )


async def adk_event_to_ag_ui(adk_event: AdkEvent) -> AsyncIterator[Any]:
    request_input = request_input_from_event(adk_event)
    if request_input:
        message = request_input.message or "追加の入力が必要です。"
        message_id = adk_event.id or request_input.interrupt_id or str(uuid4())
        yield TextMessageStartEvent(message_id=message_id, role="assistant")
        yield TextMessageContentEvent(message_id=message_id, delta=message)
        yield TextMessageEndEvent(message_id=message_id)
        yield CustomEvent(
            name="adk.request_input",
            value={
                "interruptId": request_input.interrupt_id,
                "message": request_input.message,
                "payload": jsonable(request_input.payload),
            },
        )
        return

    event_text = adk_event_text(adk_event)
    if event_text:
        message_id = adk_event.id or str(uuid4())
        yield TextMessageStartEvent(message_id=message_id, role="assistant")
        yield TextMessageContentEvent(message_id=message_id, delta=event_text)
        yield TextMessageEndEvent(message_id=message_id)


def request_input_from_event(adk_event: AdkEvent) -> RequestInput | None:
    output = adk_event.output
    if isinstance(output, RequestInput):
        return output
    if isinstance(output, list):
        return next((item for item in output if isinstance(item, RequestInput)), None)
    return None


def adk_event_text(adk_event: AdkEvent) -> str:
    content_text = content_to_text(adk_event.content)
    if content_text:
        return content_text

    output = adk_event.output
    if output is None or isinstance(output, RequestInput):
        return ""
    if isinstance(output, str):
        return output.strip()
    return json.dumps(jsonable(output), ensure_ascii=False, indent=2)


def content_to_text(content: Any) -> str:
    if not content or not getattr(content, "parts", None):
        return ""
    return "".join(part.text for part in content.parts if getattr(part, "text", None)).strip()


def latest_user_message_text(run_input: RunAgentInput) -> str:
    for message in reversed(run_input.messages):
        if message.role != "user":
            continue
        content = message.content
        if isinstance(content, str):
            return content.strip()
        text_parts = [
            part.text
            for part in content
            if getattr(part, "type", None) == "text" and getattr(part, "text", None)
        ]
        return "\n".join(text_parts).strip()
    return ""


def user_id_from_input(run_input: RunAgentInput) -> str:
    forwarded_props = run_input.forwarded_props
    if isinstance(forwarded_props, dict):
        user_id = forwarded_props.get("userId") or forwarded_props.get("user_id")
        if user_id:
            return str(user_id)
    return DEFAULT_USER_ID


def jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True, exclude_none=True)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, set):
        return [jsonable(item) for item in value]
    return value
