from __future__ import annotations

import os
from typing import Any

from google.adk import Agent, Context
from google.adk.workflow import node

from agents.coordinator.illustrator import STATE_ILLUSTRATOR_PROMPT
from agents.coordinator.planner import STATE_ITINERARY_MARKDOWN
from agents.coordinator.utils import text

STATE_AGENTMAIL_RESULT = "agentmail_result"

AGENTMAIL_AGENT_MODEL = "gemini-3.5-flash"
DEFAULT_ITINERARY_EMAIL_TO = "traveler@example.com"


def agentmail_enabled() -> bool:
    api_key = os.getenv("AGENTMAIL_API_KEY", "")
    return bool(api_key and api_key != "YOUR_AGENTMAIL_API_KEY")


def _agentmail_tools():
    if not agentmail_enabled():
        return []

    from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
    from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
    from mcp import StdioServerParameters

    return [
        McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command="npx",
                    args=[
                        "-y",
                        "agentmail-mcp",
                        "--tools",
                        "create_inbox,list_inboxes,send_message",
                    ],
                    env={
                        "AGENTMAIL_API_KEY": os.environ["AGENTMAIL_API_KEY"],
                    },
                ),
                timeout=30,
            ),
        )
    ]


agentmail_agent = Agent(
    name="agentmail_sender",
    model=AGENTMAIL_AGENT_MODEL,
    description="旅行プランと旅しおり画像情報をメールで送る。",
    instruction=(
        "あなたは旅行プラン送信担当です。AgentMail の MCP tool が使える場合は、"
        "既存 inbox を確認し、適切な inbox がなければ作成してから send_message で送信してください。"
        "件名は旅行先と日数が分かる短い日本語にしてください。"
        "本文には旅行プラン Markdown、旅しおり画像生成 prompt、"
        "画像生成結果への言及を含めてください。"
        "送信できた場合は宛先、件名、message_id または thread_id を簡潔に返してください。"
        "tool が使えない場合は送信用ドラフトだけを返し、送信済みとは書かないでください。"
    ),
    tools=_agentmail_tools(),
    mode="single_turn",
)


@node(name="maybe_send_itinerary_email", rerun_on_resume=True)
async def maybe_send_itinerary_email(ctx: Context, node_input: Any) -> Any:
    if not agentmail_enabled():
        return node_input

    result = await ctx.run_node(agentmail_agent, build_agentmail_input(ctx, node_input))
    ctx.state[STATE_AGENTMAIL_RESULT] = text(result)
    return result


def build_agentmail_input(ctx: Context, illustrator_output: Any) -> str:
    recipient = os.getenv("TRAVEL_ITINERARY_EMAIL_TO", DEFAULT_ITINERARY_EMAIL_TO)
    return "\n\n".join(
        [
            f"送信先: {recipient}",
            "旅行プラン Markdown:",
            text(ctx.state.get(STATE_ITINERARY_MARKDOWN)),
            "旅しおり画像生成 prompt:",
            text(ctx.state.get(STATE_ILLUSTRATOR_PROMPT)),
            "画像生成結果:",
            text(illustrator_output),
        ]
    )
