from __future__ import annotations

from agents._common import env_bool


def personalization_tools():
    """Return memory tools only when the optional Memory extra is enabled."""
    if not env_bool("TRAVEL_AGENT_USE_MEMORY"):
        return []

    from google.adk.tools.preload_memory_tool import PreloadMemoryTool

    return [PreloadMemoryTool()]


def personalization_instruction() -> str:
    if not env_bool("TRAVEL_AGENT_USE_MEMORY"):
        return ""

    return (
        "Memory に過去の旅行嗜好や制約がある場合は、今回の明示的な希望を優先しつつ、"
        "候補生成、確認質問、旅程作成の補助情報として使ってください。"
        "Memory の内容だけを根拠に、ユーザーが今回言っていない制約を断定しないでください。"
    )
