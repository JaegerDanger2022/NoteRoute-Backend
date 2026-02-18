# Phase 3: Slack Web API integration


async def post_message(channel_id: str, content: str, access_token: str) -> None:
    """Post a message to a Slack channel."""
    raise NotImplementedError("Slack service not yet implemented (Phase 3)")


async def list_channels(access_token: str) -> list[dict]:
    """List accessible Slack channels for slot selection."""
    raise NotImplementedError("Slack service not yet implemented (Phase 3)")
