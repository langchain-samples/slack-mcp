"""A LangChain agent that lists Slack channels through LangSmith Gateway.

Setup
-----
1. Create a Slack app at https://api.slack.com/apps and add an OAuth token with
   the `channels:read` and `groups:read` scopes. Private channels are visible
   only when the authenticated user or bot has access to them.
2. Export the credentials (never hardcode them):

       export SLACK_USER_TOKEN="xoxp-..."  # or xoxb-... for a bot token
       export LANGSMITH_API_KEY_GATEWAY="<gateway-user-service-key>"

   Or source the local `.env`:

       set -a && source .env && set +a

3. Invoke the agent:

       uv run python -c 'from slack_agent import agent; print(agent.invoke({"messages": [{"role": "user", "content": "List my private channels."}]}))'
"""

from __future__ import annotations

import os
import time

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


def _slack_client() -> WebClient:
    """Build a Slack client from SLACK_USER_TOKEN (or SLACK_TOKEN as fallback).

    The token is read from the environment only and is never logged or returned.
    """
    token = os.environ.get("SLACK_USER_TOKEN") or os.environ.get("SLACK_TOKEN")
    if not token:
        raise RuntimeError(
            "SLACK_USER_TOKEN is not set. Export a Slack OAuth token with the "
            "'groups:read' scope, e.g. `export SLACK_USER_TOKEN=xoxp-...`."
        )
    return WebClient(token=token)


def list_channels() -> str:
    """List every public and private Slack channel visible to the token.

    Returns visibility-labeled "#name (ID)" entries and summary counts.
    """
    client = _slack_client()
    channels: list[dict] = []
    cursor: str | None = None
    rate_limit_retries = 0

    # Paginate through all results and honor Slack's rate-limit response.
    while True:
        try:
            resp = client.conversations_list(
                types="public_channel,private_channel",
                exclude_archived=True,
                limit=999,
                cursor=cursor,
            )
        except SlackApiError as e:
            error = e.response.get("error", "unknown_error")
            if error != "ratelimited":
                return f"Slack API error: {error}"

            rate_limit_retries += 1
            if rate_limit_retries > 2:
                return "Slack API error: ratelimited after 2 retries"

            try:
                retry_after = int(e.response.headers.get("retry-after", "1"))
            except (TypeError, ValueError):
                retry_after = 1
            time.sleep(max(1, min(retry_after, 30)))
            continue

        rate_limit_retries = 0
        channels.extend(resp.get("channels", []))
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break

    if not channels:
        return (
            "No visible channels found (or the token lacks 'channels:read' and "
            "'groups:read')."
        )

    public_count = sum(not c.get("is_private", False) for c in channels)
    private_count = len(channels) - public_count
    ordered = sorted(
        channels,
        key=lambda c: (c.get("is_private", False), c.get("name", "")),
    )
    lines = [
        f"[{'private' if c.get('is_private', False) else 'public'}] "
        f"#{c.get('name', 'unknown')} ({c.get('id', 'unknown')})"
        for c in ordered
    ]
    return (
        f"Found {len(channels)} visible channel(s): {public_count} public, "
        f"{private_count} private.\n" + "\n".join(lines)
    )


def _gateway_api_key() -> str:
    """Read the LangSmith Gateway service key without logging it."""
    api_key = os.environ.get("LANGSMITH_API_KEY_GATEWAY") or os.environ.get(
        "LANGSMITH_API_GATEWAY_TOKEN"
    )
    if not api_key:
        raise RuntimeError(
            "LANGSMITH_API_KEY_GATEWAY is not set. Configure a gateway-user "
            "service key as a deployment secret."
        )
    return api_key


model = init_chat_model(
    model="claude-sonnet-4-6",
    model_provider="anthropic",
    base_url="https://gateway.smith.langchain.com/anthropic",
    api_key=_gateway_api_key(),
)

agent = create_agent(
    model=model,
    tools=[list_channels],
    system_prompt=(
        "You are a Slack assistant. For any question about available channels or "
        "channel membership, always call list_channels and answer from its result. "
        "Never ask for, reveal, or repeat credentials. Do not invent channels or IDs."
    ),
)

# Keep the existing deployment export name while exposing the create_agent graph.
graph = agent
