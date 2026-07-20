"""A LangChain agent that lists private Slack channels through OpenRouter.

Setup
-----
1. Create a Slack app at https://api.slack.com/apps and add an OAuth token with
   the `groups:read` scope (a User token lists the private channels *you* are in;
   a Bot token lists the private channels the *bot* has been added to).
2. Export the credentials (never hardcode them):

       export SLACK_USER_TOKEN="xoxp-..."  # or xoxb-... for a bot token
       export OPENROUTER_API_KEY="sk-or-v1-..."

   Or source the local `.env`:

       set -a && source .env && set +a

3. Invoke the agent:

       uv run python -c 'from slack_agent import agent; print(agent.invoke({"messages": [{"role": "user", "content": "List my private channels."}]}))'
"""

from __future__ import annotations

import os

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


def list_private_channels() -> str:
    """List the private Slack channels the authenticated user is a member of.

    Returns a newline-separated list of "#name (ID)" entries, or a message
    saying none were found.
    """
    client = _slack_client()
    channels: list[dict] = []
    cursor: str | None = None

    try:
        # Paginate through all results; conversations.list caps each page.
        while True:
            resp = client.conversations_list(
                types="private_channel",
                exclude_archived=True,
                limit=200,
                cursor=cursor,
            )
            channels.extend(resp.get("channels", []))
            cursor = resp.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
    except SlackApiError as e:
        # Surface the Slack error code, not the token or raw request.
        return f"Slack API error: {e.response.get('error', 'unknown_error')}"

    # A user token already scopes to channels you're in; keep the is_member
    # guard so a bot token behaves the same way.
    mine = [c for c in channels if c.get("is_member", True)]
    if not mine:
        return "You are not a member of any private channels (or the token lacks 'groups:read')."

    lines = [f"#{c['name']} ({c['id']})" for c in sorted(mine, key=lambda c: c["name"])]
    return f"Found {len(lines)} private channel(s):\n" + "\n".join(lines)


def _openrouter_api_key() -> str:
    """Read the OpenRouter credential without logging or returning it."""
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get(
        "OPEN_ROUTER_API_KEY"
    )
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Configure it as a deployment secret."
        )
    return api_key


model = init_chat_model(
    model="nvidia/nemotron-3-ultra-550b-a55b:free",
    model_provider="openai",
    base_url="https://openrouter.ai/api/v1",
    api_key=_openrouter_api_key(),
)

agent = create_agent(
    model=model,
    tools=[list_private_channels],
    system_prompt=(
        "You are a Slack assistant. When asked to list private channels, always "
        "call list_private_channels and present its result clearly. Never ask for, "
        "reveal, or repeat credentials. Do not invent channels or IDs."
    ),
)

# Keep the existing deployment export name while exposing the create_agent graph.
graph = agent
