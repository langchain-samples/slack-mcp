"""Slack channel tools exposed through a Gateway-backed LangChain agent."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import TypedDict, cast

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

CHANNEL_TYPES = "public_channel,private_channel"
PAGE_SIZE = 999
MAX_RATE_LIMIT_RETRIES = 2
MIN_RETRY_DELAY_SECONDS = 1
MAX_RETRY_DELAY_SECONDS = 30

MODEL_NAME = "claude-sonnet-4-6"
MODEL_PROVIDER = "anthropic"
GATEWAY_BASE_URL = "https://gateway.smith.langchain.com/anthropic"


class Channel(TypedDict, total=False):
    """Slack channel fields used by this application."""

    id: str
    name: str
    is_member: bool
    is_private: bool


class ChannelFetchError(RuntimeError):
    """A safe, user-facing Slack channel fetch error."""


def _slack_client() -> WebClient:
    """Build a Slack client from an environment-provided token."""
    token = os.environ.get("SLACK_USER_TOKEN") or os.environ.get("SLACK_TOKEN")
    if not token:
        raise RuntimeError(
            "SLACK_USER_TOKEN is not set. Export a Slack OAuth token with the "
            "'channels:read' and 'groups:read' scopes."
        )
    return WebClient(token=token)


def _retry_delay(error: SlackApiError) -> int:
    """Return Slack's bounded Retry-After delay."""
    try:
        delay = int(error.response.headers.get("retry-after", "1"))
    except (TypeError, ValueError):
        delay = MIN_RETRY_DELAY_SECONDS
    return max(MIN_RETRY_DELAY_SECONDS, min(delay, MAX_RETRY_DELAY_SECONDS))


def _fetch_visible_channels() -> list[Channel]:
    """Fetch all visible public/private channels with bounded retries."""
    client = _slack_client()
    channels: list[Channel] = []
    cursor: str | None = None
    retries = 0

    while True:
        try:
            response = client.conversations_list(
                types=CHANNEL_TYPES,
                exclude_archived=True,
                limit=PAGE_SIZE,
                cursor=cursor,
            )
        except SlackApiError as error:
            code = error.response.get("error", "unknown_error")
            if code != "ratelimited":
                raise ChannelFetchError(f"Slack API error: {code}") from error

            retries += 1
            if retries > MAX_RATE_LIMIT_RETRIES:
                raise ChannelFetchError(
                    f"Slack API error: ratelimited after {MAX_RATE_LIMIT_RETRIES} retries"
                ) from error

            time.sleep(_retry_delay(error))
            continue

        retries = 0
        channels.extend(cast(list[Channel], response.get("channels", [])))
        cursor = response.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            return channels


def _format_channels(channels: list[Channel], label: str) -> str:
    """Format visibility-labeled channels and summary counts."""
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
        f"Found {len(channels)} {label}: {public_count} public, "
        f"{private_count} private.\n" + "\n".join(lines)
    )


def _all_channels(_: Channel) -> bool:
    return True


def _is_eval_member(channel: Channel) -> bool:
    return channel.get("name", "").startswith("eval-") and channel.get(
        "is_member", False
    )


def _is_public(channel: Channel) -> bool:
    return not channel.get("is_private", False)


def _is_private(channel: Channel) -> bool:
    return channel.get("is_private", False)


def _channel_response(
    predicate: Callable[[Channel], bool],
    *,
    label: str,
    empty_message: str,
) -> str:
    """Fetch, filter, and format one channel-tool response."""
    try:
        channels = _fetch_visible_channels()
    except ChannelFetchError as error:
        return str(error)

    matching = [channel for channel in channels if predicate(channel)]
    if not matching:
        return empty_message
    return _format_channels(matching, label)


def list_channels() -> str:
    """List every active public/private Slack channel visible to the token."""
    return _channel_response(
        _all_channels,
        label="visible channel(s)",
        empty_message=(
            "No visible channels found (or the token lacks 'channels:read' and "
            "'groups:read')."
        ),
    )


def list_eval_channels() -> str:
    """List joined public/private Slack channels whose names start with `eval-`."""
    return _channel_response(
        _is_eval_member,
        label="eval- channel(s)",
        empty_message="No active eval- channels found for this Slack identity.",
    )


def list_public() -> str:
    """List every active public Slack channel visible to the token."""
    return _channel_response(
        _is_public,
        label="public channel(s)",
        empty_message="No active public channels are visible to this Slack identity.",
    )


def list_private() -> str:
    """List every active private Slack channel accessible to the token."""
    return _channel_response(
        _is_private,
        label="private channel(s)",
        empty_message="No active private channels are visible to this Slack identity.",
    )


def _gateway_api_key() -> str:
    """Read the LangSmith Gateway service key without logging it."""
    api_key = os.environ.get("LANGSMITH_API_GATEWAY_TOKEN")
    if not api_key:
        raise RuntimeError(
            "LANGSMITH_API_GATEWAY_TOKEN is not set. Configure a gateway-user "
            "service key as a deployment secret."
        )
    return api_key


model = init_chat_model(
    model=MODEL_NAME,
    model_provider=MODEL_PROVIDER,
    base_url=GATEWAY_BASE_URL,
    api_key=_gateway_api_key(),
)

agent = create_agent(
    model=model,
    tools=[list_channels, list_eval_channels, list_public, list_private],
    system_prompt=(
        "You are a Slack assistant. For questions about eval channels, always call "
        "list_eval_channels. For public-only channel requests, call list_public. For "
        "private-only channel requests, call list_private. For general channel "
        "listing or membership questions, call list_channels. Answer only from tool "
        "results. Never ask for, reveal, or repeat credentials. Do not invent "
        "channels or IDs."
    ),
)

# Keep the existing deployment export name while exposing the create_agent graph.
graph = agent
