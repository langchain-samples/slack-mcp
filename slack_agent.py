"""Slack channel tools exposed through a Gateway-backed LangChain agent."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import cast

from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime
from langchain_core.runnables import RunnableConfig
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.backends.protocol import EditResult, WriteResult
from deepagents.backends.utils import create_file_data
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from typing_extensions import TypedDict

CHANNEL_TYPES = "public_channel,private_channel"
PAGE_SIZE = 999
MAX_RATE_LIMIT_RETRIES = 1
MIN_RETRY_DELAY_SECONDS = 1
MAX_RETRY_DELAY_SECONDS = 5
CACHE_TTL_SECONDS = 5 * 60
STALE_CACHE_TTL_SECONDS = 60 * 60
SNAPSHOT_NAMESPACE = "slack-channel-directory"
SNAPSHOT_FILE_PATH = "/channels/directory.json"
SNAPSHOT_STORE_KEY = "/directory.json"

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

    def __init__(self, message: str, *, transient: bool = False):
        super().__init__(message)
        self.transient = transient


class SnapshotMissingError(RuntimeError):
    """The caller has not populated their durable channel file yet."""


class ReadOnlyChannelStoreBackend(StoreBackend):
    """Expose persisted channel files to the agent without model-side writes."""

    def write(self, file_path: str, content: str) -> WriteResult:
        return WriteResult(error=f"Channel snapshot files are read-only: {file_path}")

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        return EditResult(error=f"Channel snapshot files are read-only: {file_path}")


@dataclass(frozen=True)
class ChannelCacheEntry:
    """One immutable-by-convention cached Slack directory snapshot."""

    channels: tuple[Channel, ...]
    fetched_at: float


class EmptyInput(TypedDict):
    """Direct directory graphs accept no caller-controlled input."""


class ChannelResult(TypedDict):
    """Direct directory graph output."""

    result: str


class ChannelToolState(EmptyInput, ChannelResult):
    """Internal state for a direct directory graph."""


_CACHE_LOCK = threading.Lock()
_CHANNEL_CACHE: dict[str, ChannelCacheEntry] = {}
_FETCH_LOCKS: dict[str, threading.Lock] = {}


def _auth_user_field(config: RunnableConfig, field: str) -> object:
    """Read one field from LangGraph's mapping- or object-shaped auth user."""
    user = config.get("configurable", {}).get("langgraph_auth_user")
    if isinstance(user, Mapping):
        return user.get(field)
    return getattr(user, field, None)


def _slack_token_from_config(config: RunnableConfig) -> str:
    """Read the authenticated Slack token from request-scoped metadata."""
    token = _auth_user_field(config, "slack_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("Authenticated Slack credentials are unavailable.")
    return token


def _identity_from_config(config: RunnableConfig) -> str:
    """Read the stable identity produced by request authentication."""
    identity = _auth_user_field(config, "identity")
    if not isinstance(identity, str) or not identity:
        raise RuntimeError("Authenticated Slack identity is unavailable.")
    return identity


def _snapshot_namespace(config: RunnableConfig) -> tuple[str, str]:
    """Return the user-isolated durable-store namespace."""
    return (SNAPSHOT_NAMESPACE, _identity_from_config(config))


def _slack_client(token: str) -> WebClient:
    """Build a Slack client for one request-scoped token."""
    return WebClient(token=token)


def _cache_key(token: str) -> str:
    """Derive a non-reversible cache key from a Slack token."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _copy_channels(channels: tuple[Channel, ...] | list[Channel]) -> list[Channel]:
    """Return a defensive deep copy of channel data."""
    return cast(list[Channel], deepcopy(list(channels)))


def _cached_channels(key: str, *, max_age: float) -> list[Channel] | None:
    """Return a defensive copy of a cache entry within `max_age`."""
    now = time.monotonic()
    with _CACHE_LOCK:
        entry = _CHANNEL_CACHE.get(key)
        if entry is None or now - entry.fetched_at > max_age:
            return None
    return _copy_channels(entry.channels)


def _store_channels(key: str, channels: list[Channel]) -> list[Channel]:
    """Store a defensive snapshot and prune expired entries."""
    now = time.monotonic()
    entry = ChannelCacheEntry(tuple(_copy_channels(channels)), now)
    with _CACHE_LOCK:
        expired = [
            cache_key
            for cache_key, cached in _CHANNEL_CACHE.items()
            if now - cached.fetched_at > STALE_CACHE_TTL_SECONDS
        ]
        for cache_key in expired:
            _CHANNEL_CACHE.pop(cache_key, None)
        _CHANNEL_CACHE[key] = entry
    return _copy_channels(entry.channels)


def _fetch_lock(key: str) -> threading.Lock:
    """Return the per-token lock used to collapse concurrent cache misses."""
    with _CACHE_LOCK:
        return _FETCH_LOCKS.setdefault(key, threading.Lock())


def _clear_channel_cache() -> None:
    """Clear process-local cache state; intended for tests and maintenance."""
    with _CACHE_LOCK:
        _CHANNEL_CACHE.clear()
        _FETCH_LOCKS.clear()


def _retry_delay(error: SlackApiError) -> int:
    """Return Slack's bounded Retry-After delay."""
    try:
        delay = int(error.response.headers.get("retry-after", "1"))
    except (TypeError, ValueError):
        delay = MIN_RETRY_DELAY_SECONDS
    return max(MIN_RETRY_DELAY_SECONDS, min(delay, MAX_RETRY_DELAY_SECONDS))


def _fetch_channels_from_slack(token: str) -> list[Channel]:
    """Fetch the complete visible channel directory from Slack."""
    client = _slack_client(token)
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
                    f"Slack API error: ratelimited after {MAX_RATE_LIMIT_RETRIES} retries",
                    transient=True,
                ) from error

            time.sleep(_retry_delay(error))
            continue

        retries = 0
        channels.extend(cast(list[Channel], response.get("channels", [])))
        cursor = response.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            return channels


def _fetch_visible_channels(token: str) -> list[Channel]:
    """Return a cached directory, collapsing concurrent misses per token."""
    key = _cache_key(token)

    fresh = _cached_channels(key, max_age=CACHE_TTL_SECONDS)
    if fresh is not None:
        return fresh

    with _fetch_lock(key):
        fresh = _cached_channels(key, max_age=CACHE_TTL_SECONDS)
        if fresh is not None:
            return fresh

        stale = _cached_channels(key, max_age=STALE_CACHE_TTL_SECONDS)
        try:
            channels = _fetch_channels_from_slack(token)
        except ChannelFetchError as error:
            if error.transient and stale is not None:
                return stale
            raise
        return _store_channels(key, channels)


def _read_snapshot(runtime: ToolRuntime) -> list[Channel] | None:
    """Read a durable channel snapshot without exposing credentials."""
    if runtime.store is None:
        return None
    item = runtime.store.get(
        _snapshot_namespace(runtime.config),
        SNAPSHOT_STORE_KEY,
    )
    if item is None:
        return None
    content = item.value.get("content")
    if isinstance(content, list) and all(isinstance(line, str) for line in content):
        content = "\n".join(content)
    if not isinstance(content, str):
        return None
    try:
        channels = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(channels, list) or not all(
        isinstance(channel, dict) for channel in channels
    ):
        return None
    return _copy_channels(cast(list[Channel], channels))


def _write_snapshot(runtime: ToolRuntime, channels: list[Channel]) -> None:
    """Persist only channel metadata in the caller's isolated namespace."""
    if runtime.store is None:
        return
    runtime.store.put(
        _snapshot_namespace(runtime.config),
        SNAPSHOT_STORE_KEY,
        create_file_data(
            json.dumps(_copy_channels(channels), separators=(",", ":"))
        ),
        index=False,
    )


def _directory_channels(runtime: ToolRuntime) -> list[Channel]:
    """Prefer the durable snapshot; populate it from Slack on first use."""
    snapshot = _read_snapshot(runtime)
    if snapshot is not None:
        return snapshot
    if runtime.store is not None:
        raise SnapshotMissingError(
            "No saved channel directory exists yet. Call refresh_channels once, "
            "then retry this request."
        )
    channels = _fetch_visible_channels(_slack_token_from_config(runtime.config))
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


def _has_eval_prefix(channel: Channel) -> bool:
    return channel.get("name", "").startswith("eval-")


def _has_customer_prefix(channel: Channel) -> bool:
    return channel.get("name", "").startswith("customer-")


def _is_public(channel: Channel) -> bool:
    return not channel.get("is_private", False)


def _is_private(channel: Channel) -> bool:
    return channel.get("is_private", False)


def _channel_response(
    runtime: ToolRuntime,
    predicate: Callable[[Channel], bool],
    *,
    label: str,
    empty_message: str,
) -> str:
    """Fetch, filter, and format one channel-tool response."""
    try:
        channels = _directory_channels(runtime)
    except (ChannelFetchError, SnapshotMissingError) as error:
        return str(error)

    matching = [channel for channel in channels if predicate(channel)]
    if not matching:
        return empty_message
    return _format_channels(matching, label)


def list_channels(runtime: ToolRuntime) -> str:
    """List every active public/private Slack channel visible to the token."""
    return _channel_response(
        runtime,
        _all_channels,
        label="visible channel(s)",
        empty_message=(
            "No visible channels found (or the token lacks 'channels:read' and "
            "'groups:read')."
        ),
    )


def list_eval_channels(runtime: ToolRuntime) -> str:
    """List every visible public/private Slack channel named `eval-*`."""
    return _channel_response(
        runtime,
        _has_eval_prefix,
        label="eval- channel(s)",
        empty_message="No active visible eval- channels found.",
    )


def list_customer_channels(runtime: ToolRuntime) -> str:
    """List every visible public/private Slack channel named `customer-*`."""
    return _channel_response(
        runtime,
        _has_customer_prefix,
        label="customer- channel(s)",
        empty_message="No active visible customer- channels found.",
    )


def list_public(runtime: ToolRuntime) -> str:
    """List every active public Slack channel visible to the token."""
    return _channel_response(
        runtime,
        _is_public,
        label="public channel(s)",
        empty_message="No active public channels are visible to this Slack identity.",
    )


def list_private(runtime: ToolRuntime) -> str:
    """List every active private Slack channel accessible to the token."""
    return _channel_response(
        runtime,
        _is_private,
        label="private channel(s)",
        empty_message="No active private channels are visible to this Slack identity.",
    )


def refresh_channels(runtime: ToolRuntime) -> str:
    """Refresh the caller's durable channel snapshot directly from Slack."""
    try:
        token = _slack_token_from_config(runtime.config)
        channels = _fetch_channels_from_slack(token)
        _store_channels(_cache_key(token), channels)
        _write_snapshot(runtime, channels)
    except ChannelFetchError as error:
        return str(error)
    return f"Refreshed the durable snapshot with {len(channels)} channel(s)."


def _build_directory_graph(
    tool: Callable[[ToolRuntime], str],
    *,
    store: BaseStore | None = None,
):
    """Build a direct MCP graph around one deterministic directory tool."""

    def run(
        _: EmptyInput,
        config: RunnableConfig,
        store: BaseStore,
    ) -> ChannelResult:
        runtime = ToolRuntime(
            state={},
            context=None,
            config=config,
            stream_writer=lambda _: None,
            tool_call_id=None,
            store=store,
        )
        return {"result": tool(runtime)}

    builder = StateGraph(
        ChannelToolState,
        input_schema=EmptyInput,
        output_schema=ChannelResult,
    )
    builder.add_node("run", run)
    builder.add_edge(START, "run")
    builder.add_edge("run", END)
    return builder.compile(store=store)


all_channels_graph = _build_directory_graph(list_channels)
public_channels_graph = _build_directory_graph(list_public)
private_channels_graph = _build_directory_graph(list_private)
eval_channels_graph = _build_directory_graph(list_eval_channels)
customer_channels_graph = _build_directory_graph(list_customer_channels)
refresh_channels_graph = _build_directory_graph(refresh_channels)


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


def _agent_backend(runtime: ToolRuntime) -> CompositeBackend:
    """Build a user-isolated filesystem with read-only durable channel files."""
    return CompositeBackend(
        default=StateBackend(),
        routes={
            "/channels/": ReadOnlyChannelStoreBackend(
                store=runtime.store,
                namespace=lambda _: _snapshot_namespace(runtime.config),
            )
        },
    )


agent = create_deep_agent(
    model=model,
    backend=_agent_backend,
    tools=[
        list_channels,
        list_eval_channels,
        list_customer_channels,
        list_public,
        list_private,
        refresh_channels,
    ],
    system_prompt=(
        "You are a Slack assistant. For questions about eval channels, always call "
        "list_eval_channels. For questions about customer channels, always call "
        "list_customer_channels. For public-only channel requests, call list_public. "
        "For private-only channel requests, call list_private. For general channel "
        "listing or membership questions, call list_channels. Only call "
        "refresh_channels when the user explicitly asks to refresh or update the "
        "saved directory. Answer only from tool "
        "results or the caller's /channels/directory.json file. The channel "
        "filesystem is read-only and isolated to the authenticated Slack user. "
        "Never ask for, reveal, or repeat credentials. Do not invent channels or "
        "IDs."
    ),
)

# Keep the existing deployment export name while exposing the create_agent graph.
graph = agent
