"""Request authentication for the LangSmith Deployment."""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from langgraph_sdk import Auth
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

SLACK_TOKEN_HEADER = b"x-slack-user-token"
LANGSMITH_KEY_HEADER = b"x-api-key"
LANGSMITH_IDENTITY_URL = "https://api.smith.langchain.com/api/v1/orgs/current/info"
AUTH_TIMEOUT_SECONDS = 5

auth = Auth()


def _header(headers: dict[Any, Any], name: bytes) -> str | None:
    """Read one header from the byte-keyed deployment header mapping."""
    value = headers.get(name) or headers.get(name.decode("ascii"))
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="strict")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _validate_langsmith_key(api_key: str) -> bool:
    """Validate one LangSmith key against a fixed first-party endpoint."""
    request = Request(
        LANGSMITH_IDENTITY_URL,
        headers={"x-api-key": api_key},
        method="GET",
    )
    try:
        with urlopen(request, timeout=AUTH_TIMEOUT_SECONDS) as response:
            return response.status == 200
    except HTTPError as error:
        if error.code in {401, 403}:
            return False
        raise


@auth.authenticate
async def authenticate(headers: dict[Any, Any]) -> Auth.types.MinimalUserDict:
    """Validate the caller's Slack token and create a stable user identity."""
    api_key = _header(headers, LANGSMITH_KEY_HEADER)
    if api_key is None:
        raise Auth.exceptions.HTTPException(
            status_code=401,
            detail="Missing x-api-key header.",
        )
    try:
        valid_langsmith_key = await asyncio.to_thread(
            _validate_langsmith_key,
            api_key,
        )
    except (HTTPError, URLError, TimeoutError) as error:
        raise Auth.exceptions.HTTPException(
            status_code=503,
            detail="LangSmith authentication is temporarily unavailable.",
        ) from error
    if not valid_langsmith_key:
        raise Auth.exceptions.HTTPException(
            status_code=401,
            detail="LangSmith authentication failed.",
        )

    token = _header(headers, SLACK_TOKEN_HEADER)
    if token is None:
        raise Auth.exceptions.HTTPException(
            status_code=401,
            detail="Missing x-slack-user-token header.",
        )

    try:
        profile = await asyncio.to_thread(WebClient(token=token).auth_test)
    except SlackApiError as error:
        raise Auth.exceptions.HTTPException(
            status_code=401,
            detail="Slack authentication failed.",
        ) from error
    except Exception as error:
        raise Auth.exceptions.HTTPException(
            status_code=503,
            detail="Slack authentication is temporarily unavailable.",
        ) from error

    team_id = profile.get("team_id")
    user_id = profile.get("user_id")
    if not team_id or not user_id:
        raise Auth.exceptions.HTTPException(
            status_code=401,
            detail="Slack authentication returned an incomplete identity.",
        )

    return {
        "identity": f"{team_id}:{user_id}",
        "slack_token": token,
    }
