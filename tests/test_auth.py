import unittest
from unittest.mock import patch

from langgraph_sdk import Auth
from slack_sdk.errors import SlackApiError
from slack_sdk.web.slack_response import SlackResponse

import auth


class AuthenticationTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_slack_header_is_rejected(self):
        with patch("auth._validate_langsmith_key", return_value=True):
            with self.assertRaises(Auth.exceptions.HTTPException) as raised:
                await auth.authenticate({b"x-api-key": b"langsmith-key"})

        self.assertEqual(raised.exception.status_code, 401)

    async def test_missing_langsmith_header_is_rejected(self):
        with self.assertRaises(Auth.exceptions.HTTPException) as raised:
            await auth.authenticate(
                {b"x-slack-user-token": b"request-slack-token"}
            )

        self.assertEqual(raised.exception.status_code, 401)

    @patch("auth._validate_langsmith_key", return_value=False)
    async def test_invalid_langsmith_key_is_rejected(self, _validate):
        with self.assertRaises(Auth.exceptions.HTTPException) as raised:
            await auth.authenticate(
                {
                    b"x-api-key": b"invalid-langsmith-key",
                    b"x-slack-user-token": b"request-slack-token",
                }
            )

        self.assertEqual(raised.exception.status_code, 401)
        self.assertNotIn("invalid-langsmith-key", str(raised.exception))

    @patch("auth._validate_langsmith_key", return_value=True)
    @patch("auth.WebClient")
    async def test_valid_tokens_create_stable_identity(
        self,
        client_factory,
        _validate,
    ):
        client_factory.return_value.auth_test.return_value = {
            "team_id": "T123",
            "user_id": "U456",
        }

        user = await auth.authenticate(
            {
                b"x-api-key": b"langsmith-key",
                b"x-slack-user-token": b"request-slack-token",
            }
        )

        self.assertEqual(user["identity"], "T123:U456")
        self.assertEqual(user["slack_token"], "request-slack-token")
        client_factory.assert_called_once_with(token="request-slack-token")

    @patch("auth._validate_langsmith_key", return_value=True)
    @patch("auth.WebClient")
    async def test_invalid_token_returns_safe_error(
        self,
        client_factory,
        _validate,
    ):
        response = SlackResponse(
            client=None,
            http_verb="POST",
            api_url="https://slack.com/api/auth.test",
            req_args={},
            data={"ok": False, "error": "invalid_auth"},
            headers={},
            status_code=200,
        )
        client_factory.return_value.auth_test.side_effect = SlackApiError(
            "secret-request-slack-token",
            response,
        )

        with self.assertRaises(Auth.exceptions.HTTPException) as raised:
            await auth.authenticate(
                {
                    b"x-api-key": b"langsmith-key",
                    b"x-slack-user-token": b"secret-request-slack-token",
                }
            )

        self.assertEqual(raised.exception.status_code, 401)
        self.assertNotIn("secret-request-slack-token", str(raised.exception))
