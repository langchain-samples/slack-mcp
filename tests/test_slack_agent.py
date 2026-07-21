import importlib
import os
import unittest
from unittest.mock import patch

from slack_sdk.errors import SlackApiError
from slack_sdk.web.slack_response import SlackResponse

os.environ.setdefault("SLACK_USER_TOKEN", "test-slack-token")
os.environ.setdefault("LANGSMITH_API_GATEWAY_TOKEN", "test-gateway-token")

slack_agent = importlib.import_module("slack_agent")


CHANNELS = [
    {"name": "eval-public", "id": "C1", "is_private": False, "is_member": True},
    {"name": "eval-other", "id": "C2", "is_private": False, "is_member": False},
    {"name": "general", "id": "C3", "is_private": False, "is_member": True},
    {"name": "eval-private", "id": "G1", "is_private": True, "is_member": True},
    {"name": "customer-public", "id": "C4", "is_private": False},
    {"name": "customer-private", "id": "G2", "is_private": True},
]


class ChannelToolTests(unittest.TestCase):
    @patch("slack_agent._fetch_visible_channels", return_value=CHANNELS)
    def test_public_and_private_filters(self, _fetch):
        public = slack_agent.list_public()
        private = slack_agent.list_private()

        self.assertIn("Found 4 public channel(s)", public)
        self.assertNotIn("#eval-private", public)
        self.assertIn("Found 2 private channel(s)", private)
        self.assertIn("#eval-private", private)

    @patch("slack_agent._fetch_visible_channels", return_value=CHANNELS)
    def test_eval_filter_returns_all_visible_channels(self, _fetch):
        result = slack_agent.list_eval_channels()

        self.assertIn("Found 3 eval- channel(s)", result)
        self.assertIn("#eval-public", result)
        self.assertIn("#eval-private", result)
        self.assertIn("#eval-other", result)

    @patch("slack_agent._fetch_visible_channels", return_value=CHANNELS)
    def test_customer_filter_returns_all_visible_channels(self, _fetch):
        result = slack_agent.list_customer_channels()

        self.assertIn("Found 2 customer- channel(s)", result)
        self.assertIn("#customer-public", result)
        self.assertIn("#customer-private", result)

    @patch(
        "slack_agent._fetch_visible_channels",
        side_effect=slack_agent.ChannelFetchError("Slack API error: test"),
    )
    def test_safe_fetch_error_is_returned(self, _fetch):
        self.assertEqual(slack_agent.list_channels(), "Slack API error: test")

    def test_direct_directory_graph_returns_tool_result(self):
        graph = slack_agent._build_directory_graph(lambda: "directory-result")

        self.assertEqual(graph.invoke({}), {"result": "directory-result"})


class FetchTests(unittest.TestCase):
    @staticmethod
    def _rate_limit_error():
        response = SlackResponse(
            client=None,
            http_verb="GET",
            api_url="https://slack.com/api/conversations.list",
            req_args={},
            data={"ok": False, "error": "ratelimited"},
            headers={"retry-after": "1"},
            status_code=429,
        )
        return SlackApiError("ratelimited", response)

    @patch("slack_agent.time.sleep")
    @patch("slack_agent._slack_client")
    def test_rate_limit_retry_and_pagination(self, client_factory, sleep):
        client = client_factory.return_value
        client.conversations_list.side_effect = [
            self._rate_limit_error(),
            {
                "channels": [CHANNELS[0]],
                "response_metadata": {"next_cursor": "next"},
            },
            {
                "channels": [CHANNELS[-1]],
                "response_metadata": {"next_cursor": ""},
            },
        ]

        result = slack_agent._fetch_visible_channels()

        self.assertEqual(result, [CHANNELS[0], CHANNELS[-1]])
        self.assertEqual(client.conversations_list.call_count, 3)
        sleep.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
