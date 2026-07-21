import importlib
import os
import threading
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
    def setUp(self):
        slack_agent._clear_channel_cache()

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

        result = slack_agent._fetch_channels_from_slack("test-token")

        self.assertEqual(result, [CHANNELS[0], CHANNELS[-1]])
        self.assertEqual(client.conversations_list.call_count, 3)
        client_factory.assert_called_once_with("test-token")
        sleep.assert_called_once_with(1)

    @patch("slack_agent._slack_token", return_value="token-a")
    @patch("slack_agent._fetch_channels_from_slack", return_value=CHANNELS)
    def test_fresh_cache_reuses_one_slack_fetch(self, fetch, _token):
        first = slack_agent._fetch_visible_channels()
        second = slack_agent._fetch_visible_channels()

        self.assertEqual(first, CHANNELS)
        self.assertEqual(second, CHANNELS)
        fetch.assert_called_once_with("token-a")

    @patch("slack_agent._fetch_channels_from_slack")
    @patch("slack_agent._slack_token", side_effect=["token-a", "token-b"])
    def test_cache_isolated_by_token_hash(self, _token, fetch):
        fetch.side_effect = lambda token: [
            {"name": token, "id": "C1", "is_private": False}
        ]

        first = slack_agent._fetch_visible_channels()
        second = slack_agent._fetch_visible_channels()

        self.assertEqual(first[0]["name"], "token-a")
        self.assertEqual(second[0]["name"], "token-b")
        self.assertEqual(fetch.call_count, 2)
        self.assertNotIn("token-a", slack_agent._CHANNEL_CACHE)
        self.assertNotIn("token-b", slack_agent._CHANNEL_CACHE)

    @patch("slack_agent._slack_token", return_value="token-a")
    @patch("slack_agent._fetch_channels_from_slack", return_value=CHANNELS)
    def test_cache_returns_defensive_copies(self, fetch, _token):
        first = slack_agent._fetch_visible_channels()
        first[0]["name"] = "mutated"

        second = slack_agent._fetch_visible_channels()

        self.assertEqual(second[0]["name"], "eval-public")
        fetch.assert_called_once()

    @patch("slack_agent._slack_token", return_value="token-a")
    @patch("slack_agent.time.monotonic", return_value=600.0)
    @patch(
        "slack_agent._fetch_channels_from_slack",
        side_effect=slack_agent.ChannelFetchError(
            "Slack API error: ratelimited",
            transient=True,
        ),
    )
    def test_stale_cache_fallback_on_slack_error(self, fetch, _clock, _token):
        key = slack_agent._cache_key("token-a")
        slack_agent._CHANNEL_CACHE[key] = slack_agent.ChannelCacheEntry(
            tuple(CHANNELS),
            fetched_at=0.0,
        )

        result = slack_agent._fetch_visible_channels()

        self.assertEqual(result, CHANNELS)
        fetch.assert_called_once_with("token-a")

    @patch("slack_agent._slack_token", return_value="token-a")
    @patch("slack_agent.time.monotonic", return_value=600.0)
    @patch(
        "slack_agent._fetch_channels_from_slack",
        side_effect=slack_agent.ChannelFetchError("Slack API error: invalid_auth"),
    )
    def test_stale_cache_not_used_for_auth_error(self, _fetch, _clock, _token):
        key = slack_agent._cache_key("token-a")
        slack_agent._CHANNEL_CACHE[key] = slack_agent.ChannelCacheEntry(
            tuple(CHANNELS),
            fetched_at=0.0,
        )

        with self.assertRaisesRegex(slack_agent.ChannelFetchError, "invalid_auth"):
            slack_agent._fetch_visible_channels()

    @patch("slack_agent._slack_token", return_value="token-a")
    @patch("slack_agent._fetch_channels_from_slack")
    def test_concurrent_misses_share_one_fetch(self, fetch, _token):
        started = threading.Event()
        release = threading.Event()

        def slow_fetch(_token_value):
            started.set()
            release.wait(timeout=2)
            return CHANNELS

        fetch.side_effect = slow_fetch
        results = []

        first = threading.Thread(
            target=lambda: results.append(slack_agent._fetch_visible_channels())
        )
        second = threading.Thread(
            target=lambda: results.append(slack_agent._fetch_visible_channels())
        )
        first.start()
        started.wait(timeout=2)
        second.start()
        release.set()
        first.join(timeout=2)
        second.join(timeout=2)

        self.assertEqual(results, [CHANNELS, CHANNELS])
        fetch.assert_called_once_with("token-a")


class GraphTests(unittest.TestCase):
    def test_all_direct_graphs_have_empty_input_schema(self):
        graphs = [
            slack_agent.all_channels_graph,
            slack_agent.public_channels_graph,
            slack_agent.private_channels_graph,
            slack_agent.eval_channels_graph,
            slack_agent.customer_channels_graph,
        ]

        for graph in graphs:
            with self.subTest(graph=graph):
                self.assertEqual(graph.get_input_jsonschema().get("properties"), {})


if __name__ == "__main__":
    unittest.main()
