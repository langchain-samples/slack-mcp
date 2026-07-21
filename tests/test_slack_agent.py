import importlib
import json
import os
import threading
import unittest
from collections import UserDict
from types import SimpleNamespace
from unittest.mock import patch

from slack_sdk.errors import SlackApiError
from slack_sdk.web.slack_response import SlackResponse
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import Runtime, ServerInfo

os.environ.setdefault("LANGSMITH_API_GATEWAY_TOKEN", "test-gateway-token")

slack_agent = importlib.import_module("slack_agent")


CHANNELS = [
    {"name": "eval-public", "id": "C1", "is_private": False, "is_member": True},
    {"name": "eval-other", "id": "C2", "is_private": False, "is_member": False},
    {"name": "general", "id": "C3", "is_private": False, "is_member": True},
    {"name": "eval-private", "id": "G1", "is_private": True, "is_member": True},
    {
        "name": "customer-public",
        "id": "C4",
        "is_private": False,
        "is_member": False,
    },
    {
        "name": "customer-private",
        "id": "G2",
        "is_private": True,
        "is_member": True,
    },
]


def runtime(
    token="test-slack-token",
    identity="T1:U1",
    store=None,
):
    return SimpleNamespace(
        config={
            "configurable": {
                "langgraph_auth_user": {
                    "identity": identity,
                    "slack_token": token,
                },
            }
        },
        store=store,
        state={},
    )


def config(token="test-slack-token"):
    return runtime(token).config


class ChannelToolTests(unittest.TestCase):
    @patch("slack_agent._fetch_visible_channels", return_value=CHANNELS)
    def test_public_and_private_filters(self, fetch):
        public = slack_agent.list_public(runtime())
        private = slack_agent.list_private(runtime())

        self.assertIn("Found 4 public channel(s)", public)
        self.assertNotIn("#eval-private", public)
        self.assertIn("Found 2 private channel(s)", private)
        self.assertIn("#eval-private", private)
        self.assertEqual(fetch.call_args_list[0].args, ("test-slack-token",))

    @patch("slack_agent._fetch_visible_channels", return_value=CHANNELS)
    def test_listing_exposes_definitive_membership(self, _fetch):
        result = slack_agent.list_customer_channels(runtime())

        self.assertIn("[public, not-joined] #customer-public", result)
        self.assertIn("[private, joined] #customer-private", result)

    @patch("slack_agent._fetch_visible_channels", return_value=CHANNELS)
    def test_eval_filter_returns_all_visible_channels(self, _fetch):
        result = slack_agent.list_eval_channels(runtime())

        self.assertIn("Found 3 eval- channel(s)", result)
        self.assertIn("#eval-public", result)
        self.assertIn("#eval-private", result)
        self.assertIn("#eval-other", result)

    @patch("slack_agent._fetch_visible_channels", return_value=CHANNELS)
    def test_customer_filter_returns_all_visible_channels(self, _fetch):
        result = slack_agent.list_customer_channels(runtime())

        self.assertIn("Found 2 customer- channel(s)", result)
        self.assertIn("#customer-public", result)
        self.assertIn("#customer-private", result)

    @patch(
        "slack_agent._fetch_visible_channels",
        side_effect=slack_agent.ChannelFetchError("Slack API error: test"),
    )
    def test_safe_fetch_error_is_returned(self, _fetch):
        self.assertEqual(
            slack_agent.list_channels(runtime()),
            "Slack API error: test",
        )

    def test_direct_directory_graph_returns_tool_result(self):
        def directory_tool(tool_runtime):
            self.assertEqual(
                slack_agent._slack_token_from_config(tool_runtime.config),
                "request-token",
            )
            return "directory-result"

        graph = slack_agent._build_directory_graph(
            directory_tool,
            store=InMemoryStore(),
        )

        self.assertEqual(
            graph.invoke({}, config=config("request-token")),
            {"result": "directory-result"},
        )

    def test_missing_request_credentials_fail_without_secret_details(self):
        with self.assertRaisesRegex(RuntimeError, "credentials are unavailable"):
            slack_agent.list_channels(SimpleNamespace(config={}, store=None))

    def test_auth_user_mapping_shape_is_supported(self):
        request_config = {
            "configurable": {
                "langgraph_auth_user": UserDict(
                    {"identity": "T1:U1", "slack_token": "request-token"}
                )
            }
        }

        self.assertEqual(
            slack_agent._identity_from_config(request_config),
            "T1:U1",
        )
        self.assertEqual(
            slack_agent._slack_token_from_config(request_config),
            "request-token",
        )

    def test_auth_user_object_shape_is_supported(self):
        request_config = {
            "configurable": {
                "langgraph_auth_user": SimpleNamespace(
                    identity="T1:U1",
                    slack_token="request-token",
                )
            }
        }

        self.assertEqual(
            slack_agent._identity_from_config(request_config),
            "T1:U1",
        )
        self.assertEqual(
            slack_agent._slack_token_from_config(request_config),
            "request-token",
        )

    @patch("slack_agent.get_runtime")
    def test_hosted_server_user_supplies_auth_fields(self, runtime_getter):
        runtime_getter.return_value = Runtime(
            server_info=ServerInfo(
                assistant_id="assistant-id",
                graph_id="graph-id",
                user=UserDict(
                    {
                        "identity": "T1:HOSTED",
                        "slack_token": "hosted-request-token",
                    }
                ),
            )
        )

        self.assertEqual(
            slack_agent._identity_from_config({}),
            "T1:HOSTED",
        )
        self.assertEqual(
            slack_agent._slack_token_from_config({}),
            "hosted-request-token",
        )


class DurableSnapshotTests(unittest.TestCase):
    def setUp(self):
        slack_agent._clear_channel_cache()
        self.store = InMemoryStore()

    @patch("slack_agent._fetch_visible_channels", return_value=CHANNELS)
    def test_warm_snapshot_skips_slack(self, fetch):
        caller = runtime(store=self.store)
        slack_agent._write_snapshot(caller, CHANNELS)

        first = slack_agent.list_eval_channels(caller)
        second = slack_agent.list_eval_channels(caller)

        self.assertEqual(first, second)
        fetch.assert_not_called()

    @patch("slack_agent._fetch_visible_channels")
    def test_missing_deployed_snapshot_does_not_call_slack(self, fetch):
        result = slack_agent.list_channels(runtime(store=self.store))

        self.assertIn("Call refresh_channels once", result)
        fetch.assert_not_called()

    @patch("slack_agent._fetch_visible_channels")
    def test_snapshots_are_isolated_by_authenticated_identity(self, fetch):
        liam_runtime = runtime(identity="T1:LIAM", store=self.store)
        morgan_runtime = runtime(identity="T1:MORGAN", store=self.store)
        slack_agent._write_snapshot(
            liam_runtime,
            [
                {
                    "name": "eval-liam",
                    "id": "C1",
                    "is_private": False,
                    "is_member": True,
                }
            ],
        )
        slack_agent._write_snapshot(
            morgan_runtime,
            [
                {
                    "name": "eval-morgan",
                    "id": "C2",
                    "is_private": False,
                    "is_member": True,
                }
            ],
        )

        liam = slack_agent.list_eval_channels(liam_runtime)
        morgan = slack_agent.list_eval_channels(morgan_runtime)

        self.assertIn("eval-liam", liam)
        self.assertNotIn("eval-morgan", liam)
        self.assertIn("eval-morgan", morgan)
        self.assertNotIn("eval-liam", morgan)
        fetch.assert_not_called()

    @patch("slack_agent._fetch_channels_from_slack", return_value=CHANNELS)
    def test_refresh_replaces_snapshot_without_persisting_token(self, fetch):
        caller = runtime(token="never-persist-this-token", store=self.store)

        result = slack_agent.refresh_channels(caller)
        item = self.store.get(
            (slack_agent.SNAPSHOT_NAMESPACE, "T1:U1"),
            slack_agent.SNAPSHOT_STORE_KEY,
        )

        self.assertIn("Refreshed", result)
        self.assertIsNotNone(item)
        self.assertNotIn("never-persist-this-token", repr(item.value))
        fetch.assert_called_once_with("never-persist-this-token")

    def test_snapshot_is_minimized_searchable_jsonl(self):
        caller = runtime(store=self.store)
        source = [
            {
                "id": "C1",
                "name": "customer-acme",
                "is_private": False,
                "is_member": False,
                "topic": {"value": "must not persist"},
            },
            {
                "id": "G1",
                "name": "eval-acme",
                "is_private": True,
                "purpose": {"value": "must not persist"},
            },
        ]

        slack_agent._write_snapshot(caller, source)
        item = self.store.get(
            (slack_agent.SNAPSHOT_NAMESPACE, "T1:U1"),
            slack_agent.SNAPSHOT_STORE_KEY,
        )
        lines = item.value["content"].splitlines()
        names = [json.loads(line)["name"] for line in lines]

        self.assertEqual(len(lines), 2)
        self.assertTrue(all(line.startswith("{") for line in lines))
        self.assertEqual(names, sorted(names))
        self.assertNotIn("topic", item.value["content"])
        self.assertNotIn("purpose", item.value["content"])
        self.assertEqual(
            slack_agent._read_snapshot(caller)[1]["is_member"],
            True,
        )

    def test_legacy_json_snapshot_remains_readable(self):
        caller = runtime(store=self.store)
        self.store.put(
            (slack_agent.SNAPSHOT_NAMESPACE, "T1:U1"),
            slack_agent.LEGACY_SNAPSHOT_STORE_KEY,
            slack_agent.create_file_data(json.dumps(CHANNELS)),
        )

        self.assertEqual(slack_agent._read_snapshot(caller), CHANNELS)

    @patch("slack_agent._fetch_visible_channels", return_value=CHANNELS)
    def test_agent_filesystem_can_read_but_not_write_snapshot(self, _fetch):
        caller = runtime(store=self.store)
        slack_agent._write_snapshot(caller, CHANNELS)
        backend = slack_agent._agent_backend(caller)

        content = backend.read(slack_agent.SNAPSHOT_FILE_PATH)
        denied = backend.write(slack_agent.SNAPSHOT_FILE_PATH, "[]")

        self.assertIsNone(content.error)
        self.assertIn("eval-public", content.file_data["content"])
        self.assertIn("read-only", denied.error)

    @patch("slack_agent._fetch_visible_channels")
    def test_direct_mcp_graph_reads_durable_snapshot(self, fetch):
        caller = runtime(store=self.store)
        slack_agent._write_snapshot(caller, CHANNELS)
        graph = slack_agent._build_directory_graph(
            slack_agent.list_eval_channels,
            store=self.store,
        )

        result = graph.invoke({}, config=caller.config)

        self.assertIn("#eval-public", result["result"])
        fetch.assert_not_called()


class AccountMatchingTests(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryStore()
        self.caller = runtime(store=self.store)
        slack_agent._write_snapshot(
            self.caller,
            [
                {
                    "id": "C1",
                    "name": "customer-housecall-pro",
                    "is_private": False,
                    "is_member": False,
                },
                {
                    "id": "G1",
                    "name": "eval-cognite",
                    "is_private": True,
                    "is_member": True,
                },
                {
                    "id": "C2",
                    "name": "internal-oracle",
                    "is_private": False,
                    "is_member": True,
                },
                {
                    "id": "C3",
                    "name": "eval-harvey-ai",
                    "is_private": False,
                    "is_member": False,
                },
                {
                    "id": "C4",
                    "name": "customer-acme",
                    "is_private": False,
                    "is_member": False,
                },
                {
                    "id": "G2",
                    "name": "eval-acme",
                    "is_private": True,
                    "is_member": True,
                },
                {
                    "id": "C5",
                    "name": "partner-northwind",
                    "is_private": False,
                    "is_member": True,
                },
                {
                    "id": "C6",
                    "name": "general",
                    "is_private": False,
                    "is_member": True,
                },
            ],
        )

    def test_matches_accounts_with_membership_and_all_supported_prefixes(self):
        result = json.loads(
            slack_agent.match_accounts_to_channels(
                [
                    "Housecall Pro, Inc.",
                    "Cognite",
                    "Oracle",
                    "Northwind",
                    "General",
                    "Missing Co",
                ],
                self.caller,
            )
        )["accounts"]

        self.assertEqual(result[0]["status"], "matched")
        self.assertEqual(result[0]["matches"][0]["is_member"], False)
        self.assertEqual(result[1]["matches"][0]["visibility"], "private")
        self.assertEqual(result[1]["matches"][0]["is_member"], True)
        self.assertEqual(result[2]["matches"][0]["channel_name"], "internal-oracle")
        self.assertEqual(result[2]["matches"][0]["channel_type"], "internal")
        self.assertEqual(result[3]["matches"][0]["channel_type"], "partner")
        self.assertEqual(result[4]["matches"][0]["channel_type"], "unprefixed")
        self.assertEqual(result[5]["status"], "unmatched")

    def test_ambiguous_exact_and_possible_matches_are_not_guessed(self):
        result = json.loads(
            slack_agent.match_accounts_to_channels(
                ["Acme", "Harvey"], self.caller
            )
        )["accounts"]

        self.assertEqual(result[0]["status"], "ambiguous")
        self.assertEqual(len(result[0]["matches"]), 2)
        self.assertEqual(result[1]["status"], "ambiguous")
        self.assertEqual(result[1]["matches"][0]["match_type"], "possible")

    def test_account_input_is_bounded(self):
        too_many = ["account"] * (slack_agent.MAX_ACCOUNT_MATCH_INPUTS + 1)

        result = json.loads(
            slack_agent.match_accounts_to_channels(too_many, self.caller)
        )

        self.assertIn("error", result)

    def test_direct_match_graph_has_bounded_structured_input(self):
        graph = slack_agent._build_account_match_graph(store=self.store)

        result = json.loads(
            graph.invoke(
                {"accounts": ["Cognite"]},
                config=self.caller.config,
            )["result"]
        )

        self.assertEqual(result["accounts"][0]["status"], "matched")


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

    @patch("slack_agent._fetch_channels_from_slack", return_value=CHANNELS)
    def test_fresh_cache_reuses_one_slack_fetch(self, fetch):
        first = slack_agent._fetch_visible_channels("token-a")
        second = slack_agent._fetch_visible_channels("token-a")

        self.assertEqual(first, CHANNELS)
        self.assertEqual(second, CHANNELS)
        fetch.assert_called_once_with("token-a")

    @patch("slack_agent._fetch_channels_from_slack")
    def test_cache_isolated_by_token_hash(self, fetch):
        fetch.side_effect = lambda token: [
            {"name": token, "id": "C1", "is_private": False}
        ]

        first = slack_agent._fetch_visible_channels("token-a")
        second = slack_agent._fetch_visible_channels("token-b")

        self.assertEqual(first[0]["name"], "token-a")
        self.assertEqual(second[0]["name"], "token-b")
        self.assertEqual(fetch.call_count, 2)
        self.assertNotIn("token-a", slack_agent._CHANNEL_CACHE)
        self.assertNotIn("token-b", slack_agent._CHANNEL_CACHE)

    @patch("slack_agent._fetch_channels_from_slack", return_value=CHANNELS)
    def test_cache_returns_defensive_copies(self, fetch):
        first = slack_agent._fetch_visible_channels("token-a")
        first[0]["name"] = "mutated"

        second = slack_agent._fetch_visible_channels("token-a")

        self.assertEqual(second[0]["name"], "eval-public")
        fetch.assert_called_once()

    @patch("slack_agent.time.monotonic", return_value=600.0)
    @patch(
        "slack_agent._fetch_channels_from_slack",
        side_effect=slack_agent.ChannelFetchError(
            "Slack API error: ratelimited",
            transient=True,
        ),
    )
    def test_stale_cache_fallback_on_slack_error(self, fetch, _clock):
        key = slack_agent._cache_key("token-a")
        slack_agent._CHANNEL_CACHE[key] = slack_agent.ChannelCacheEntry(
            tuple(CHANNELS),
            fetched_at=0.0,
        )

        result = slack_agent._fetch_visible_channels("token-a")

        self.assertEqual(result, CHANNELS)
        fetch.assert_called_once_with("token-a")

    @patch("slack_agent.time.monotonic", return_value=600.0)
    @patch(
        "slack_agent._fetch_channels_from_slack",
        side_effect=slack_agent.ChannelFetchError("Slack API error: invalid_auth"),
    )
    def test_stale_cache_not_used_for_auth_error(self, _fetch, _clock):
        key = slack_agent._cache_key("token-a")
        slack_agent._CHANNEL_CACHE[key] = slack_agent.ChannelCacheEntry(
            tuple(CHANNELS),
            fetched_at=0.0,
        )

        with self.assertRaisesRegex(slack_agent.ChannelFetchError, "invalid_auth"):
            slack_agent._fetch_visible_channels("token-a")

    @patch("slack_agent._fetch_channels_from_slack")
    def test_concurrent_misses_share_one_fetch(self, fetch):
        started = threading.Event()
        release = threading.Event()

        def slow_fetch(_token_value):
            started.set()
            release.wait(timeout=2)
            return CHANNELS

        fetch.side_effect = slow_fetch
        results = []

        first = threading.Thread(
            target=lambda: results.append(
                slack_agent._fetch_visible_channels("token-a")
            )
        )
        second = threading.Thread(
            target=lambda: results.append(
                slack_agent._fetch_visible_channels("token-a")
            )
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
            slack_agent.refresh_channels_graph,
        ]

        for graph in graphs:
            with self.subTest(graph=graph):
                self.assertEqual(graph.get_input_jsonschema().get("properties"), {})

    def test_agent_tool_schemas_do_not_expose_runtime_or_token(self):
        tools = slack_agent.agent.nodes["tools"].bound.tools_by_name
        slack_tools = {
            "list_channels",
            "list_eval_channels",
            "list_customer_channels",
            "list_public",
            "list_private",
            "refresh_channels",
        }
        for name in slack_tools:
            with self.subTest(tool=name):
                self.assertEqual(tools[name].args, {})

        self.assertEqual(
            set(tools["match_accounts_to_channels"].args),
            {"accounts"},
        )

    def test_account_match_graph_exposes_only_accounts(self):
        schema = slack_agent.account_match_graph.get_input_jsonschema()

        self.assertEqual(set(schema["properties"]), {"accounts"})


if __name__ == "__main__":
    unittest.main()
