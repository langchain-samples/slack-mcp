# LangSmith Gateway-backed Slack MCP agent

This project exposes a LangChain `create_agent` graph through the LangSmith
Deployment MCP endpoint. The agent uses `claude-sonnet-4-6` through LangSmith
Gateway and can call Slack to list the private channels visible to the
configured Slack identity.

## Local setup

Create a Slack app with the `groups:read` scope. Prefer a bot token and add the
bot only to channels that every authorized caller may discover.

Store the token in an untracked `.env` file:

```sh
SLACK_USER_TOKEN=xoxb-...
LANGSMITH_API_KEY_GATEWAY=<gateway-user-service-key>
```

Invoke the agent locally:

```sh
set -a && source .env && set +a
uv run python -c 'from slack_agent import agent; print(agent.invoke({"messages": [{"role": "user", "content": "List my private channels."}]}))'
```

## LangSmith Deployment

Deploy the repository and configure `SLACK_USER_TOKEN` and
`LANGSMITH_API_KEY_GATEWAY` as deployment secrets. Do not put either token in
source control. The Anthropic provider key remains in the LangSmith workspace's
Gateway configuration and is not needed by this deployment. The graph is
automatically available as the `list_private_slack_channels` tool at the
deployment's `/mcp/` endpoint.

MCP clients authenticate to the deployment using its `x-api-key`. This key is
separate from the Slack token. Anyone authorized to call the MCP endpoint sees
the channels visible to the shared Slack identity.

Because this is a model-backed agent, Slack tool results are sent to
Claude through LangSmith Gateway so the model can compose its response.
