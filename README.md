# LangSmith Gateway-backed Slack MCP agent

This project exposes a LangChain `create_agent` graph through the LangSmith
Deployment MCP endpoint. The agent uses `claude-sonnet-4-6` through LangSmith
Gateway and can call Slack to list the private channels visible to the
configured Slack identity, along with all visible public channels.

The agent has four Slack tools:

- `list_channels`: all visible public and accessible private channels.
- `list_eval_channels`: only active `eval-` channels where the Slack identity is
  a member, across both public and private channels.
- `list_public`: all active public channels visible to the Slack identity.
- `list_private`: all active private channels accessible to the Slack identity.

## Local setup

Create a Slack app with the `channels:read` and `groups:read` scopes. Prefer a
bot token and add the bot only to private channels that every authorized caller
may discover. Public channels visible to the token are also returned.

Store the token in an untracked `.env` file:

```sh
SLACK_USER_TOKEN=xoxb-...
LANGSMITH_API_GATEWAY_TOKEN=<gateway-user-service-key>
```

`SLACK_USER_TOKEN` must have `channels:read` and `groups:read`. The Gateway key
must use the `gateway-user` role. `LANGSMITH_API_GATEWAY_TOKEN` is the only
supported Gateway credential name; `SLACK_TOKEN` remains a local Slack-token
fallback.

Invoke the agent locally:

```sh
set -a && source .env && set +a
uv run python -c 'from slack_agent import agent; print(agent.invoke({"messages": [{"role": "user", "content": "List my private channels."}]}))'
```

## LangSmith Deployment

Deploy the repository and configure `SLACK_USER_TOKEN` and
`LANGSMITH_API_GATEWAY_TOKEN` as deployment secrets. Do not put either token in
source control. The Anthropic provider key remains in the LangSmith workspace's
Gateway configuration and is not needed by this deployment. The graph is
automatically available as the `list_private_slack_channels` tool at the
deployment's `/mcp/` endpoint.

MCP clients authenticate to the deployment using its `x-api-key`. This key is
separate from the Slack token. Anyone authorized to call the MCP endpoint sees
all public and accessible private channels visible to the shared Slack identity.

Because this is a model-backed agent, Slack tool results are sent to
Claude through LangSmith Gateway so the model can compose its response.
