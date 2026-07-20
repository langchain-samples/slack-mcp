# OpenRouter-backed Slack MCP agent

This project exposes a LangChain `create_agent` graph through the LangSmith
Deployment MCP endpoint. The agent uses OpenRouter's
`nvidia/nemotron-3-ultra-550b-a55b:free` model and can call Slack to list the
private channels visible to the configured Slack identity.

## Local setup

Create a Slack app with the `groups:read` scope. Prefer a bot token and add the
bot only to channels that every authorized caller may discover.

Store the token in an untracked `.env` file:

```sh
SLACK_USER_TOKEN=xoxb-...
OPENROUTER_API_KEY=sk-or-v1-...
```

Invoke the agent locally:

```sh
set -a && source .env && set +a
uv run python -c 'from slack_agent import agent; print(agent.invoke({"messages": [{"role": "user", "content": "List my private channels."}]}))'
```

## LangSmith Deployment

Deploy the repository and configure `SLACK_USER_TOKEN` and
`OPENROUTER_API_KEY` as deployment secrets. Do not put either token in source
control. The graph is automatically available as the
`list_private_slack_channels` tool at the deployment's `/mcp/` endpoint.

MCP clients authenticate to the deployment using its `x-api-key`. This key is
separate from the Slack token. Anyone authorized to call the MCP endpoint sees
the channels visible to the shared Slack identity.

Because this is a model-backed agent, Slack tool results are sent to
OpenRouter/Nvidia so the model can compose its response.
