# LangSmith Gateway-backed Slack MCP agent

This project exposes a LangChain `create_agent` graph through the LangSmith
Deployment MCP endpoint. The agent uses `claude-sonnet-4-6` through LangSmith
Gateway and can call Slack to list the private channels visible to the
configured Slack identity, along with all visible public channels.

The agent has six Slack tools:

- `list_channels`: all visible public and accessible private channels.
- `list_eval_channels`: every visible public and private `eval-` channel.
- `list_customer_channels`: every visible public and private `customer-` channel.
- `list_public`: all active public channels visible to the Slack identity.
- `list_private`: all active private channels accessible to the Slack identity.
- `refresh_channels`: replace the caller's durable snapshot from Slack.

## Local setup

Create a Slack app with the `channels:read` and `groups:read` scopes. Each caller
provides their own Slack user OAuth token at request time, so results reflect
that person's Slack access.

Store only the model Gateway credential in an untracked `.env` file:

```sh
LANGSMITH_API_GATEWAY_TOKEN=<gateway-user-service-key>
```

The Gateway key must use the `gateway-user` role.
`LANGSMITH_API_GATEWAY_TOKEN` is the only supported Gateway credential name.

Invoke the agent locally:

```sh
set -a && source .env && set +a
# Invoke through the LangGraph server with x-slack-user-token; direct Python
# invocation does not run deployment authentication.
```

## LangSmith Deployment

Deploy the repository and configure `LANGSMITH_API_GATEWAY_TOKEN` as a deployment
secret. Do not store Slack user tokens in the deployment environment. The
Anthropic provider key remains in the LangSmith workspace's Gateway
configuration and is not needed by this deployment.

Fleet must send both headers on every MCP request:

```text
x-api-key: <LangSmith deployment key>
x-slack-user-token: <caller's Slack user OAuth token>
```

Custom authentication validates the LangSmith key against the LangSmith API and
the Slack token against Slack, then makes the Slack token available only as
request metadata. That authenticated Slack identity selects a private durable
filesystem namespace; Liam and Morgan therefore read different
`/channels/directory.json` files without putting either name in a caller-
controlled tool argument. Channel results reflect the caller's own visibility.
The Slack token is never written to the filesystem or placed in graph state,
model messages, tool results, or application logs.

Because this is a model-backed agent, Slack tool results are sent to
Claude through LangSmith Gateway so the model can compose its response.

Every directory operation is also exposed as a direct, argument-free MCP tool:

- `list_channels`
- `list_public`
- `list_private`
- `list_eval_channels`
- `list_customer_channels`
- `refresh_channels`

These tools bypass Claude. List operations read a durable, user-isolated Deep
Agents virtual file backed by LangGraph Store, which survives deployment
restarts and is shared across replicas. The conversational agent can read and
search this file but cannot modify it. If the file is missing, list tools return
immediately with an instruction to run `refresh_channels`; they never silently
fall through to a slow Slack scan. `refresh_channels` creates or replaces the
snapshot. The store contains channel metadata only—never either
credential. A lightweight five-minute process cache still collapses concurrent
cold misses, with a one-hour stale fallback during temporary Slack rate limits.
