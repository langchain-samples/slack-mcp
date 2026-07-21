# LangSmith Gateway-backed Slack MCP agent

This project exposes a Deep Agent graph through the LangSmith Deployment MCP
endpoint. The agent uses `claude-sonnet-4-6` through LangSmith Gateway and can
list the private channels visible to the authenticated Slack identity, along
with all visible public channels.

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

## Fleet custom MCP setup

Each Fleet user must configure the MCP with their own Slack user token. Do not
share one Fleet MCP credential profile between users: Liam's profile must contain
Liam's Slack token, Morgan's must contain Morgan's, and so on.

### Prerequisites

Obtain both credentials before starting:

1. A valid LangSmith API key authorized to access the deployment.
2. The current user's Slack user OAuth token with `channels:read` and
   `groups:read` scopes.

Keep both values in an approved secret manager. Do not paste them into source
control, chat messages, logs, or shared documentation.

### Add the server in Fleet

1. Open Fleet's MCP settings and select **Add custom MCP**.
2. Enter a user-specific name, such as `MCP-Slack-Channels-Liam`.
3. Enter this exact URL, with no trailing slash:

   ```text
   https://slack-private-channel-read-ffbdd8bf080155209516781f051ac271.us.langgraph.app/mcp
   ```

4. Select **Static Headers**. Do not select either OAuth option.
5. Add these two headers:

   | Header | Value |
   | --- | --- |
   | `x-api-key` | The user's valid LangSmith API key |
   | `x-slack-user-token` | The user's Slack user OAuth token |

6. Save the server. Fleet sends both headers automatically on every MCP
   request.

If Fleet says the URL already exists, edit the existing MCP entry for that user
instead of creating a duplicate. Replace its authentication mode and headers
with the values above.

### Initialize and use the channel directory

The first list request does not perform a slow Slack scan automatically. Ask the
agent to call `refresh_channels` once to create the authenticated user's durable
`/channels/directory.json` file. The initial refresh can take approximately
15–20 seconds for a large workspace.

After that refresh, normal tools read the durable file and typically return in
under a second:

- `list_eval_channels`
- `list_customer_channels`
- `list_public`
- `list_private`
- `list_channels`

Run `refresh_channels` again whenever the user wants the saved directory updated.
Refreshing one user never changes another user's file.

Suggested verification requests:

```text
Refresh my Slack channel directory.
List every eval- channel visible to me.
List my private Slack channels.
```

### Troubleshooting

| Error | Resolution |
| --- | --- |
| `MCP OAuth discovery failed` or dynamic client registration error | Reconfigure the server to use **Static Headers**, not OAuth. |
| `Authentication failed` | Verify both headers, confirm the LangSmith key is active, and confirm the Slack token is a valid user OAuth token. |
| `No saved channel directory exists yet` | Call `refresh_channels` once, then retry the list request. |
| Fleet reports that the URL already exists | Edit the user's existing MCP entry instead of adding another copy. |
| A list request returns old channels | Call `refresh_channels` to replace that user's snapshot. |
| A refresh takes several seconds | This is expected: refresh queries Slack. Normal list tools read the stored file and should be much faster. |

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
