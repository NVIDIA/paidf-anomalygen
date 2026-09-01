# Automatic Mask Placement (AMP) GUI

The Automatic Mask Placement GUI is the **primary and recommended interface** for most users.
It provides visualization and interactive control, making it the most efficient way
to design, debug, and validate mask placement strategies.

The CLI interface is intended for batch processing and fully reproducible large-scale generation
after configurations have been verified through the GUI.

To start the GUI server:

```shell
AMP_PORT=5000 python3 -m anomalygen.auto_mask_placement.gui.backend.app
```

`AMP_PORT` specifies the port number on which the GUI backend server listens.
If not explicitly set, the backend will use its default port 5000.

## Opening the UI

On startup the launcher prints a URL that carries a one-time token:

```text
🌐 Open http://127.0.0.1:5000/?token=<token>
```

**Open that URL as printed.** Plain `http://localhost:<AMP_PORT>` serves the page, but every
`/api/*` call returns `401` until the token has been exchanged. Opening the printed URL swaps the
token for a `SameSite=Strict` cookie and redirects to a clean `/`, so the token does not stay in
the address bar or in browser history.

Why there is a token at all: the server binds to loopback, and loopback is not an access control —
every process running as your user can reach the port. The token is what stops another local
process from driving the API, and the `SameSite=Strict` cookie is what stops a web page you visit
from doing the same through your browser.

## Environment variables

| Variable          | Default        | Purpose                                                      |
| ----------------- | -------------- | ------------------------------------------------------------ |
| `AMP_PORT`        | `5000`         | Port to listen on (loopback only).                           |
| `AMP_AUTH_TOKEN`  | new each start | Pins the token so a script knows the URL in advance.         |
| `AMP_OUTPUT_ROOT` | the checkout   | Tree the GUI may read and write; paths outside it get `400`. |

Two cautions:

- **`AMP_AUTH_TOKEN` is readable by other processes of the same user** (`/proc/<pid>/environ`, and
  `docker inspect` for a container). Leaving it unset — so a fresh token is generated per start and
  never leaves the terminal — is the stronger option. Set it only when a script needs the URL in
  advance, and treat it as a credential.
- **Restarting the server invalidates the previous token.** Reopen the newly printed URL; a stale
  cookie gives `401`.

## Scope of the token

Anyone holding the token has the full authority of the API — upload, generate, download results,
clear results. There are no separate accounts or roles. The per-browser session id
(`X-AMP-Session`) only keeps concurrent tabs writing to separate directories; it is **not** an
isolation boundary and does not restrict what a token holder can reach.

## Running in a container

The server binds `127.0.0.1` *inside* the container, so publishing a port with `docker run -p`
will not reach it. Use `--network host`, or run the GUI on the host.
