#!/usr/bin/env python3
"""
Patches /__substrate/space/server.ts and _home.tsx with Resonance routes.
Fast and idempotent — safe to run on every zo-space start via ~/.zo_secrets hook.
"""
import os
import sys

SERVER_TS = "/__substrate/space/server.ts"
HOME_TSX = "/__substrate/space/routes/pages/_home.tsx"

# Use a raw string to avoid escape-sequence warnings
ROUTES = r"""
// Resonance: proxy API calls to port 8420
app.all("/resonance/api/*", async (c) => {
  const url = c.req.url.replace(/.*\/resonance\/api/, "http://127.0.0.1:8420/api");
  const resp = await fetch(url, {
    method: c.req.method,
    headers: c.req.raw.headers,
    body: ["GET", "HEAD"].includes(c.req.method) ? undefined : c.req.raw.body,
  });
  return new Response(resp.body, { status: resp.status, headers: resp.headers });
});

// Resonance: serve frontend
app.get("/resonance", (c) => c.redirect("/resonance/", 301));
app.get("/resonance/*", async (c) => {
  const reqPath = c.req.path.replace(/^\/resonance/, "") || "/index.html";
  const filePath = reqPath === "/" ? "/index.html" : reqPath;
  const file = Bun.file(`${ASSETS_DIR}/resonance${filePath}`);
  if (await file.exists()) {
    const headers: Record<string, string> = { "Content-Type": file.type };
    if (filePath === "/index.html") headers["Cache-Control"] = "no-cache";
    return new Response(file.stream(), { headers });
  }
  const index = Bun.file(`${ASSETS_DIR}/resonance/index.html`);
  return new Response(index.stream(), {
    headers: { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-cache" },
  });
});

"""

MARKER = 'app.get("/api/_health"'


def patch_server_ts():
    if not os.path.isfile(SERVER_TS):
        print("server.ts not found, skipping")
        return
    with open(SERVER_TS) as f:
        content = f.read()
    if "/resonance/api" in content:
        return  # already patched, fast path
    if MARKER not in content:
        print("server.ts: health endpoint marker not found")
        return
    content = content.replace(MARKER, ROUTES + MARKER)
    with open(SERVER_TS, "w") as f:
        f.write(content)
    print("server.ts patched")


def patch_home_tsx():
    if not os.path.isfile(HOME_TSX):
        print("_home.tsx not found, skipping")
        return
    with open(HOME_TSX) as f:
        content = f.read()
    old = 'href="https://github.com/rohan-singhvi/Resonance"'
    new = 'href="/resonance/"'
    if old not in content:
        return  # already patched or different format, fast path
    content = content.replace(old, new)
    with open(HOME_TSX, "w") as f:
        f.write(content)
    print("_home.tsx patched")


if __name__ == "__main__":
    try:
        patch_server_ts()
    except Exception as e:
        print(f"server.ts error: {e}", file=sys.stderr)
    try:
        patch_home_tsx()
    except Exception as e:
        print(f"_home.tsx error: {e}", file=sys.stderr)
