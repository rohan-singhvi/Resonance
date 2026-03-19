#!/usr/bin/env python3
"""
Patches /__substrate/space/server.ts with Resonance routes and home page badge.
Fast and idempotent — safe to run on every zo-space start via ~/.zo_secrets hook.
"""
import os
import sys

SERVER_TS = "/__substrate/space/server.ts"

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

// Resonance: inject badge script into home page HTML
app.use("/*", async (c, next) => {
  await next();
  const ct = c.res.headers.get("content-type") || "";
  if (c.req.path !== "/" && c.req.path !== "") return;
  if (!ct.includes("text/html")) return;
  const html = await c.res.text();
  if (html.includes("resonance-badge")) return;
  const badge = `<script>
(function(){
  var a=document.createElement("a");
  a.id="resonance-badge";
  a.href="/resonance/";
  a.textContent="Try Resonance";
  a.style.cssText="position:fixed;bottom:20px;right:20px;background:#6366f1;color:#fff;padding:10px 18px;border-radius:8px;font-family:Inter,system-ui,sans-serif;font-size:14px;font-weight:600;text-decoration:none;z-index:9999;box-shadow:0 4px 12px rgba(99,102,241,.4);transition:transform .15s,box-shadow .15s";
  a.onmouseenter=function(){a.style.transform="translateY(-2px)";a.style.boxShadow="0 6px 16px rgba(99,102,241,.5)"};
  a.onmouseleave=function(){a.style.transform="";a.style.boxShadow="0 4px 12px rgba(99,102,241,.4)"};
  document.body.appendChild(a);
})();
</script>`;
  c.res = new Response(html.replace("</body>", badge + "</body>"), {
    status: c.res.status,
    headers: c.res.headers,
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
    if "/resonance/api" in content and "resonance-badge" in content:
        return  # already patched, fast path
    # Strip old partial patch if badge is missing
    if "/resonance/api" in content and "resonance-badge" not in content:
        # Remove old patch block so we can re-insert the full one
        idx = content.find("// Resonance: proxy API calls")
        if idx != -1:
            end_idx = content.find(MARKER, idx)
            if end_idx != -1:
                content = content[:idx] + content[end_idx:]
    if MARKER not in content:
        print("server.ts: health endpoint marker not found")
        return
    content = content.replace(MARKER, ROUTES + MARKER)
    with open(SERVER_TS, "w") as f:
        f.write(content)
    print("server.ts patched")


if __name__ == "__main__":
    try:
        patch_server_ts()
    except Exception as e:
        print(f"server.ts error: {e}", file=sys.stderr)
