/**
 * Dev-only Solid pod stub for Unity SolidPodClient (port 8484).
 * Use until agentbox or solid-pod-rs-server is running.
 * Accepts NIP-98 Authorization headers without strict verification.
 */
import http from "node:http";

const HOST = process.env.POD_STUB_HOST ?? "127.0.0.1";
const PORT = Number(process.env.POD_STUB_PORT ?? 8484);
const REQUIRE_NIP98 = process.env.POD_REQUIRE_NIP98 === "1";

const solidDiscovery = {
  "@context": "https://www.w3.org/ns/solid/terms#",
  "@type": "ServiceResource",
  "solid:storageDescription": "VisionFlow dev pod stub",
};

function hasNip98(req) {
  const auth = req.headers.authorization ?? "";
  return auth.startsWith("Nostr ");
}

const server = http.createServer((req, res) => {
  const path = req.url?.split("?")[0] ?? "/";

  if (path === "/health" || path === "/.well-known/solid") {
    res.writeHead(200, { "Content-Type": "application/ld+json" });
    res.end(JSON.stringify(solidDiscovery));
    return;
  }

  if (path === "/profile/card" && req.method === "GET") {
    res.writeHead(200, { "Content-Type": "application/ld+json" });
    res.end(JSON.stringify({ "@id": "#me", "name": "VisionFlow XR Agent (dev stub)" }));
    return;
  }

  if (path === "/nostr/publish" && req.method === "PUT") {
    let body = "";
    req.on("data", (c) => { body += c; });
    req.on("end", () => {
      console.log("[pod-stub] provenance bead", body.slice(0, 120));
      res.writeHead(201, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ok: true, dev: true }));
    });
    return;
  }

  if (REQUIRE_NIP98 && !hasNip98(req)) {
    res.writeHead(401, { "Content-Type": "text/plain" });
    res.end("NIP-98 required");
    return;
  }

  if (req.method === "GET") {
    res.writeHead(200, { "Content-Type": "application/ld+json" });
    res.end(JSON.stringify({ "@id": path, "dev": true }));
    return;
  }

  if (["PUT", "PATCH", "DELETE"].includes(req.method ?? "")) {
    req.on("data", () => {});
    req.on("end", () => {
      res.writeHead(req.method === "PUT" ? 201 : 204);
      res.end();
    });
    return;
  }

  res.writeHead(404);
  res.end("not found");
});

server.listen(PORT, HOST, () => {
  console.log(`[pod-stub] http://${HOST}:${PORT} (NIP-98 verify=${REQUIRE_NIP98 ? "on" : "off"})`);
});
