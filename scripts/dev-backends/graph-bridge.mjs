/**
 * Dev-only JSON-RPC graph stream for Unity MCPBridgeClient.
 * VisionClaw main exposes /wss (binary) on :4000, not ws://localhost:8765/v1/graph.
 * Remove when VisionClaw ships the canonical graph.subscribe endpoint.
 */
import http from "node:http";
import { WebSocketServer } from "ws";

const HOST = process.env.GRAPH_BRIDGE_HOST ?? "127.0.0.1";
const PORT = Number(process.env.GRAPH_BRIDGE_PORT ?? 8765);
const PATH = "/v1/graph";
const TICK_HZ = Number(process.env.GRAPH_TICK_HZ ?? 10);
const REQUIRE_NIP98 = process.env.GRAPH_REQUIRE_NIP98 === "1";

const sampleNodes = [
  { id: "n1", label: "VisionFlow", type: "owl:Class", x: 0, y: 0, z: 0, mass: 1, community: 0, pagerank: 0.9 },
  { id: "n2", label: "Graph Delta", type: "owl:Class", x: 1.2, y: 0.4, z: -0.3, mass: 1, community: 0, pagerank: 0.6 },
  { id: "n3", label: "Solid Pod", type: "owl:Class", x: -1.0, y: 0.2, z: 0.8, mass: 1, community: 1, pagerank: 0.5 },
];
const sampleEdges = [
  { id: "e1", source: "n1", target: "n2", relation: "subClassOf", weight: 1 },
  { id: "e2", source: "n1", target: "n3", relation: "relatedTo", weight: 0.8 },
];

function notify(ws, method, params) {
  ws.send(JSON.stringify({ jsonrpc: "2.0", method, params }));
}

function handleSubscribe(ws, params) {
  const tick = Date.now();
  notify(ws, "graph.snapshot", {
    nodes: sampleNodes,
    edges: sampleEdges,
    tick,
  });

  const intervalMs = Math.max(100, Math.round(1000 / TICK_HZ));
  let t = 0;
  const timer = setInterval(() => {
    if (ws.readyState !== ws.OPEN) {
      clearInterval(timer);
      return;
    }
    t += 1;
    const phase = t * 0.05;
    notify(ws, "graph.delta", {
      updated_nodes: sampleNodes.map((n, i) => ({
        ...n,
        y: n.y + Math.sin(phase + i) * 0.02,
      })),
      updated_edges: [],
      removed_nodes: [],
      removed_edges: [],
      tick: Date.now(),
    });
  }, intervalMs);

  ws.on("close", () => clearInterval(timer));
}

const server = http.createServer((_req, res) => {
  res.writeHead(200, { "Content-Type": "text/plain" });
  res.end("VisionFlow dev graph bridge — use WebSocket upgrade on /v1/graph\n");
});

const wss = new WebSocketServer({ noServer: true });

server.on("upgrade", (req, socket, head) => {
  const url = new URL(req.url ?? "/", `http://${req.headers.host ?? "localhost"}`);
  if (url.pathname !== PATH) {
    socket.write("HTTP/1.1 404 Not Found\r\n\r\n");
    socket.destroy();
    return;
  }

  const auth = req.headers.authorization ?? "";
  if (REQUIRE_NIP98 && !auth.startsWith("Nostr ")) {
    socket.write("HTTP/1.1 401 Unauthorized\r\n\r\n");
    socket.destroy();
    return;
  }

  wss.handleUpgrade(req, socket, head, (ws) => {
    wss.emit("connection", ws, req);
  });
});

wss.on("connection", (ws) => {
  ws.on("message", (raw) => {
    try {
      const msg = JSON.parse(String(raw));
      if (msg.method === "graph.subscribe") {
        handleSubscribe(ws, msg.params ?? {});
      }
    } catch (err) {
      console.error("[graph-bridge] bad message:", err.message);
    }
  });
});

server.listen(PORT, HOST, () => {
  console.log(`[graph-bridge] ws://${HOST}:${PORT}${PATH} (tick ${TICK_HZ} Hz)`);
});
