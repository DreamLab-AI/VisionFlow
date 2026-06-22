using System;
using System.Collections.Generic;
using System.Net.WebSockets;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using VisionFlow.Crypto;
using VisionFlow.Graph;

namespace VisionFlow.Network
{
    public sealed class MCPBridgeClient : IDisposable
    {
        // ── Configuration ──────────────────────────────────────────────────

        public string  EndpointUrl    { get; }
        public int     ReconnectDelay { get; set; } = 3_000; // ms
        public int     BufferSize     { get; set; } = 65_536; // 64 KB
        public int     GraphTickRateHz { get; set; } = 10;

        // ── Events (fired on background thread — dispatch to main via Queue) ─

        public event Action<GraphSnapshot> OnSnapshot;
        public event Action<GraphDelta>    OnDelta;
        public event Action<GovernanceRequest> OnGovernanceRequest;
        public event Action<NostrEvent>        OnNostrEvent;
        public event Action<string>        OnError;
        public event Action                OnConnected;
        public event Action                OnDisconnected;

        // ── Main-thread dispatch queue ─────────────────────────────────────

        private readonly Queue<Action> _mainThreadQueue = new();
        private readonly object        _queueLock       = new();

        // ── Private state ──────────────────────────────────────────────────

        private ClientWebSocket    _ws;
        private CancellationTokenSource _cts;
        private bool _disposed;
        private int  _msgIdCounter;

        // ── Constructor ────────────────────────────────────────────────────

        public MCPBridgeClient(string endpointUrl)
        {
            EndpointUrl = endpointUrl;
        }

        // ── Connect ────────────────────────────────────────────────────────

        public async Task ConnectAsync()
        {
            _cts?.Dispose();
            _cts = new CancellationTokenSource();

            while (!_disposed && !_cts.IsCancellationRequested)
            {
                try
                {
                    _ws?.Dispose();
                    _ws = new ClientWebSocket();

                    // Attach NIP-98 auth header
                    string authHeader = NostrIdentity.Instance.BuildNip98Header(
                        EndpointUrl, "GET");
                    _ws.Options.SetRequestHeader("Authorization", authHeader);

                    Debug.Log($"[MCPBridge] Connecting → {EndpointUrl}");
                    await _ws.ConnectAsync(new Uri(EndpointUrl), _cts.Token);

                    Enqueue(() => OnConnected?.Invoke());
                    Debug.Log("[MCPBridge] Connected.");

                    await SendSubscribeAsync();
                    await SendNostrSubscribeAsync();
                    await ReceiveLoopAsync();
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    Enqueue(() => OnError?.Invoke(ex.Message));
                    Debug.LogWarning($"[MCPBridge] Error: {ex.Message}. Reconnecting in {ReconnectDelay}ms…");
                    Enqueue(() => OnDisconnected?.Invoke());

                    await Task.Delay(ReconnectDelay, _cts.Token).ContinueWith(_ => { });
                }
            }
        }

        // ── Send helpers ───────────────────────────────────────────────────

        private Task SendSubscribeAsync()
        {
            var req = new
            {
                jsonrpc = "2.0",
                method  = "graph.subscribe",
                id      = NextId(),
                @params = new { mode = "delta", tick_rate_hz = Mathf.Clamp(GraphTickRateHz, 1, 120) },
            };
            return SendJsonAsync(req);
        }

        private Task SendNostrSubscribeAsync()
        {
            var req = new
            {
                jsonrpc = "2.0",
                method  = "nostr.subscribe",
                id      = NextId(),
                @params = new { kinds = new[] { 31402, 31403, 30001 } },
            };
            return SendJsonAsync(req);
        }

        /// <summary>Re-send graph.subscribe with current <see cref="GraphTickRateHz"/> (after slider change).</summary>
        public Task ResubscribeGraphAsync() => SendSubscribeAsync();

        public Task SendAsync(string method, object @params = null)
        {
            var req = new
            {
                jsonrpc = "2.0",
                method,
                id      = NextId(),
                @params,
            };
            return SendJsonAsync(req);
        }

        private async Task SendJsonAsync(object payload)
        {
            if (_ws?.State != WebSocketState.Open) return;
            byte[] bytes = Encoding.UTF8.GetBytes(VisionFlowJson.Serialize(payload));
            await _ws.SendAsync(
                new ArraySegment<byte>(bytes),
                WebSocketMessageType.Text,
                true,
                _cts.Token);
        }

        // ── Receive loop ───────────────────────────────────────────────────

        private async Task ReceiveLoopAsync()
        {
            var buffer = new byte[BufferSize];

            while (_ws.State == WebSocketState.Open && !_cts.IsCancellationRequested)
            {
                var result = await _ws.ReceiveAsync(new ArraySegment<byte>(buffer), _cts.Token);

                if (result.MessageType == WebSocketMessageType.Close)
                {
                    await _ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "close", _cts.Token);
                    break;
                }

                string json = Encoding.UTF8.GetString(buffer, 0, result.Count);
                DispatchMessage(json);
            }

            Enqueue(() => OnDisconnected?.Invoke());
        }

        private void DispatchMessage(string json)
        {
            try
            {
                var msg = VisionFlowJson.Deserialize<JsonRpcMessage>(json);
                if (msg == null) return;

                // Route by method name (notifications) or result shape
                string method = msg.Method ?? string.Empty;

                if (method == "graph.snapshot")
                {
                    var snap = msg.Params?.ToObject<GraphSnapshot>();
                    if (snap != null) Enqueue(() => OnSnapshot?.Invoke(snap));
                }
                else if (method == "graph.delta")
                {
                    var delta = msg.Params?.ToObject<GraphDelta>();
                    if (delta != null) Enqueue(() => OnDelta?.Invoke(delta));
                }
                else if (method == "nostr.event")
                {
                    var nostr = ExtractNostrEvent(msg.Params);
                    if (nostr == null) return;

                    Enqueue(() => OnNostrEvent?.Invoke(nostr));

                    if (GovernanceRequest.TryFromNostrEvent(nostr, out var gov))
                        Enqueue(() => OnGovernanceRequest?.Invoke(gov));
                }
                else if (msg.Error != null && msg.Error.Code != 0)
                {
                    Enqueue(() => OnError?.Invoke($"RPC error {msg.Error.Code}: {msg.Error.Message}"));
                }
            }
            catch (JsonException ex)
            {
                Debug.LogWarning($"[MCPBridge] JSON parse error: {ex.Message}");
            }
        }

        // ── Main-thread queue (call DrainMainThread() from a MonoBehaviour.Update) ─

        private void Enqueue(Action action)
        {
            lock (_queueLock) _mainThreadQueue.Enqueue(action);
        }

        /// <summary>Call this every frame from a MonoBehaviour to dispatch events on main thread.</summary>
        public void DrainMainThread()
        {
            lock (_queueLock)
            {
                while (_mainThreadQueue.Count > 0)
                    _mainThreadQueue.Dequeue()?.Invoke();
            }
        }

        // ── IDisposable ────────────────────────────────────────────────────

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _cts?.Cancel();
            _cts?.Dispose();
            _ws?.Dispose();
        }

        private static NostrEvent ExtractNostrEvent(JToken parameters)
        {
            if (parameters == null) return null;
            if (parameters["event"] != null)
                return parameters["event"]?.ToObject<NostrEvent>();
            if (parameters["kind"] != null)
                return parameters.ToObject<NostrEvent>();
            return null;
        }

        private string NextId() => (++_msgIdCounter).ToString();
    }
}
