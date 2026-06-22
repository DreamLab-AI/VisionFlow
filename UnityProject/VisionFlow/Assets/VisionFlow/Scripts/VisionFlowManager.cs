using System;
using System.Threading.Tasks;
using UnityEngine;
using VisionFlow.Crypto;
using VisionFlow.Graph;
using VisionFlow.Network;
using VisionFlow.UI;

namespace VisionFlow
{
    public sealed class VisionFlowManager : MonoBehaviour
    {
        [Header("VisionClaw WebSocket")]
        [SerializeField] private string visionClawWsUrl  = "ws://localhost:8765/v1/graph";

        [Header("Agentbox Solid Pod")]
        [SerializeField] private string solidPodBaseUrl  = "http://localhost:8484";

        [Header("Bridge tuning")]
        [SerializeField] private int graphTickRateHz = 10;
        [SerializeField] private bool showDebugHud = true;

        [Header("Scene References")]
        [SerializeField] private GraphVisualizer graphVisualizer;
        [SerializeField] private NodeInspectorUI nodeInspectorUI;
        [SerializeField] private GovernanceRequestUI governanceRequestUI;
        [SerializeField] private EventFeedUI eventFeedUI;

        public static VisionFlowManager Instance { get; private set; }

        public NostrIdentity  Identity  { get; private set; }
        public MCPBridgeClient Bridge   { get; private set; }
        public SolidPodClient  PodClient { get; private set; }
        public NodeInspectorUI NodeInspector => nodeInspectorUI;

        private void Awake()
        {
            if (Instance != null && Instance != this) { Destroy(gameObject); return; }
            Instance = this;
            DontDestroyOnLoad(gameObject);

            InitServices();
        }

        private async void Start()
        {
            Debug.Log($"[VisionFlow] DID: {Identity.Did}");
            await ConnectWithRetryAsync();
        }

        private void Update()
        {
            Bridge?.DrainMainThread();
        }

        private void OnDestroy()
        {
            Bridge?.Dispose();
            PodClient?.Dispose();
            Identity?.Dispose();
        }

        private void InitServices()
        {
            Identity  = NostrIdentity.Instance;
            PodClient = new SolidPodClient(solidPodBaseUrl, Identity);

            if (showDebugHud && GetComponent<VisionFlowDebugHud>() == null)
                gameObject.AddComponent<VisionFlowDebugHud>();

            CreateBridge();
        }

        private void CreateBridge()
        {
            Bridge?.Dispose();
            Bridge = new MCPBridgeClient(visionClawWsUrl);
            Bridge.GraphTickRateHz = Mathf.Clamp(graphTickRateHz, 1, 120);

            Bridge.OnSnapshot           += OnSnapshot;
            Bridge.OnDelta              += OnDelta;
            Bridge.OnNostrEvent         += OnNostrEvent;
            Bridge.OnGovernanceRequest  += OnGovernanceRequest;
            Bridge.OnError              += msg => Debug.LogWarning($"[Bridge] {msg}");
            Bridge.OnConnected          += () => Debug.Log("[VisionFlow] Graph stream connected.");
            Bridge.OnDisconnected       += () => Debug.Log("[VisionFlow] Graph stream disconnected.");
        }

        private async Task ConnectWithRetryAsync()
        {
            try
            {
                await Bridge.ConnectAsync();
            }
            catch (Exception ex)
            {
                Debug.LogError($"[VisionFlow] Fatal bridge error: {ex.Message}");
            }
        }

        private void OnSnapshot(GraphSnapshot snap)
        {
            Debug.Log($"[VisionFlow] Snapshot: {snap.Nodes.Count} nodes, {snap.Edges.Count} edges");
            graphVisualizer?.ApplySnapshot(snap);
        }

        private void OnDelta(GraphDelta delta)
        {
            graphVisualizer?.ApplyDelta(delta);
        }

        private void OnNostrEvent(NostrEvent ev)
        {
            eventFeedUI?.PushEvent(ev);
        }

        private void OnGovernanceRequest(GovernanceRequest request)
        {
            Debug.Log($"[VisionFlow] Governance request {request.EventId} from {request.AgentDid}");
            governanceRequestUI?.Enqueue(request);
        }

        public void SetGraphTickRate(int hz)
        {
            graphTickRateHz = Mathf.Clamp(hz, 1, 120);
            if (Bridge == null) return;

            Bridge.GraphTickRateHz = graphTickRateHz;
            _ = Bridge.ResubscribeGraphAsync();
        }

        private void OnApplicationPause(bool paused)
        {
            if (paused)
            {
                Bridge?.Dispose();
                Bridge = null;
                return;
            }

            CreateBridge();
            _ = ConnectWithRetryAsync();
        }

        /// <summary>Publish outbound governance request (kind 31402).</summary>
        public Task RequestGovernanceApproval(string agentDid, string action, string context)
        {
            return Bridge.SendAsync("nostr.publish", new
            {
                kind    = 31402,
                pubkey  = Identity.HexPubKey,
                tags    = new[]
                {
                    new[] { "agent",  agentDid },
                    new[] { "action", action   },
                },
                content    = context,
                created_at = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            });
        }

        /// <summary>Publish human decision (kind 31403) referencing request event id.</summary>
        public async Task<bool> PublishGovernanceDecision(string requestEventId, string decision, string reason = null)
        {
            if (string.IsNullOrEmpty(requestEventId))
                return false;

            var tags = new System.Collections.Generic.List<string[]>
            {
                new[] { "e", requestEventId },
                new[] { "decision", decision },
            };

            if (!string.IsNullOrWhiteSpace(reason))
                tags.Add(new[] { "reason", reason });

            try
            {
                await Bridge.SendAsync("nostr.publish", new
                {
                    kind    = 31403,
                    pubkey  = Identity.HexPubKey,
                    tags    = tags.ToArray(),
                    content = reason ?? string.Empty,
                    created_at = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                });
                return true;
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[VisionFlow] PublishGovernanceDecision failed: {ex.Message}");
                return false;
            }
        }
    }
}
