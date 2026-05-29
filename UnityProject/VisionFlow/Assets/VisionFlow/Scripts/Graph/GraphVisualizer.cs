using System.Collections.Generic;
using UnityEngine;
using VisionFlow.Interaction;

namespace VisionFlow.Graph
{
    public sealed class GraphVisualizer : MonoBehaviour
    {
        [Header("Prefabs & Materials")]
        [SerializeField] private GameObject nodePrefab;
        [SerializeField] private Material   edgeMaterial;

        [Header("Performance")]
        [SerializeField] private int  nodePoolCapacity = 512;
        [SerializeField] private int  maxNodes         = 10_000;
        [SerializeField] private bool useNodePool      = true;
        [SerializeField] private bool useEdgeBatcher   = true;

        [Header("Layout")]
        [Tooltip("Multiplies VisionClaw x/y/z before layout normalization (CUDA layout units → metres).")]
        [SerializeField] private float scaleFactor = 0.01f;
        [Tooltip("Fit graph into this diameter (metres) around GraphRoot so nodes do not overlap.")]
        [SerializeField] private float layoutSpanMeters = 0.9f;
        [Tooltip("When all nodes share the same CUDA position, spread them on a circle with this radius.")]
        [SerializeField] private float fallbackCircleRadius = 0.35f;
        [SerializeField] private bool  autoNormalizeLayout = true;
        [SerializeField] private float edgeWidth = 0.002f;

        [Header("Debug")]
        [SerializeField] private bool logLayoutOnSnapshot;

        [Header("Colours — OWL relation type")]
        [SerializeField] private Color colourSubClassOf    = new Color(0.12f, 0.56f, 1.00f);
        [SerializeField] private Color colourDisjointWith  = new Color(0.93f, 0.27f, 0.27f);
        [SerializeField] private Color colourDefault       = new Color(0.65f, 0.65f, 0.65f);

        [Header("Node colour by community")]
        [SerializeField] private Gradient communityGradient;

        private NodePool    _nodePool;
        private EdgeBatcher _edgeBatcher;
        private Transform   _nodeRoot;
        private Transform   _edgeRoot;
        private Transform   _xrCamera;

        private readonly Dictionary<string, NodeView> _nodes = new();

        private Vector3 _layoutCenter;
        private float   _layoutUniformScale = 1f;
        private Dictionary<string, Vector3> _circularLayoutPositions;

        private const float InteractionColliderRadius = 0.04f;

        public IReadOnlyDictionary<string, NodeView> Nodes => _nodes;

        private void Awake()
        {
            _nodeRoot = CreateChild("Nodes");
            _edgeRoot = CreateChild("Edges");

            if (useNodePool)
            {
                _nodePool = _nodeRoot.gameObject.AddComponent<NodePool>();
                _nodePool.Configure(nodePrefab, _nodeRoot, nodePoolCapacity);
            }

            if (useEdgeBatcher)
            {
                _edgeBatcher = _edgeRoot.gameObject.AddComponent<EdgeBatcher>();
                var mat = edgeMaterial != null
                    ? edgeMaterial
                    : new Material(Shader.Find("Universal Render Pipeline/Unlit") ?? Shader.Find("Sprites/Default"));
                _edgeBatcher.Initialize(
                    _edgeRoot,
                    mat,
                    (EdgeBatcher.TryResolveNodePosition)ResolveNodeWorldPosition);
            }

            _xrCamera = Camera.main != null ? Camera.main.transform : null;
        }

        public void ApplySnapshot(GraphSnapshot snap)
        {
            ClearAll();
            if (autoNormalizeLayout)
                RebuildLayoutTransform(snap.Nodes);

            if (logLayoutOnSnapshot)
                LogLayout(snap.Nodes);

            foreach (var n in snap.Nodes) AddOrUpdateNode(n);
            foreach (var e in snap.Edges) AddOrUpdateEdge(e);
            ReindexNodes();
        }

        public void ApplyDelta(GraphDelta delta)
        {
            foreach (var n in delta.UpdatedNodes) AddOrUpdateNode(n);
            foreach (var e in delta.UpdatedEdges) AddOrUpdateEdge(e);
            foreach (var id in delta.RemovedNodes) RemoveNode(id);
            foreach (var id in delta.RemovedEdges) RemoveEdge(id);
            ReindexNodes();
        }

        public bool TryGetNodeView(string id, out NodeView view) => _nodes.TryGetValue(id, out view);

        public void HighlightNode(string id, bool on)
        {
            if (_nodes.TryGetValue(id, out var view))
                view.SetFeedHighlight(on);
        }

        public void ClearNodeHighlights()
        {
            foreach (var view in _nodes.Values)
                view?.SetFeedHighlight(false);
        }

        private void AddOrUpdateNode(GraphNode data)
        {
            if (_nodes.Count >= maxNodes && !_nodes.ContainsKey(data.Id))
            {
                Debug.LogWarning($"[GraphVisualizer] Node cap {maxNodes} reached; skipping {data.Id}");
                return;
            }

            if (!_nodes.TryGetValue(data.Id, out var view))
            {
                view = AcquireNodeView(data.Id);
                _nodes[data.Id] = view;
            }

            view.transform.localPosition = GetNodePosition(data);
            if (_xrCamera != null)
                view.SetCamera(_xrCamera);

            Color nodeColor = communityGradient != null && communityGradient.colorKeys.Length > 0
                ? communityGradient.Evaluate((data.Community % 16) / 16f)
                : Color.white;

            view.SetData(data, nodeColor);
        }

        private void RemoveNode(string id)
        {
            if (!_nodes.TryGetValue(id, out var view)) return;

            _nodes.Remove(id);
            if (_nodePool != null)
                _nodePool.Release(id);
            else if (view != null)
                Destroy(view.gameObject);
        }

        private void AddOrUpdateEdge(GraphEdge data)
        {
            if (!_nodes.ContainsKey(data.Source) || !_nodes.ContainsKey(data.Target))
                return;

            if (_edgeBatcher != null)
            {
                _edgeBatcher.SetEdge(
                    data.Id,
                    data.Source,
                    data.Target,
                    RelationColor(data.Relation),
                    edgeWidth * data.Weight);
            }
        }

        private void RemoveEdge(string id)
        {
            _edgeBatcher?.RemoveEdge(id);
        }

        private void LateUpdate()
        {
            if (_xrCamera == null && Camera.main != null)
                _xrCamera = Camera.main.transform;

            _edgeBatcher?.LateUpdateEdges(_nodes);
        }

        private NodeView AcquireNodeView(string id)
        {
            if (_nodePool != null)
                return _nodePool.Acquire(id);

            GameObject go;
            if (nodePrefab != null)
            {
                go = Instantiate(nodePrefab, _nodeRoot);
                go.name = id;
            }
            else
            {
                go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                go.name = id;
                go.transform.SetParent(_nodeRoot, false);
                go.transform.localScale = Vector3.one * 0.04f;
            }

            var view = go.GetComponent<NodeView>() ?? go.AddComponent<NodeView>();
            EnsureInteractionComponents(go);
            return view;
        }

        private bool ResolveNodeWorldPosition(string id, out Vector3 worldPos)
        {
            worldPos = Vector3.zero;
            if (!_nodes.TryGetValue(id, out var view) || view == null)
                return false;

            worldPos = view.transform.position;
            return true;
        }

        private void ReindexNodes()
        {
            int i = 0;
            foreach (var kv in _nodes)
            {
                if (kv.Value != null)
                    kv.Value.SetPoolIndex(i++);
            }
        }

        private Vector3 GetRawPosition(GraphNode data) =>
            new Vector3(data.X, data.Y, data.Z) * scaleFactor;

        private Vector3 GetNodePosition(GraphNode data)
        {
            if (_circularLayoutPositions != null &&
                _circularLayoutPositions.TryGetValue(data.Id, out var fixedPos))
                return fixedPos;

            if (!autoNormalizeLayout)
                return GetRawPosition(data);

            return (GetRawPosition(data) - _layoutCenter) * _layoutUniformScale;
        }

        private void RebuildLayoutTransform(List<GraphNode> nodes)
        {
            _circularLayoutPositions = null;
            _layoutCenter            = Vector3.zero;
            _layoutUniformScale      = 1f;

            if (nodes == null || nodes.Count == 0)
                return;

            var raws = new Vector3[nodes.Count];
            for (int i = 0; i < nodes.Count; i++)
                raws[i] = GetRawPosition(nodes[i]);

            var center = Vector3.zero;
            foreach (var p in raws)
                center += p;
            center /= raws.Length;

            float maxDist = 0f;
            foreach (var p in raws)
                maxDist = Mathf.Max(maxDist, Vector3.Distance(p, center));

            if (maxDist < 1e-5f)
            {
                _circularLayoutPositions = new Dictionary<string, Vector3>();
                int count = nodes.Count;
                float radius = Mathf.Max(fallbackCircleRadius, 0.12f * count);
                for (int i = 0; i < count; i++)
                {
                    float angle = (Mathf.PI * 2f * i) / count;
                    _circularLayoutPositions[nodes[i].Id] = new Vector3(
                        Mathf.Cos(angle) * radius,
                        0f,
                        Mathf.Sin(angle) * radius);
                }
                return;
            }

            _layoutCenter       = center;
            _layoutUniformScale = layoutSpanMeters / (2f * maxDist);
        }

        private void LogLayout(List<GraphNode> nodes)
        {
            foreach (var n in nodes)
            {
                var raw = GetRawPosition(n);
                var pos = GetNodePosition(n);
                Debug.Log($"[GraphVisualizer] {n.Id} cuda=({n.X:F3},{n.Y:F3},{n.Z:F3}) raw={raw} world={pos}");
            }
        }

        private Color RelationColor(string relation) => relation switch
        {
            "subClassOf"   => colourSubClassOf,
            "disjointWith" => colourDisjointWith,
            _              => colourDefault,
        };

        private void ClearAll()
        {
            if (_nodePool != null)
            {
                _nodePool.ReleaseAll();
            }
            else
            {
                foreach (var v in _nodes.Values)
                    if (v) Destroy(v.gameObject);
            }

            _nodes.Clear();
            _edgeBatcher?.Clear();
            _circularLayoutPositions = null;
        }

        public static void EnsureInteractionComponents(GameObject go)
        {
            if (go.GetComponent<NodeInteractable>() == null)
                go.AddComponent<NodeInteractable>();

            var col = go.GetComponent<SphereCollider>();
            if (col == null)
                col = go.AddComponent<SphereCollider>();

            col.isTrigger = false;
            col.radius    = InteractionColliderRadius;
        }

        private Transform CreateChild(string n)
        {
            var go = new GameObject(n);
            go.transform.SetParent(transform, false);
            return go.transform;
        }
    }
}
