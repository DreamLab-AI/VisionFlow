using System.Collections.Generic;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;

namespace VisionFlow.Graph
{
    /// <summary>Batched line mesh for all graph edges (replaces per-edge LineRenderers).</summary>
    [RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
    public sealed class EdgeBatcher : MonoBehaviour
    {
        [SerializeField] private Material lineMaterial;
        [SerializeField] private float    defaultWidth = 0.002f;
        [SerializeField] private int      jobThreshold = 2048;

        private readonly Dictionary<string, EdgeRecord> _edges = new();
        private readonly List<string>                   _edgeOrder = new();

        private MeshFilter   _meshFilter;
        private MeshRenderer _meshRenderer;
        private Mesh         _mesh;

        private Vector3[] _vertices;
        private Color[]   _colors;
        private int[]       _indices;

        private NativeArray<Vector3> _jobVertices;
        private NativeArray<int>     _jobSrcIndices;
        private NativeArray<int>     _jobTgtIndices;
        private NativeArray<Vector3> _jobNodePositions;
        private bool _jobArraysAllocated;

        private TryResolveNodePosition _resolveWorldPosition;
        private Transform                _space;

        public delegate bool TryResolveNodePosition(string nodeId, out Vector3 worldPosition);

        private struct EdgeRecord
        {
            public string SrcId;
            public string TgtId;
            public Color  Color;
            public float  Width;
            public int    SrcIndex;
            public int    TgtIndex;
        }

        public int EdgeCount => _edges.Count;

        public void Initialize(Transform edgeSpace, Material material, TryResolveNodePosition resolveWorldPosition)
        {
            _space                  = edgeSpace != null ? edgeSpace : transform;
            lineMaterial            = material;
            _resolveWorldPosition   = resolveWorldPosition;

            _meshFilter   = GetComponent<MeshFilter>();
            _meshRenderer = GetComponent<MeshRenderer>();
            _mesh         = new Mesh { name = "GraphEdges", indexFormat = UnityEngine.Rendering.IndexFormat.UInt32 };
            _mesh.MarkDynamic();
            _meshFilter.sharedMesh = _mesh;

            if (lineMaterial != null)
                _meshRenderer.sharedMaterial = lineMaterial;
        }

        public void RebuildNodeIndexMap(IReadOnlyDictionary<string, NodeView> nodes)
        {
            int i = 0;
            foreach (var kv in nodes)
            {
                if (kv.Value == null || !kv.Value.gameObject.activeInHierarchy)
                    continue;
                kv.Value.SetPoolIndex(i++);
            }

            EnsureJobNodeBuffer(i);
            int idx = 0;
            foreach (var kv in nodes)
            {
                if (kv.Value == null || !kv.Value.gameObject.activeInHierarchy)
                    continue;
                _jobNodePositions[idx++] = kv.Value.transform.position;
                kv.Value.SetPoolIndex(idx - 1);
            }
        }

        public void SetEdge(string id, string srcId, string tgtId, Color color, float width = 0f)
        {
            if (string.IsNullOrEmpty(id)) return;

            if (!_edges.ContainsKey(id))
                _edgeOrder.Add(id);

            _edges[id] = new EdgeRecord
            {
                SrcId  = srcId,
                TgtId  = tgtId,
                Color  = color,
                Width  = width > 0f ? width : defaultWidth,
            };
        }

        public void RemoveEdge(string id)
        {
            if (!_edges.Remove(id)) return;
            _edgeOrder.Remove(id);
        }

        public void Clear()
        {
            _edges.Clear();
            _edgeOrder.Clear();
            ApplyMesh(0);
        }

        public void LateUpdateEdges(IReadOnlyDictionary<string, NodeView> nodes)
        {
            if (_resolveWorldPosition == null || _edgeOrder.Count == 0)
            {
                ApplyMesh(0);
                return;
            }

            int edgeCount = _edgeOrder.Count;
            int vertCount = edgeCount * 2;
            EnsureBuffers(vertCount);

            int writtenVerts = edgeCount >= jobThreshold
                ? RefreshWithJob(nodes, edgeCount)
                : RefreshMainThread(edgeCount);

            ApplyMesh(writtenVerts / 2);
        }

        private int RefreshMainThread(int edgeCount)
        {
            int v = 0;
            for (int e = 0; e < edgeCount; e++)
            {
                var rec = _edges[_edgeOrder[e]];
                if (!_resolveWorldPosition(rec.SrcId, out var src) ||
                    !_resolveWorldPosition(rec.TgtId, out var tgt))
                    continue;

                src = _space.InverseTransformPoint(src);
                tgt = _space.InverseTransformPoint(tgt);

                _vertices[v]     = src;
                _vertices[v + 1] = tgt;
                _colors[v]       = rec.Color;
                _colors[v + 1]   = rec.Color;
                v += 2;
            }

            return v;
        }

        private int RefreshWithJob(IReadOnlyDictionary<string, NodeView> nodes, int edgeCount)
        {
            int maxIndex = 0;
            foreach (var kv in nodes)
            {
                if (kv.Value == null || !kv.Value.gameObject.activeInHierarchy)
                    continue;

                int idx = kv.Value.PoolIndex;
                if (idx < 0) continue;

                EnsureJobNodeBuffer(idx + 1);
                _jobNodePositions[idx] = kv.Value.transform.position;
                maxIndex = Mathf.Max(maxIndex, idx + 1);
            }

            if (!_jobArraysAllocated)
                return RefreshMainThread(edgeCount);

            EnsureJobEdgeBuffers(edgeCount);
            int validEdges = 0;
            for (int e = 0; e < edgeCount; e++)
            {
                var rec = _edges[_edgeOrder[e]];
                int si = FindPoolIndex(nodes, rec.SrcId);
                int ti = FindPoolIndex(nodes, rec.TgtId);
                if (si < 0 || ti < 0) continue;

                _jobSrcIndices[validEdges] = si;
                _jobTgtIndices[validEdges] = ti;
                _edges[_edgeOrder[e]] = rec;
                validEdges++;
            }

            if (validEdges == 0)
                return 0;

            var job = new UpdateEdgeVerticesJob
            {
                NodePositions = _jobNodePositions,
                SrcIndices    = _jobSrcIndices,
                TgtIndices    = _jobTgtIndices,
                LocalToWorld  = _space.worldToLocalMatrix,
                Vertices      = _jobVertices,
            };
            job.Schedule(validEdges, 64).Complete();

            int written = 0;
            for (int e = 0; e < edgeCount; e++)
            {
                var rec = _edges[_edgeOrder[e]];
                if (FindPoolIndex(nodes, rec.SrcId) < 0 || FindPoolIndex(nodes, rec.TgtId) < 0)
                    continue;

                int vi = written * 2;
                _vertices[vi]     = _jobVertices[vi];
                _vertices[vi + 1] = _jobVertices[vi + 1];
                _colors[vi]       = rec.Color;
                _colors[vi + 1]   = rec.Color;
                written++;
            }

            return written * 2;
        }

        private static int FindPoolIndex(IReadOnlyDictionary<string, NodeView> nodes, string id)
        {
            if (string.IsNullOrEmpty(id) || !nodes.TryGetValue(id, out var view) || view == null)
                return -1;
            return view.PoolIndex;
        }

        private void ApplyMesh(int edgeCount)
        {
            int vertCount = edgeCount * 2;
            if (vertCount == 0)
            {
                _mesh.Clear();
                return;
            }

            if (_vertices.Length == vertCount)
            {
                _mesh.vertices = _vertices;
                _mesh.colors   = _colors;
            }
            else
            {
                var verts = new Vector3[vertCount];
                var cols  = new Color[vertCount];
                System.Array.Copy(_vertices, verts, vertCount);
                System.Array.Copy(_colors, cols, vertCount);
                _mesh.vertices = verts;
                _mesh.colors   = cols;
            }

            if (_indices == null || _indices.Length < vertCount)
            {
                _indices = new int[vertCount];
                for (int i = 0; i < vertCount; i++)
                    _indices[i] = i;
            }

            _mesh.SetIndices(_indices, 0, vertCount, MeshTopology.Lines, 0, false);
            _mesh.RecalculateBounds();
        }

        private void EnsureBuffers(int vertCount)
        {
            if (_vertices != null && _vertices.Length >= vertCount)
                return;

            _vertices = new Vector3[vertCount];
            _colors   = new Color[vertCount];
            _indices  = new int[vertCount];
            for (int i = 0; i < vertCount; i++)
                _indices[i] = i;
        }

        private void EnsureJobNodeBuffer(int count)
        {
            if (_jobArraysAllocated && _jobNodePositions.Length >= count)
                return;

            DisposeJobArrays();
            if (count <= 0) return;

            _jobNodePositions = new NativeArray<Vector3>(count, Allocator.Persistent);
            _jobArraysAllocated = true;
        }

        private void EnsureJobEdgeBuffers(int edgeCount)
        {
            int vertCount = edgeCount * 2;
            if (_jobVertices.IsCreated && _jobVertices.Length >= vertCount && _jobSrcIndices.Length >= edgeCount)
                return;

            if (_jobVertices.IsCreated) _jobVertices.Dispose();
            if (_jobSrcIndices.IsCreated) _jobSrcIndices.Dispose();
            if (_jobTgtIndices.IsCreated) _jobTgtIndices.Dispose();

            _jobVertices    = new NativeArray<Vector3>(vertCount, Allocator.Persistent);
            _jobSrcIndices  = new NativeArray<int>(edgeCount, Allocator.Persistent);
            _jobTgtIndices  = new NativeArray<int>(edgeCount, Allocator.Persistent);
        }

        private void DisposeJobArrays()
        {
            if (_jobNodePositions.IsCreated) _jobNodePositions.Dispose();
            if (_jobVertices.IsCreated) _jobVertices.Dispose();
            if (_jobSrcIndices.IsCreated) _jobSrcIndices.Dispose();
            if (_jobTgtIndices.IsCreated) _jobTgtIndices.Dispose();
            _jobArraysAllocated = false;
        }

        private void OnDestroy() => DisposeJobArrays();

        private struct UpdateEdgeVerticesJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<Vector3> NodePositions;
            [ReadOnly] public NativeArray<int>     SrcIndices;
            [ReadOnly] public NativeArray<int>     TgtIndices;
            public Matrix4x4 LocalToWorld;
            public NativeArray<Vector3> Vertices;

            public void Execute(int edgeIndex)
            {
                int vi = edgeIndex * 2;
                int si = SrcIndices[edgeIndex];
                int ti = TgtIndices[edgeIndex];

                Vector3 src = si >= 0 && si < NodePositions.Length ? NodePositions[si] : Vector3.zero;
                Vector3 tgt = ti >= 0 && ti < NodePositions.Length ? NodePositions[ti] : Vector3.zero;

                Vertices[vi]     = LocalToWorld.MultiplyPoint3x4(src);
                Vertices[vi + 1] = LocalToWorld.MultiplyPoint3x4(tgt);
            }
        }
    }
}
