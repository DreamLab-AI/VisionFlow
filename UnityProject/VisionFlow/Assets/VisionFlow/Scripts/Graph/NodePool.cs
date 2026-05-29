using System.Collections.Generic;
using UnityEngine;
using VisionFlow.Interaction;

namespace VisionFlow.Graph
{
    /// <summary>Pre-allocated pool of <see cref="NodeView"/> instances (default 512).</summary>
    public sealed class NodePool : MonoBehaviour
    {
        [SerializeField] private GameObject nodePrefab;
        [SerializeField] private int        initialCapacity = 512;

        private readonly Queue<NodeView>              _free   = new();
        private readonly Dictionary<string, NodeView>   _active = new();

        public IReadOnlyDictionary<string, NodeView> Active => _active;
        public int ActiveCount => _active.Count;

        public void Configure(GameObject prefab, Transform parent, int capacity)
        {
            nodePrefab       = prefab;
            initialCapacity  = Mathf.Max(capacity, 16);
            transform.SetParent(parent, false);
            Prewarm();
        }

        private void Awake() => Prewarm();

        public bool TryGet(string id, out NodeView view) => _active.TryGetValue(id, out view);

        public NodeView Acquire(string id)
        {
            if (_active.TryGetValue(id, out var existing))
                return existing;

            var view = _free.Count > 0
                ? _free.Dequeue()
                : CreateInstance();

            view.gameObject.name = id;
            view.gameObject.SetActive(true);
            _active[id] = view;
            return view;
        }

        public void Release(string id)
        {
            if (!_active.TryGetValue(id, out var view))
                return;

            _active.Remove(id);
            view.SetData(null, Color.white);
            view.SetHighlight(false);
            view.SetSelected(false);
            view.gameObject.SetActive(false);
            _free.Enqueue(view);
        }

        public void ReleaseAll()
        {
            var ids = new List<string>(_active.Keys);
            foreach (var id in ids)
                Release(id);
        }

        private void Prewarm()
        {
            while (_free.Count + _active.Count < initialCapacity)
                _free.Enqueue(CreateInstance());
        }

        private NodeView CreateInstance()
        {
            GameObject go;
            if (nodePrefab != null)
            {
                go = Instantiate(nodePrefab, transform);
            }
            else
            {
                go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                go.transform.SetParent(transform, false);
                go.transform.localScale = Vector3.one * 0.04f;
            }

            go.SetActive(false);
            var view = go.GetComponent<NodeView>() ?? go.AddComponent<NodeView>();
            GraphVisualizer.EnsureInteractionComponents(go);
            return view;
        }
    }
}
