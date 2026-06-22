using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using VisionFlow.Graph;
using VisionFlow.Network;

namespace VisionFlow.UI
{
    /// <summary>Spatial Nostr timeline (kinds 30001, 31402, 31403).</summary>
    public sealed class EventFeedUI : MonoBehaviour
    {
        [SerializeField] private Transform         xrCameraRig;
        [SerializeField] private RectTransform     contentRoot;
        [SerializeField] private EventFeedItem     itemPrefab;
        [SerializeField] private ScrollRect        scrollRect;
        [SerializeField] private TextMeshProUGUI   labelEmpty;
        [SerializeField] private int               maxItems = 40;
        [SerializeField] private float             followDistance = 0.55f;
        [SerializeField] private float             followSmoothing = 8f;
        [SerializeField] private Vector3           panelOffset = new Vector3(-0.4f, 0f, 0f);

        private readonly List<EventFeedItem> _items = new();
        private readonly List<NostrEvent>    _events = new();
        private GraphVisualizer              _graph;
        private string                       _highlightedNodeId;
        private Vector3                      _targetPos;

        private void Awake()
        {
            if (labelEmpty != null)
                labelEmpty.gameObject.SetActive(true);
        }

        private void Start()
        {
            _graph = FindAnyObjectByType<GraphVisualizer>();
        }

        private void LateUpdate()
        {
            if (xrCameraRig == null) return;

            Vector3 forward = xrCameraRig.forward;
            forward.y = 0;
            _targetPos = xrCameraRig.position + forward.normalized * followDistance + panelOffset;

            transform.position = Vector3.Lerp(
                transform.position, _targetPos, Time.deltaTime * followSmoothing);
            transform.LookAt(xrCameraRig.position);
            transform.Rotate(0, 180, 0);
        }

        public void PushEvent(NostrEvent ev)
        {
            if (ev == null) return;

            _events.Insert(0, ev);
            while (_events.Count > maxItems)
                _events.RemoveAt(_events.Count - 1);

            RefreshList();
        }

        private void RefreshList()
        {
            if (labelEmpty != null)
                labelEmpty.gameObject.SetActive(_events.Count == 0);

            if (contentRoot == null || itemPrefab == null)
                return;

            while (_items.Count < _events.Count)
            {
                var row = Instantiate(itemPrefab, contentRoot);
                row.OnSelected += OnItemSelected;
                _items.Add(row);
            }

            for (int i = 0; i < _items.Count; i++)
            {
                if (i < _events.Count)
                    _items[i].Bind(_events[i]);
                else
                    _items[i].gameObject.SetActive(false);
            }

            if (scrollRect != null)
                Canvas.ForceUpdateCanvases();
        }

        private void OnItemSelected(NostrEvent ev)
        {
            if (_graph == null)
                _graph = FindAnyObjectByType<GraphVisualizer>();

            if (_graph == null || ev == null)
                return;

            _graph.ClearNodeHighlights();

            var nodeIds = new List<string>();
            ev.CollectReferencedNodeIds(nodeIds);

            string hit = null;
            foreach (var id in nodeIds)
            {
                if (_graph.TryGetNodeView(id, out _))
                {
                    hit = id;
                    break;
                }
            }

            if (hit == null && nodeIds.Count > 0)
                hit = nodeIds[0];

            if (string.IsNullOrEmpty(hit))
            {
                Debug.Log($"[EventFeedUI] No graph node for event {ev.Id}");
                return;
            }

            _highlightedNodeId = hit;
            _graph.HighlightNode(hit, true);
            Debug.Log($"[EventFeedUI] Highlight node {hit} from event kind {ev.Kind}");
        }
    }
}
