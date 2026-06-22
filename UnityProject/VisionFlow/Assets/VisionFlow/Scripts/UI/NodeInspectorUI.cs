using UnityEngine;
using UnityEngine.UI;
using TMPro;
using VisionFlow.Graph;

namespace VisionFlow.UI
{
    /// <summary>
    /// Floating XR panel that shows OWL metadata for the selected node.
    ///
    /// Attach to a World Space Canvas GameObject.
    /// Wire Inspector references to TMP labels and the close button.
    ///
    /// Activated by NodeView selection events (XRI Select).
    /// </summary>
    public sealed class NodeInspectorUI : MonoBehaviour
    {
        // ── Inspector ──────────────────────────────────────────────────────

        [Header("Labels")]
        [SerializeField] private TextMeshProUGUI labelTitle;
        [SerializeField] private TextMeshProUGUI labelType;
        [SerializeField] private TextMeshProUGUI labelDid;
        [SerializeField] private TextMeshProUGUI labelProvenance;
        [SerializeField] private TextMeshProUGUI labelPageRank;
        [SerializeField] private TextMeshProUGUI labelCommunity;

        [Header("Buttons")]
        [SerializeField] private Button btnClose;
        [SerializeField] private Button btnWriteBead;   // write provenance bead to pod

        [Header("Follow settings")]
        [SerializeField] private Transform xrCameraRig;
        [SerializeField] private float     followDistance = 0.6f;
        [SerializeField] private float     followSmoothing = 8f;

        // ── Runtime ────────────────────────────────────────────────────────

        private GraphNode _current;
        private bool      _visible;
        private Vector3   _targetPos;

        // ── Unity lifecycle ────────────────────────────────────────────────

        private void Awake()
        {
            btnClose?.onClick.AddListener(Hide);
            btnWriteBead?.onClick.AddListener(OnWriteBead);
            gameObject.SetActive(false);
        }

        private void LateUpdate()
        {
            if (!_visible) return;
            if (xrCameraRig == null) return;

            // Panel follows gaze direction, 60 cm in front of camera
            Vector3 forward   = xrCameraRig.forward;
            forward.y         = 0;
            _targetPos        = xrCameraRig.position + forward.normalized * followDistance
                                + Vector3.up * -0.05f;

            transform.position = Vector3.Lerp(
                transform.position, _targetPos, Time.deltaTime * followSmoothing);

            transform.LookAt(xrCameraRig.position);
            transform.Rotate(0, 180, 0);   // face towards user
        }

        // ── Public API ─────────────────────────────────────────────────────

        public void Show(GraphNode node)
        {
            _current = node;
            _visible = true;
            gameObject.SetActive(true);
            Refresh();
        }

        public void Hide()
        {
            _visible = false;
            gameObject.SetActive(false);
        }

        // ── Private ────────────────────────────────────────────────────────

        private void Refresh()
        {
            if (_current == null) return;

            SetLabel(labelTitle,      _current.Label ?? _current.Id);
            SetLabel(labelType,       $"Type: {_current.Type ?? "—"}");
            SetLabel(labelDid,        $"DID: did:nostr:{VisionFlowManager.Instance?.Identity?.HexPubKey ?? "…"}");
            SetLabel(labelProvenance, $"Bead: {_current.Provenance ?? "none"}");
            SetLabel(labelPageRank,   $"PageRank: {_current.PageRank:F4}");
            SetLabel(labelCommunity,  $"Community: {_current.Community}");
        }

        private static void SetLabel(TextMeshProUGUI label, string text)
        {
            if (label != null) label.text = text;
        }

        private async void OnWriteBead()
        {
            if (_current == null) return;
            if (VisionFlowManager.Instance?.PodClient == null) return;

            bool ok = await VisionFlowManager.Instance.PodClient.WriteProvenanceBeadAsync(
                subjectId    : _current.Id,
                predicateUri : "https://schema.org/about",
                objectValue  : _current.Label ?? _current.Id);

            Debug.Log(ok
                ? $"[NodeInspector] Bead written for {_current.Id}"
                : $"[NodeInspector] Bead write failed for {_current.Id}");
        }
    }
}
