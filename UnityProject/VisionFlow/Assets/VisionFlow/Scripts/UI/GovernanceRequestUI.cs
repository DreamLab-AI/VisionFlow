using System.Collections.Generic;
using System.Threading.Tasks;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using VisionFlow.Network;

namespace VisionFlow.UI
{
    /// <summary>
    /// World Space panel for human-in-the-loop governance (kind 31402 → 31403).
    /// </summary>
    public sealed class GovernanceRequestUI : MonoBehaviour
    {
        [Header("Labels")]
        [SerializeField] private TextMeshProUGUI labelTitle;
        [SerializeField] private TextMeshProUGUI labelAgent;
        [SerializeField] private TextMeshProUGUI labelAction;
        [SerializeField] private TextMeshProUGUI labelContext;
        [SerializeField] private TextMeshProUGUI labelQueue;

        [Header("Buttons")]
        [SerializeField] private Button btnApprove;
        [SerializeField] private Button btnReject;
        [SerializeField] private TMP_InputField inputRejectReason;

        [Header("HUD")]
        [SerializeField] private GovernanceHUDBadge hudBadge;

        [Header("Panel")]
        [SerializeField] private GameObject reviewPanel;

        [Header("Follow")]
        [SerializeField] private Transform xrCameraRig;
        [SerializeField] private float followDistance = 0.65f;
        [SerializeField] private float followSmoothing = 8f;
        [SerializeField] private Vector3 panelOffset = new Vector3(0.35f, 0f, 0f);

        private readonly Queue<GovernanceRequest> _queue = new();
        private GovernanceRequest _current;
        private bool _visible;
        private Vector3 _targetPos;

        public int PendingCount => _queue.Count + (_current != null ? 1 : 0);

        private void Awake()
        {
            btnApprove?.onClick.AddListener(OnApprove);
            btnReject?.onClick.AddListener(OnReject);
            if (reviewPanel == null)
            {
                var panel = transform.Find("Panel");
                if (panel != null)
                    reviewPanel = panel.gameObject;
            }

            SetReviewVisible(false);
            if (inputRejectReason != null)
                inputRejectReason.gameObject.SetActive(false);
        }

        private void LateUpdate()
        {
            if (!_visible || xrCameraRig == null) return;

            Vector3 forward = xrCameraRig.forward;
            forward.y = 0;
            _targetPos = xrCameraRig.position + forward.normalized * followDistance + panelOffset;

            transform.position = Vector3.Lerp(
                transform.position, _targetPos, Time.deltaTime * followSmoothing);
            transform.LookAt(xrCameraRig.position);
            transform.Rotate(0, 180, 0);
        }

        public void Enqueue(GovernanceRequest request)
        {
            if (request == null) return;
            _queue.Enqueue(request);
            RefreshBadge();

            if (_current == null)
                ShowNext();
        }

        private void ShowNext()
        {
            if (_queue.Count == 0)
            {
                Hide();
                return;
            }

            _current = _queue.Dequeue();
            _visible = true;
            SetReviewVisible(true);
            if (inputRejectReason != null)
            {
                inputRejectReason.text = string.Empty;
                inputRejectReason.gameObject.SetActive(false);
            }

            Refresh();
            RefreshBadge();
            Debug.Log($"[GovernanceUI] Reviewing request {_current.EventId}");
        }

        private void Hide()
        {
            _current = null;
            _visible = false;
            SetReviewVisible(false);
            RefreshBadge();
        }

        private void SetReviewVisible(bool visible)
        {
            if (reviewPanel != null)
                reviewPanel.SetActive(visible);
        }

        private void Refresh()
        {
            if (_current == null) return;

            SetLabel(labelTitle,   "Governance request");
            SetLabel(labelAgent,   $"Agent: {_current.AgentDid}");
            SetLabel(labelAction,  $"Action: {_current.Action}");
            SetLabel(labelContext, string.IsNullOrEmpty(_current.Context)
                ? "Context: —"
                : $"Context: {_current.Context}");
            SetLabel(labelQueue,   _queue.Count > 0 ? $"Queued: {_queue.Count}" : string.Empty);
        }

        private void RefreshBadge() => hudBadge?.SetPendingCount(PendingCount);

        private void OnApprove() => PublishDecisionAsync("approved", null);

        private void OnReject()
        {
            if (inputRejectReason != null && !inputRejectReason.gameObject.activeSelf)
            {
                inputRejectReason.gameObject.SetActive(true);
                return;
            }

            string reason = inputRejectReason != null ? inputRejectReason.text : string.Empty;
            PublishDecisionAsync("rejected", reason);
        }

        private async void PublishDecisionAsync(string decision, string reason)
        {
            if (_current == null) return;
            var manager = VisionFlowManager.Instance;
            if (manager == null) return;

            bool ok = await manager.PublishGovernanceDecision(_current.EventId, decision, reason);
            Debug.Log(ok
                ? $"[GovernanceUI] Published {decision} for {_current.EventId}"
                : $"[GovernanceUI] Failed to publish {decision} for {_current.EventId}");

            ShowNext();
        }

        private static void SetLabel(TextMeshProUGUI label, string text)
        {
            if (label != null) label.text = text;
        }
    }
}
