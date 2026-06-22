using UnityEngine;
using VisionFlow;

namespace VisionFlow.UI
{
    /// <summary>Runtime debug overlay: graph delta tick rate slider.</summary>
    public sealed class VisionFlowDebugHud : MonoBehaviour
    {
        [SerializeField] private bool  visible = true;
        [SerializeField] private Rect  panelRect = new Rect(12, 12, 280, 90);

        private float _tickRate = 10f;
        private bool  _dirty;

        private void Start()
        {
            if (VisionFlowManager.Instance?.Bridge != null)
                _tickRate = VisionFlowManager.Instance.Bridge.GraphTickRateHz;
        }

        private void OnGUI()
        {
            if (!visible || VisionFlowManager.Instance == null)
                return;

            GUI.Box(panelRect, "VisionFlow Debug");

            var inner = new Rect(panelRect.x + 10, panelRect.y + 28, panelRect.width - 20, 22);
            GUI.Label(new Rect(inner.x, inner.y - 22, inner.width, 18), $"Graph tick rate: {_tickRate:F0} Hz");

            float next = GUI.HorizontalSlider(inner, _tickRate, 1f, 60f);
            if (!Mathf.Approximately(next, _tickRate))
            {
                _tickRate = next;
                _dirty    = true;
            }

            if (_dirty && GUI.Button(new Rect(inner.x, inner.y + 28, 120, 22), "Apply"))
            {
                _dirty = false;
                VisionFlowManager.Instance.SetGraphTickRate((int)_tickRate);
            }
        }
    }
}
