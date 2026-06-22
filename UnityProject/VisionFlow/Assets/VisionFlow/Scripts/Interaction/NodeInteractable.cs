using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;
using UnityEngine.XR.Interaction.Toolkit.Interactables;
using VisionFlow.Graph;
using VisionFlow.UI;

namespace VisionFlow.Interaction
{
    [RequireComponent(typeof(NodeView))]
    [DisallowMultipleComponent]
    public sealed class NodeInteractable : XRSimpleInteractable
    {
        private NodeView         _nodeView;
        private NodeInspectorUI _inspector;

        private void Start()
        {
            _nodeView = GetComponent<NodeView>();
            _inspector = VisionFlowManager.Instance != null
                ? VisionFlowManager.Instance.NodeInspector
                : FindAnyObjectByType<NodeInspectorUI>();
        }

        protected override void OnHoverEntered(HoverEnterEventArgs args)
        {
            base.OnHoverEntered(args);
            _nodeView?.SetHighlight(true);
        }

        protected override void OnHoverExited(HoverExitEventArgs args)
        {
            if (args.isCanceled)
                return;

            base.OnHoverExited(args);
            _nodeView?.SetHighlight(false);
        }

        protected override void OnSelectEntered(SelectEnterEventArgs args)
        {
            base.OnSelectEntered(args);
            _nodeView?.SetSelected(true);

            if (_nodeView?.Data == null)
                return;

            if (_inspector == null)
                _inspector = VisionFlowManager.Instance?.NodeInspector
                              ?? FindAnyObjectByType<NodeInspectorUI>();

            _inspector?.Show(_nodeView.Data);
            Debug.Log($"[NodeInteractable] Selected {_nodeView.Data.Id}");
        }

        protected override void OnSelectExited(SelectExitEventArgs args)
        {
            if (args.isCanceled)
                return;

            base.OnSelectExited(args);
            _nodeView?.SetSelected(false);
        }
    }
}
