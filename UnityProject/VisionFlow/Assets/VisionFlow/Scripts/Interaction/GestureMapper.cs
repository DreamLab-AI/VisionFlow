using UnityEngine;
using UnityEngine.XR.Hands;
using UnityEngine.XR.Interaction.Toolkit;
using UnityEngine.XR.Interaction.Toolkit.Interactables;
using UnityEngine.XR.Interaction.Toolkit.Interactors;
using VisionFlow.Graph;

namespace VisionFlow.Interaction
{
    /// <summary>
    /// Hand-tracking fallback: pinch → raycast → <see cref="NodeInteractable"/> select via
    /// <see cref="NearFarInteractor"/> on the matching hand.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class GestureMapper : MonoBehaviour
    {
        [SerializeField] private GraphRaycaster       graphRaycaster;
        [SerializeField] private XRInteractionManager interactionManager;
        [SerializeField] private XRHandTrackingEvents leftHandEvents;
        [SerializeField] private XRHandTrackingEvents rightHandEvents;

        [Header("Pinch")]
        [SerializeField] private float pinchDistanceThreshold = 0.028f;
        [Tooltip("Optional pinch strength from OpenXR hand interaction (0–1).")]
        [SerializeField] private float pinchStrengthThreshold = 0.75f;

        [Header("Aim")]
        [SerializeField] private float selectRayLength  = 0.75f;
        [SerializeField] private float raySphereRadius  = 0.03f;
        [SerializeField] private LayerMask interactableLayers = ~0;
        [SerializeField] private bool    enableHoverHighlight = true;

        private readonly HandState _left  = new();
        private readonly HandState _right = new();
        private bool _loggedReady;

        private void Awake()
        {
            if (graphRaycaster == null)
                graphRaycaster = GetComponent<GraphRaycaster>()
                                 ?? FindAnyObjectByType<GraphRaycaster>();

            if (interactionManager == null)
                interactionManager = FindAnyObjectByType<XRInteractionManager>();

            CacheHandEvents();
        }

        private void OnEnable()
        {
            BindHand(leftHandEvents,  _left,  Handedness.Left);
            BindHand(rightHandEvents, _right, Handedness.Right);
        }

        private void OnDisable()
        {
            UnbindHand(leftHandEvents,  _left);
            UnbindHand(rightHandEvents, _right);
            ReleaseAllSelections();
            ClearAllHovers();
        }

        private void CacheHandEvents()
        {
            if (leftHandEvents != null && rightHandEvents != null)
                return;

            foreach (var handEvents in FindObjectsByType<XRHandTrackingEvents>(FindObjectsInactive.Include))
            {
                if (handEvents.handedness == Handedness.Left && leftHandEvents == null)
                    leftHandEvents = handEvents;
                else if (handEvents.handedness == Handedness.Right && rightHandEvents == null)
                    rightHandEvents = handEvents;
            }
        }

        private void BindHand(XRHandTrackingEvents events, HandState state, Handedness handedness)
        {
            if (events == null)
                return;

            state.Handedness = handedness;
            state.OnJointsUpdated = args => OnJointsUpdated(args, state);
            events.jointsUpdated.AddListener(state.OnJointsUpdated);
        }

        private static void UnbindHand(XRHandTrackingEvents events, HandState state)
        {
            if (events == null || state.OnJointsUpdated == null)
                return;

            events.jointsUpdated.RemoveListener(state.OnJointsUpdated);
            state.OnJointsUpdated = null;
        }

        private void OnJointsUpdated(XRHandJointsUpdatedEventArgs args, HandState state)
        {
            if (!args.hand.isTracked)
            {
                EndPinch(state);
                SetHover(state, null);
                return;
            }

            bool pinching = IsPinching(args);
            if (!TryGetAimRay(args, out Vector3 origin, out Vector3 direction))
            {
                EndPinch(state);
                return;
            }

            var target = RaycastNode(origin, direction);
            if (enableHoverHighlight)
                SetHover(state, target);

            if (pinching && !state.WasPinching)
                BeginPinchSelect(state, target);
            else if (!pinching && state.WasPinching)
                EndPinch(state);

            state.WasPinching = pinching;

            if (!_loggedReady && (leftHandEvents != null || rightHandEvents != null))
            {
                _loggedReady = true;
                Debug.Log("[GestureMapper] Hand pinch → graph select active.");
            }
        }

        private bool IsPinching(XRHandJointsUpdatedEventArgs args)
        {
#if XR_HANDS_1_5_OR_NEWER
            if (args.subsystem != null)
            {
                var gestures = args.hand.handedness == Handedness.Left
                    ? args.subsystem.leftHandCommonGestures
                    : args.hand.handedness == Handedness.Right
                        ? args.subsystem.rightHandCommonGestures
                        : null;

                if (gestures != null && gestures.TryGetPinchStrength(out float strength))
                    return strength >= pinchStrengthThreshold;
            }
#endif

            var thumb = args.hand.GetJoint(XRHandJointID.ThumbTip);
            var index = args.hand.GetJoint(XRHandJointID.IndexTip);
            if (!thumb.TryGetPose(out Pose thumbPose) || !index.TryGetPose(out Pose indexPose))
                return false;

            return Vector3.Distance(thumbPose.position, indexPose.position) < pinchDistanceThreshold;
        }

        private bool TryGetAimRay(XRHandJointsUpdatedEventArgs args, out Vector3 origin, out Vector3 direction)
        {
            origin    = Vector3.zero;
            direction = Vector3.forward;

#if XR_HANDS_1_5_OR_NEWER
            if (args.subsystem != null)
            {
                var gestures = args.hand.handedness == Handedness.Left
                    ? args.subsystem.leftHandCommonGestures
                    : args.hand.handedness == Handedness.Right
                        ? args.subsystem.rightHandCommonGestures
                        : null;

                if (gestures != null && gestures.TryGetPinchPose(out Pose pinchPose))
                {
                    origin    = pinchPose.position;
                    direction = pinchPose.forward.sqrMagnitude > 1e-6f
                        ? pinchPose.forward
                        : pinchPose.rotation * Vector3.forward;
                    return true;
                }
            }
#endif

            var thumb = args.hand.GetJoint(XRHandJointID.ThumbTip);
            var index = args.hand.GetJoint(XRHandJointID.IndexTip);
            var wrist = args.hand.GetJoint(XRHandJointID.Wrist);

            if (!index.TryGetPose(out Pose indexPose))
                return false;

            origin = thumb.TryGetPose(out Pose thumbPose)
                ? Vector3.Lerp(thumbPose.position, indexPose.position, 0.5f)
                : indexPose.position;

            if (wrist.TryGetPose(out Pose wristPose))
            {
                direction = (indexPose.position - wristPose.position).normalized;
                if (direction.sqrMagnitude > 1e-6f)
                    return true;
            }

            direction = indexPose.rotation * Vector3.forward;
            return direction.sqrMagnitude > 1e-6f;
        }

        private NodeInteractable RaycastNode(Vector3 origin, Vector3 direction)
        {
            if (Physics.SphereCast(
                    origin,
                    raySphereRadius,
                    direction,
                    out RaycastHit hit,
                    selectRayLength,
                    interactableLayers,
                    QueryTriggerInteraction.Collide))
            {
                return hit.collider.GetComponentInParent<NodeInteractable>();
            }

            return null;
        }

        private NearFarInteractor GetInteractor(Handedness handedness)
        {
            if (graphRaycaster == null)
                return null;

            return handedness == Handedness.Left
                ? graphRaycaster.LeftNearFar
                : graphRaycaster.RightNearFar;
        }

        private void BeginPinchSelect(HandState state, NodeInteractable target)
        {
            if (target == null || interactionManager == null)
                return;

            var interactor = GetInteractor(state.Handedness) as IXRSelectInteractor;
            if (interactor == null)
                return;

            var interactable = (IXRSelectInteractable)target;
            if (interactor.isSelectActive)
            {
                foreach (var selected in interactor.interactablesSelected)
                {
                    if (selected == interactable)
                        return;
                }
            }

            interactionManager.SelectEnter(interactor, interactable);
            state.GestureSelected = interactable;
        }

        private void EndPinch(HandState state)
        {
            if (!state.WasPinching && state.GestureSelected == null)
                return;

            var interactor = GetInteractor(state.Handedness) as IXRSelectInteractor;
            if (interactor != null && state.GestureSelected != null && interactionManager != null)
                interactionManager.SelectExit(interactor, state.GestureSelected);

            state.GestureSelected = null;
            state.WasPinching   = false;
        }

        private void ReleaseAllSelections()
        {
            EndPinch(_left);
            EndPinch(_right);
        }

        private void SetHover(HandState state, NodeInteractable target)
        {
            var view = target != null ? target.GetComponent<NodeView>() : null;
            if (state.HoverView == view)
                return;

            state.HoverView?.SetHighlight(false);
            state.HoverView = view;
            state.HoverView?.SetHighlight(true);
        }

        private void ClearAllHovers()
        {
            SetHover(_left, null);
            SetHover(_right, null);
        }

        private sealed class HandState
        {
            public Handedness Handedness;
            public bool WasPinching;
            public IXRSelectInteractable GestureSelected;
            public NodeView HoverView;
            public UnityEngine.Events.UnityAction<XRHandJointsUpdatedEventArgs> OnJointsUpdated;
        }
    }
}
