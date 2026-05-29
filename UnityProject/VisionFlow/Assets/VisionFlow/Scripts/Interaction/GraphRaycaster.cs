using UnityEngine;
using UnityEngine.Serialization;
using UnityEngine.XR.Interaction.Toolkit;
using UnityEngine.XR.Interaction.Toolkit.Interactors;
using Unity.XR.CoreUtils;

namespace VisionFlow.Interaction
{
    public sealed class GraphRaycaster : MonoBehaviour
    {
        [SerializeField] private XROrigin xrOrigin;

        [FormerlySerializedAs("leftRay")]
        [SerializeField] private NearFarInteractor leftNearFar;

        [FormerlySerializedAs("rightRay")]
        [SerializeField] private NearFarInteractor rightNearFar;

        public NearFarInteractor LeftNearFar  => leftNearFar;
        public NearFarInteractor RightNearFar => rightNearFar;

        private void Awake()
        {
            if (FindAnyObjectByType<XRInteractionManager>() == null)
            {
                Debug.LogWarning("[GraphRaycaster] No XRInteractionManager in scene.");
                return;
            }

            if (xrOrigin == null)
                xrOrigin = FindAnyObjectByType<XROrigin>();

            if (leftNearFar == null || rightNearFar == null)
                CacheNearFarInteractors();

            LogSetupState();
        }

        private void CacheNearFarInteractors()
        {
            if (xrOrigin == null)
            {
                foreach (var nf in FindObjectsByType<NearFarInteractor>(FindObjectsInactive.Include))
                    AssignByHand(nf);
                return;
            }

            foreach (var nf in xrOrigin.GetComponentsInChildren<NearFarInteractor>(true))
                AssignByHand(nf);

            if (leftNearFar == null || rightNearFar == null)
            {
                var all = xrOrigin.GetComponentsInChildren<NearFarInteractor>(true);
                if (leftNearFar == null && all.Length > 0)  leftNearFar  = all[0];
                if (rightNearFar == null && all.Length > 1) rightNearFar = all[1];
            }
        }

        private void AssignByHand(NearFarInteractor nf)
        {
            var side = nf.handedness switch
            {
                InteractorHandedness.Left  => 0,
                InteractorHandedness.Right => 1,
                _ => -1,
            };

            if (side < 0)
            {
                var name = nf.gameObject.name.ToLowerInvariant();
                if (name.Contains("left"))  side = 0;
                if (name.Contains("right")) side = 1;
            }

            if (side == 0 && leftNearFar == null)
                leftNearFar = nf;
            else if (side == 1 && rightNearFar == null)
                rightNearFar = nf;
        }

        private void LogSetupState()
        {
            if (leftNearFar != null && rightNearFar != null)
            {
                Debug.Log($"[GraphRaycaster] NearFar left={leftNearFar.name}, right={rightNearFar.name}");
                return;
            }

            if (leftNearFar != null || rightNearFar != null)
            {
                Debug.Log("[GraphRaycaster] One NearFarInteractor wired — graph select OK on that controller.");
                return;
            }

            Debug.LogWarning("[GraphRaycaster] No NearFarInteractor on XR rig. Menu: GameObject → VisionFlow → Wire Near-Far Interactors.");
        }
    }
}
