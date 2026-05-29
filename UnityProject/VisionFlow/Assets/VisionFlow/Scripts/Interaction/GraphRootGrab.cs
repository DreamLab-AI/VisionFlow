using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit.Interactables;

namespace VisionFlow.Interaction
{
    [RequireComponent(typeof(Rigidbody))]
    public sealed class GraphRootGrab : XRGrabInteractable
    {
        protected override void Awake()
        {
            var rb = GetComponent<Rigidbody>();
            rb.isKinematic = true;
            rb.useGravity  = false;
            base.Awake();
        }
    }
}
