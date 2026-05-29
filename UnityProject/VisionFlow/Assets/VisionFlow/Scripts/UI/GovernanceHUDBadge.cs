using TMPro;
using UnityEngine;

namespace VisionFlow.UI
{
    /// <summary>Persistent queue count for inbound governance requests (kind 31402).</summary>
    public sealed class GovernanceHUDBadge : MonoBehaviour
    {
        [SerializeField] private TextMeshProUGUI label;
        [SerializeField] private GameObject      badgeRoot;

        public void SetPendingCount(int count)
        {
            if (badgeRoot != null)
                badgeRoot.SetActive(count > 0);

            if (label != null)
                label.text = count > 0 ? count.ToString() : string.Empty;
        }
    }
}
