using System;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using VisionFlow.Network;

namespace VisionFlow.UI
{
    public sealed class EventFeedItem : MonoBehaviour
    {
        [SerializeField] private Image             stripe;
        [SerializeField] private TextMeshProUGUI   labelKind;
        [SerializeField] private TextMeshProUGUI   labelSummary;
        [SerializeField] private Button            btnSelect;

        public NostrEvent Event { get; private set; }

        public event Action<NostrEvent> OnSelected;

        private void Awake()
        {
            btnSelect?.onClick.AddListener(() =>
            {
                if (Event != null)
                    OnSelected?.Invoke(Event);
            });
        }

        public void Bind(NostrEvent ev)
        {
            Event = ev;
            if (ev == null)
            {
                gameObject.SetActive(false);
                return;
            }

            gameObject.SetActive(true);
            var color = NostrEvent.KindColor(ev.Kind);
            if (stripe != null)
                stripe.color = color;

            if (labelKind != null)
            {
                labelKind.text  = NostrEvent.KindLabel(ev.Kind);
                labelKind.color = color;
            }

            if (labelSummary != null)
            {
                string snippet = string.IsNullOrEmpty(ev.Content)
                    ? ev.Id
                    : ev.Content;
                if (snippet.Length > 72)
                    snippet = snippet.Substring(0, 69) + "…";
                labelSummary.text = snippet;
            }
        }
    }
}
