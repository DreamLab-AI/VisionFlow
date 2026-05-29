using System.Collections.Generic;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace VisionFlow.Network
{
    /// <summary>NIP-01 Nostr event envelope from VisionClaw <c>nostr.event</c> notifications.</summary>
    public sealed class NostrEvent
    {
        [JsonProperty("id")]         public string Id         { get; set; }
        [JsonProperty("pubkey")]     public string Pubkey     { get; set; }
        [JsonProperty("created_at")] public long   CreatedAt { get; set; }
        [JsonProperty("kind")]       public int    Kind       { get; set; }
        [JsonProperty("tags")]       public List<List<string>> Tags { get; set; }
        [JsonProperty("content")]    public string Content    { get; set; }
        [JsonProperty("sig")]        public string Sig        { get; set; }

        public string GetTag(string name)
        {
            if (Tags == null) return null;
            foreach (var tag in Tags)
            {
                if (tag == null || tag.Count == 0) continue;
                if (tag[0] == name && tag.Count > 1)
                    return tag[1];
            }
            return null;
        }

        /// <summary>Graph node ids referenced by Nostr tags (e, node, entity, graph, …).</summary>
        public void CollectReferencedNodeIds(List<string> into)
        {
            if (into == null || Tags == null) return;

            foreach (var tag in Tags)
            {
                if (tag == null || tag.Count < 2) continue;
                string key = tag[0];
                if (key is not ("e" or "node" or "entity" or "graph" or "subject" or "object"))
                    continue;

                string value = tag[1];
                if (!string.IsNullOrEmpty(value) && !into.Contains(value))
                    into.Add(value);
            }
        }

        public static Color KindColor(int kind) => kind switch
        {
            30001 => new Color(0.2f, 0.85f, 0.55f),
            31402 => new Color(1f, 0.55f, 0.2f),
            31403 => new Color(0.45f, 0.65f, 1f),
            _     => new Color(0.75f, 0.75f, 0.8f),
        };

        public static string KindLabel(int kind) => kind switch
        {
            30001 => "Provenance bead",
            31402 => "Governance request",
            31403 => "Governance decision",
            _     => $"Kind {kind}",
        };
    }

    /// <summary>Agent Control Surface governance request (kind 31402).</summary>
    public sealed class GovernanceRequest
    {
        public string EventId   { get; set; }
        public string Pubkey    { get; set; }
        public string AgentDid  { get; set; }
        public string Action    { get; set; }
        public string Context   { get; set; }
        public long   CreatedAt { get; set; }

        public static bool TryFromNostrEvent(NostrEvent ev, out GovernanceRequest request)
        {
            request = null;
            if (ev == null || ev.Kind != 31402 || string.IsNullOrEmpty(ev.Id))
                return false;

            request = new GovernanceRequest
            {
                EventId   = ev.Id,
                Pubkey    = ev.Pubkey,
                AgentDid  = ev.GetTag("agent") ?? $"did:nostr:{ev.Pubkey}",
                Action    = ev.GetTag("action") ?? "—",
                Context   = ev.Content ?? string.Empty,
                CreatedAt = ev.CreatedAt,
            };
            return true;
        }
    }
}
