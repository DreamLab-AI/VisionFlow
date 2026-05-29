using System;
using System.Net.Http;
using System.Text;
using Newtonsoft.Json;
using System.Threading.Tasks;
using UnityEngine;
using VisionFlow.Crypto;
using VisionFlow.Graph;

namespace VisionFlow.Network
{
    public sealed class SolidPodClient : IDisposable
    {
        // ── State ──────────────────────────────────────────────────────────

        private readonly HttpClient     _http;
        private readonly NostrIdentity  _identity;
        public  readonly string         BaseUrl;

        // ── Constructor ────────────────────────────────────────────────────

        public SolidPodClient(string baseUrl, NostrIdentity identity = null)
        {
            BaseUrl   = baseUrl.TrimEnd('/');
            _identity = identity ?? NostrIdentity.Instance;
            _http     = new HttpClient { Timeout = TimeSpan.FromSeconds(10) };
        }

        // ── LDP read ───────────────────────────────────────────────────────

        /// <summary>GET a resource from the Solid pod as raw string.</summary>
        public async Task<string> GetAsync(string path)
        {
            string url = $"{BaseUrl}{path}";
            using var req = new HttpRequestMessage(HttpMethod.Get, url);
            req.Headers.TryAddWithoutValidation("Authorization",
                _identity.BuildNip98Header(url, "GET"));
            req.Headers.TryAddWithoutValidation("Accept", "application/ld+json, text/turtle;q=0.9");

            using var res = await _http.SendAsync(req);
            res.EnsureSuccessStatusCode();
            return await res.Content.ReadAsStringAsync();
        }

        /// <summary>GET and deserialise to T using System.Text.Json.</summary>
        public async Task<T> GetJsonAsync<T>(string path)
        {
            string json = await GetAsync(path);
            return VisionFlowJson.Deserialize<T>(json);
        }

        // ── LDP write ──────────────────────────────────────────────────────

        /// <summary>PUT a JSON-LD document to the pod.</summary>
        public async Task<bool> PutJsonAsync(string path, object payload)
        {
            string url  = $"{BaseUrl}{path}";
            string json = VisionFlowJson.Serialize(payload);

            using var req = new HttpRequestMessage(HttpMethod.Put, url);
            req.Headers.TryAddWithoutValidation("Authorization",
                _identity.BuildNip98Header(url, "PUT", json));
            req.Content = new StringContent(json, Encoding.UTF8, "application/ld+json");

            using var res = await _http.SendAsync(req);
            return res.IsSuccessStatusCode;
        }

        /// <summary>PATCH (partial update) via N3 Patch or JSON Merge Patch.</summary>
        public async Task<bool> PatchJsonAsync(string path, object patchPayload)
        {
            string url  = $"{BaseUrl}{path}";
            string json = VisionFlowJson.Serialize(patchPayload);

            using var req = new HttpRequestMessage(HttpMethod.Patch, url);
            req.Headers.TryAddWithoutValidation("Authorization",
                _identity.BuildNip98Header(url, "PATCH", json));
            req.Content = new StringContent(json, Encoding.UTF8, "application/merge-patch+json");

            using var res = await _http.SendAsync(req);
            return res.IsSuccessStatusCode;
        }

        /// <summary>DELETE a resource.</summary>
        public async Task<bool> DeleteAsync(string path)
        {
            string url = $"{BaseUrl}{path}";
            using var req = new HttpRequestMessage(HttpMethod.Delete, url);
            req.Headers.TryAddWithoutValidation("Authorization",
                _identity.BuildNip98Header(url, "DELETE"));

            using var res = await _http.SendAsync(req);
            return res.IsSuccessStatusCode;
        }

        // ── Provenance bead (Nostr kind 30001) ────────────────────────────

        /// <summary>
        /// Writes an immutable provenance bead to the relay via the pod's
        /// /nostr/publish endpoint (agentbox relay bridge).
        /// </summary>
        public async Task<bool> WriteProvenanceBeadAsync(
            string subjectId, string predicateUri, string objectValue, string[] ancestorBeadIds = null)
        {
            var bead = new
            {
                kind   = 30001,
                pubkey = _identity.HexPubKey,
                tags   = new[]
                {
                    new[] { "subject",   subjectId    },
                    new[] { "predicate", predicateUri },
                    new[] { "object",    objectValue  },
                },
                content      = objectValue,
                created_at   = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                prior_bead_ids = ancestorBeadIds ?? Array.Empty<string>(),
            };

            return await PutJsonAsync("/nostr/publish", bead);
        }

        // ── WebID profile ─────────────────────────────────────────────────

        /// <summary>Resolves this agent's WebID profile from the pod.</summary>
        public Task<string> GetWebIdProfileAsync() =>
            GetAsync($"/profile/card");

        // ── IDisposable ────────────────────────────────────────────────────

        public void Dispose() => _http.Dispose();
    }
}
