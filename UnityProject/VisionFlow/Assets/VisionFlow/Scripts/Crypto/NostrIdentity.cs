using System;
using System.Security.Cryptography;
using System.Text;
using VisionFlow.Graph;
using UnityEngine;

namespace VisionFlow.Crypto
{
    public sealed class NostrIdentity : IDisposable
    {
        private const string PrefKey = "vf_nostr_privkey_b64";
        public string HexPubKey { get; private set; }
        public string Did => $"did:nostr:{HexPubKey}";
        private byte[] _privKeyBytes;
        private static NostrIdentity _instance;
        public static NostrIdentity Instance => _instance ??= new NostrIdentity();
        private NostrIdentity() => Load();

        private void Load()
        {
            if (PlayerPrefs.HasKey(PrefKey))
            {
                _privKeyBytes = Convert.FromBase64String(PlayerPrefs.GetString(PrefKey));
                if (!TryApplyPrivateKey(_privKeyBytes))
                {
                    Debug.LogWarning("[NostrIdentity] Stored key invalid after crypto upgrade — generating new identity.");
                    PlayerPrefs.DeleteKey(PrefKey);
                    Generate();
                    return;
                }
                Debug.Log($"[NostrIdentity] Loaded existing identity: {Did}");
            }
            else
            {
                Generate();
            }
        }

        private void Generate()
        {
            _privKeyBytes = CreateValidPrivateKey();
            HexPubKey     = SchnorrHelper.DerivePublicKeyHex(_privKeyBytes);
            PlayerPrefs.SetString(PrefKey, Convert.ToBase64String(_privKeyBytes));
            PlayerPrefs.Save();
            Debug.Log($"[NostrIdentity] Generated new identity: {Did}");
        }

        public void Reset()
        {
            PlayerPrefs.DeleteKey(PrefKey);
            Generate();
        }

        public string BuildNip98Header(string url, string httpMethod, string bodyJson = null)
        {
            long   createdAt  = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
            string payloadHex = bodyJson != null ? Sha256Hex(bodyJson) : string.Empty;

            var ev = new
            {
                pubkey     = HexPubKey,
                created_at = createdAt,
                kind       = 27235,
                tags       = new[]
                {
                    new[] { "u",       url        },
                    new[] { "method",  httpMethod },
                    new[] { "payload", payloadHex },
                },
                content = string.Empty,
            };

            string evJson  = VisionFlowJson.Serialize(ev);
            string evId    = Sha256Hex(evJson);
            byte[] evIdBytes = SchnorrHelper.HexToBytes(evId);
            string sig     = SchnorrHelper.SignBip340Hex(_privKeyBytes, evIdBytes);

            var signedEv = new
            {
                id         = evId,
                pubkey     = HexPubKey,
                created_at = createdAt,
                kind       = 27235,
                tags       = ev.tags,
                content    = string.Empty,
                sig        = sig,
            };

            string signedJson = VisionFlowJson.Serialize(signedEv);
            return "Nostr " + Convert.ToBase64String(Encoding.UTF8.GetBytes(signedJson));
        }

        private bool TryApplyPrivateKey(byte[] privKey)
        {
            if (!SchnorrHelper.TryDerivePublicKeyHex(privKey, out var hex))
                return false;
            _privKeyBytes = privKey;
            HexPubKey     = hex;
            return true;
        }

        private static byte[] CreateValidPrivateKey()
        {
            var sk = new byte[32];
            while (true)
            {
                RandomNumberGenerator.Fill(sk);
                if (SchnorrHelper.TryDerivePublicKeyHex(sk, out _))
                    return sk;
            }
        }

        private static string Sha256Hex(string input)
        {
            using var sha256 = SHA256.Create();
            byte[] bytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(input));
            return SchnorrHelper.BytesToHex(bytes);
        }
        public void Dispose()
        {
            if (_privKeyBytes != null)
                Array.Clear(_privKeyBytes, 0, _privKeyBytes.Length);
        }
    }
}
