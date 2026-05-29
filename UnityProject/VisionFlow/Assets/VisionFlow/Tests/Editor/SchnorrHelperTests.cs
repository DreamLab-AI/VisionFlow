using System;
using System.Security.Cryptography;
using NUnit.Framework;
using VisionFlow.Crypto;

namespace VisionFlow.Tests.Editor
{
    public sealed class SchnorrHelperTests
    {
        [Test]
        public void SignAndVerify_RoundTrip()
        {
            byte[] priv = CreateValidPrivateKey();
            string pub  = SchnorrHelper.DerivePublicKeyHex(priv);

            byte[] msg = new byte[32];
            RandomNumberGenerator.Fill(msg);

            string sig = SchnorrHelper.SignBip340Hex(priv, msg);
            Assert.IsTrue(SchnorrHelper.VerifyBip340(pub, msg, sig),
                "BIP-340 verify should accept signature from matching key.");
        }

        [Test]
        public void Sign_KnownMessage_MatchesPubKey()
        {
            byte[] priv = CreateValidPrivateKey();
            string pub  = SchnorrHelper.DerivePublicKeyHex(priv);

            // Fixed 32-byte hash (not a Nostr wire vector — local sanity check only).
            byte[] msg = SchnorrHelper.HexToBytes(
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");

            string sig = SchnorrHelper.SignBip340Hex(priv, msg);
            Assert.AreEqual(128, sig.Length);
            Assert.IsTrue(SchnorrHelper.VerifyBip340(pub, msg, sig));
        }

        [Test]
        public void DerivePublicKey_Is32ByteXOnlyHex()
        {
            byte[] priv = CreateValidPrivateKey();
            string pub  = SchnorrHelper.DerivePublicKeyHex(priv);
            Assert.AreEqual(64, pub.Length);
            Assert.IsTrue(SchnorrHelper.TryDerivePublicKeyHex(priv, out var again));
            Assert.AreEqual(pub, again);
        }

        private static byte[] CreateValidPrivateKey()
        {
            var sk = new byte[32];
            do RandomNumberGenerator.Fill(sk);
            while (!SchnorrHelper.TryDerivePublicKeyHex(sk, out _));
            return sk;
        }
    }
}
