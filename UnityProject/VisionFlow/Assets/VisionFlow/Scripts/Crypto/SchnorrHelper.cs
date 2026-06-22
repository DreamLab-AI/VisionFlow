using System;
using NBitcoin.Secp256k1;

namespace VisionFlow.Crypto
{
    internal static class SchnorrHelper
    {
        public static bool TryDerivePublicKeyHex(ReadOnlySpan<byte> privKey32, out string hexPubKey)
        {
            hexPubKey = null;
            if (privKey32.Length != 32)
                return false;

            if (!Context.Instance.TryCreateECPrivKey(privKey32, out var key))
                return false;

            using (key)
            {
                hexPubKey = BytesToHex(key.CreateXOnlyPubKey().ToBytes());
                return true;
            }
        }

        public static string DerivePublicKeyHex(byte[] privKey32)
        {
            if (!TryDerivePublicKeyHex(privKey32, out var hex))
                throw new InvalidOperationException("Invalid secp256k1 private key (zero or out of range).");
            return hex;
        }

        /// <summary>Sign a 32-byte message hash (Nostr event id) with BIP-340 Schnorr.</summary>
        public static string SignBip340Hex(byte[] privKey32, ReadOnlySpan<byte> messageHash32)
        {
            if (messageHash32.Length != 32)
                throw new ArgumentException("Message hash must be 32 bytes.", nameof(messageHash32));

            if (!Context.Instance.TryCreateECPrivKey(privKey32, out var key))
                throw new InvalidOperationException("Invalid secp256k1 private key.");

            using (key)
                return BytesToHex(key.SignBIP340(messageHash32).ToBytes());
        }

        public static bool VerifyBip340(string hexPubKey, ReadOnlySpan<byte> messageHash32, string hexSig)
        {
            if (messageHash32.Length != 32)
                return false;

            var pubBytes  = HexToBytes(hexPubKey);
            var sigBytes  = HexToBytes(hexSig);
            if (pubBytes.Length != 32 || sigBytes.Length != 64)
                return false;

            if (!Context.Instance.TryCreateXOnlyPubKey(pubBytes, out var xonly))
                return false;

            if (!SecpSchnorrSignature.TryCreate(sigBytes, out var sig))
                return false;

            return xonly.SigVerifyBIP340(sig, messageHash32);
        }

        public static byte[] HexToBytes(string hex)
        {
            if (hex == null)
                throw new ArgumentNullException(nameof(hex));
            if (hex.Length % 2 != 0)
                throw new ArgumentException("Hex string must have even length.", nameof(hex));

            var bytes = new byte[hex.Length / 2];
            for (int i = 0; i < bytes.Length; i++)
                bytes[i] = Convert.ToByte(hex.Substring(i * 2, 2), 16);
            return bytes;
        }

        public static string BytesToHex(ReadOnlySpan<byte> bytes)
        {
            var chars = new char[bytes.Length * 2];
            for (int i = 0; i < bytes.Length; i++)
            {
                byte b = bytes[i];
                chars[i * 2]     = GetHexNibble(b >> 4);
                chars[i * 2 + 1] = GetHexNibble(b & 0xF);
            }
            return new string(chars);
        }

        private static char GetHexNibble(int v) =>
            (char)(v < 10 ? '0' + v : 'a' + (v - 10));
    }
}
