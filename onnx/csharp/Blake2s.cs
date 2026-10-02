using System;
using System.Text;

namespace IndoNameGender
{
    /// <summary>Unkeyed BLAKE2s (RFC 7693), enough for the 8-byte word-vocab keys.</summary>
    public static class Blake2s
    {
        static readonly uint[] IV =
        {
            0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
            0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
        };

        static readonly byte[][] Sigma =
        {
            new byte[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
            new byte[] { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
            new byte[] { 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
            new byte[] { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
            new byte[] { 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
            new byte[] { 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
            new byte[] { 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
            new byte[] { 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
            new byte[] { 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
            new byte[] { 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
        };

        static uint Rotr(uint x, int n) { return (x >> n) | (x << (32 - n)); }

        static void G(uint[] v, int a, int b, int c, int d, uint x, uint y)
        {
            v[a] = v[a] + v[b] + x; v[d] = Rotr(v[d] ^ v[a], 16);
            v[c] = v[c] + v[d];     v[b] = Rotr(v[b] ^ v[c], 12);
            v[a] = v[a] + v[b] + y; v[d] = Rotr(v[d] ^ v[a], 8);
            v[c] = v[c] + v[d];     v[b] = Rotr(v[b] ^ v[c], 7);
        }

        static void Compress(uint[] h, byte[] block, ulong t, bool last)
        {
            var m = new uint[16];
            for (int i = 0; i < 16; i++) m[i] = BitConverter.ToUInt32(block, i * 4);
            var v = new uint[16];
            Array.Copy(h, v, 8);
            Array.Copy(IV, 0, v, 8, 8);
            v[12] ^= (uint)t;
            v[13] ^= (uint)(t >> 32);
            if (last) v[14] = ~v[14];
            for (int r = 0; r < 10; r++)
            {
                var s = Sigma[r];
                G(v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
                G(v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
                G(v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
                G(v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
                G(v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
                G(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
                G(v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
                G(v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
            }
            for (int i = 0; i < 8; i++) h[i] ^= v[i] ^ v[i + 8];
        }

        /// <summary>Returns the first <paramref name="digestSize"/> bytes of the digest.</summary>
        public static byte[] Hash(byte[] data, int digestSize)
        {
            var h = new uint[8];
            Array.Copy(IV, h, 8);
            h[0] ^= 0x01010000u ^ (uint)digestSize;   // fanout 1, depth 1, no key

            int offset = 0;
            while (data.Length - offset > 64)
            {
                var block = new byte[64];
                Array.Copy(data, offset, block, 0, 64);
                offset += 64;
                Compress(h, block, (ulong)offset, false);
            }
            var tail = new byte[64];
            Array.Copy(data, offset, tail, 0, data.Length - offset);
            Compress(h, tail, (ulong)data.Length, true);

            var full = new byte[32];
            for (int i = 0; i < 8; i++) BitConverter.GetBytes(h[i]).CopyTo(full, i * 4);
            var outBytes = new byte[digestSize];
            Array.Copy(full, outBytes, digestSize);
            return outBytes;
        }

        public static string HexDigest(string text, int digestSize)
        {
            var bytes = Hash(Encoding.UTF8.GetBytes(text), digestSize);
            var sb = new StringBuilder(bytes.Length * 2);
            foreach (var b in bytes) sb.Append(b.ToString("x2"));
            return sb.ToString();
        }
    }
}
