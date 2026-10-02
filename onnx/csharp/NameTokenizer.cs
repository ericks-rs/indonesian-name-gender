using System;
using System.Collections.Generic;
using System.IO;
using System.Web.Script.Serialization;

namespace IndoNameGender
{
    /// <summary>
    /// Turns a name into the fixed-length id array the ONNX models expect.
    /// Mirrors CharTokenizer / WordTokenizer in demo/inference.py: lowercase, then one id per
    /// character (Char models) or one id per whitespace-separated word (Word models), unknown
    /// items map to id 1, and the array is cut or zero-padded to max_len.
    /// </summary>
    public class NameTokenizer
    {
        readonly bool _isChar;
        readonly Dictionary<string, int> _vocab;
        readonly int _maxLen;
        readonly int _digestSize;
        readonly string _salt;
        readonly bool _hashed;

        public int MaxLen { get { return _maxLen; } }

        NameTokenizer(bool isChar, Dictionary<string, int> vocab, int maxLen,
                      int digestSize, string salt, bool hashed)
        {
            _isChar = isChar; _vocab = vocab; _maxLen = maxLen;
            _digestSize = digestSize; _salt = salt; _hashed = hashed;
        }

        /// <param name="jsonPath">char_vocab.json or word_vocab.json</param>
        public static NameTokenizer Load(string jsonPath, bool isChar)
        {
            var js = new JavaScriptSerializer { MaxJsonLength = int.MaxValue };
            var root = js.Deserialize<Dictionary<string, object>>(File.ReadAllText(jsonPath));
            var raw = (Dictionary<string, object>)root[isChar ? "char2idx" : "word2idx"];
            var vocab = new Dictionary<string, int>(raw.Count);
            foreach (var kv in raw) vocab[kv.Key] = Convert.ToInt32(kv.Value);
            int maxLen = Convert.ToInt32(root["max_len"]);
            if (isChar) return new NameTokenizer(true, vocab, maxLen, 0, null, false);
            return new NameTokenizer(false, vocab, maxLen, Convert.ToInt32(root["digest_size"]),
                                     (string)root["salt"], (bool)root["hashed"]);
        }

        string Key(string word)
        {
            if (!_hashed) return word;
            if (word.StartsWith("<") && word.EndsWith(">")) return word;
            return Blake2s.HexDigest(_salt + word, _digestSize);
        }

        public long[] Encode(string name)
        {
            var ids = new List<long>();
            var lower = (name ?? "").ToLowerInvariant();
            if (_isChar)
            {
                foreach (char ch in lower)
                {
                    int id;
                    ids.Add(_vocab.TryGetValue(ch.ToString(), out id) ? id : 1);
                }
            }
            else
            {
                foreach (var w in lower.Split((char[])null, StringSplitOptions.RemoveEmptyEntries))
                {
                    int id;
                    ids.Add(_vocab.TryGetValue(Key(w), out id) ? id : 1);
                }
            }
            var result = new long[_maxLen];                    // zero = padding
            for (int i = 0; i < Math.Min(ids.Count, _maxLen); i++) result[i] = ids[i];
            return result;
        }
    }
}
