using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
#if NET
using System.Text.Json;
#else
using System.Web.Script.Serialization;
#endif
using IndoNameGender;

// Checks the C# tokenizer and BLAKE2s against what Python produced (parity_cases.json and
// hash_cases.json). Needs no ONNX Runtime. Usage: TokenizerCheck <path to onnx/models>
static class TokenizerCheck
{
    static int Main(string[] args)
    {
        string dir = args.Length > 0 ? args[0] : "models";
        var hashes = new List<KeyValuePair<string, string>>();            // word, expected key
        var cases = new List<KeyValuePair<string, KeyValuePair<string, long[]>>>(); // name, (model, ids)

#if NET
        using (var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "hash_cases.json"))))
            foreach (var h in doc.RootElement.EnumerateArray())
                hashes.Add(new KeyValuePair<string, string>(h.GetProperty("word").GetString(),
                                                            h.GetProperty("key").GetString()));
        using (var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "parity_cases.json"))))
            foreach (var name in doc.RootElement.EnumerateObject())
                foreach (var model in name.Value.EnumerateObject())
                    cases.Add(new KeyValuePair<string, KeyValuePair<string, long[]>>(name.Name,
                        new KeyValuePair<string, long[]>(model.Name,
                            model.Value.GetProperty("ids").EnumerateArray().Select(e => e.GetInt64()).ToArray())));
#else
        var js = new JavaScriptSerializer { MaxJsonLength = int.MaxValue };
        foreach (var h in js.Deserialize<List<Dictionary<string, object>>>(
                     File.ReadAllText(Path.Combine(dir, "hash_cases.json"))))
            hashes.Add(new KeyValuePair<string, string>((string)h["word"], (string)h["key"]));
        var parsed = js.Deserialize<Dictionary<string, Dictionary<string, Dictionary<string, object>>>>(
            File.ReadAllText(Path.Combine(dir, "parity_cases.json")));
        foreach (var name in parsed)
            foreach (var model in name.Value)
                cases.Add(new KeyValuePair<string, KeyValuePair<string, long[]>>(name.Key,
                    new KeyValuePair<string, long[]>(model.Key,
                        ((System.Collections.ArrayList)model.Value["ids"]).Cast<object>()
                            .Select(Convert.ToInt64).ToArray())));
#endif

        int bad = 0, total = 0;
        foreach (var h in hashes)
        {
            string got = h.Key.StartsWith("<") && h.Key.EndsWith(">")
                ? h.Key : Blake2s.HexDigest("indonamegender-v1" + h.Key, 8);
            total++;
            if (got != h.Value) { bad++; Console.WriteLine("HASH MISMATCH for length-" + h.Key.Length + " word"); }
        }

        var tokenizers = new Dictionary<string, NameTokenizer>
        {
            { "Char", NameTokenizer.Load(Path.Combine(dir, "char_vocab.json"), true) },
            { "Word", NameTokenizer.Load(Path.Combine(dir, "word_vocab.json"), false) },
        };
        foreach (var c in cases)
        {
            var got = tokenizers[c.Value.Key.StartsWith("Char") ? "Char" : "Word"].Encode(c.Key);
            total++;
            if (!c.Value.Value.SequenceEqual(got)) { bad++; Console.WriteLine("IDS MISMATCH " + c.Key + " / " + c.Value.Key); }
        }

        Console.WriteLine(bad == 0 ? "OK: " + total + " checks match Python" : "FAILED: " + bad + " of " + total);
        return bad == 0 ? 0 : 1;
    }
}
