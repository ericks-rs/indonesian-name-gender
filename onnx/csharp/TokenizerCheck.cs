using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Web.Script.Serialization;
using IndoNameGender;

// Checks the C# tokenizer and BLAKE2s against what Python produced (parity_cases.json and
// hash_cases.json). Needs no ONNX Runtime. Usage: TokenizerCheck.exe <path to onnx/models>
static class TokenizerCheck
{
    static int Main(string[] args)
    {
        string dir = args.Length > 0 ? args[0] : "models";
        var js = new JavaScriptSerializer { MaxJsonLength = int.MaxValue };
        int bad = 0, total = 0;

        var hashes = js.Deserialize<List<Dictionary<string, object>>>(
            File.ReadAllText(Path.Combine(dir, "hash_cases.json")));
        foreach (var h in hashes)
        {
            string word = (string)h["word"], want = (string)h["key"];
            string got = word.StartsWith("<") && word.EndsWith(">") ? word : Blake2s.HexDigest("indonamegender-v1" + word, 8);
            total++;
            if (got != want) { bad++; Console.WriteLine("HASH MISMATCH for length-" + word.Length + " word"); }
        }

        var cases = js.Deserialize<Dictionary<string, Dictionary<string, Dictionary<string, object>>>>(
            File.ReadAllText(Path.Combine(dir, "parity_cases.json")));
        var tokenizers = new Dictionary<string, NameTokenizer>
        {
            { "Char", NameTokenizer.Load(Path.Combine(dir, "char_vocab.json"), true) },
            { "Word", NameTokenizer.Load(Path.Combine(dir, "word_vocab.json"), false) },
        };
        foreach (var name in cases)
            foreach (var model in name.Value)
            {
                var want = ((System.Collections.ArrayList)model.Value["ids"]).Cast<object>()
                           .Select(Convert.ToInt64).ToArray();
                var got = tokenizers[model.Key.StartsWith("Char") ? "Char" : "Word"].Encode(name.Key);
                total++;
                if (!want.SequenceEqual(got)) { bad++; Console.WriteLine("IDS MISMATCH " + name.Key + " / " + model.Key); }
            }

        Console.WriteLine(bad == 0 ? "OK: " + total + " checks match Python" : "FAILED: " + bad + " of " + total);
        return bad == 0 ? 0 : 1;
    }
}
