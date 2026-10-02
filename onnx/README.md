# ONNX export

The eight grid models (seed 42) exported to ONNX, so they can run without PyTorch, for example
from .NET.

| File | Size |
|---|---|
| `CharBiRNN`, `CharBiLSTM`, `CharBiGRU`, `CharTransformer` | 0.1 to 2.5 MB each |
| `WordBiRNN`, `WordBiLSTM`, `WordBiGRU`, `WordTransformer` | 9.7 to 24.6 MB each |

The `.onnx` files are attached to the GitHub Release `v1.1.0`, not stored in the repository.
Download them with

```bash
python onnx/download_models.py            # all eight
python onnx/download_models.py CharBiLSTM # one model
```

The tokenizer files (`models/char_vocab.json`, `models/word_vocab.json`) are in the repository.

## Model interface

| | Name | Type | Shape |
|---|---|---|---|
| input | `ids` | int64 | `[batch, max_len]`, 0 is padding, 1 is unknown |
| output | `prob_female` | float32 | `[batch]`, probability of class P |
| output | `attention` | float32 | `[batch, max_len]`, attention-pooling weights, 0 on padding |

`max_len` is 50 for the Char models (one id per character) and 8 for the Word models (one id per
word). A name is lowercased first. Word ids come from a hashed vocabulary: the key of a word is
the first 8 bytes of `BLAKE2s(salt + word)` as hex, with the salt given in `word_vocab.json`.
Words not in the vocabulary map to 1.

On 2,010 names the ONNX outputs differ from PyTorch by at most 1.3e-06 and no label changes
(`export_onnx.py` prints this check).

## Python

```python
import json, numpy as np, onnxruntime as ort

vocab = json.load(open("onnx/models/char_vocab.json", encoding="utf-8"))
ids = [vocab["char2idx"].get(c, 1) for c in "banowati larasati"][:vocab["max_len"]]
ids += [0] * (vocab["max_len"] - len(ids))

sess = ort.InferenceSession("onnx/models/CharBiLSTM.onnx")
prob_female, attention = sess.run(None, {"ids": np.array([ids], dtype=np.int64)})
print(prob_female[0])
```

## C# (.NET)

`csharp/` holds four files with no dependency except ONNX Runtime. They build for .NET 6 and
later (ASP.NET Core) and for .NET Framework 4.x.

| File | Purpose |
|---|---|
| `GenderClassifier.cs` | loads a model, `Predict(name)` returns label, probability, attention |
| `NameTokenizer.cs` | name to ids, same rules as the Python tokenizers |
| `Blake2s.cs` | BLAKE2s for the Word vocabulary keys (.NET has none built in) |
| `TokenizerCheck.cs` | console test that compares the C# ids and hashes with `parity_cases.json` and `hash_cases.json` |

### ASP.NET Core Web API

1. `dotnet add package Microsoft.ML.OnnxRuntime`.
2. Copy `Blake2s.cs`, `NameTokenizer.cs` and `GenderClassifier.cs` into the project, and copy the
   `.onnx` files and the two `*_vocab.json` files to a `models` folder. Mark them
   `Copy to Output Directory` or resolve the folder from `ContentRootPath`.
3. Register one `GenderClassifier` as a singleton. `InferenceSession.Run` is thread-safe, and the
   container disposes the instance at shutdown. Do not create one per request.

`Program.cs` (minimal API):

```csharp
using IndoNameGender;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddSingleton(_ => new GenderClassifier(
    Path.Combine(builder.Environment.ContentRootPath, "models"), "CharBiLSTM"));
var app = builder.Build();

app.MapGet("/api/gender", (string name, GenderClassifier classifier) =>
{
    if (string.IsNullOrWhiteSpace(name) || name.Length > 200)
        return Results.BadRequest("name is required and at most 200 characters");
    var p = classifier.Predict(name);
    return Results.Ok(new { p.Label, p.ProbFemale, p.Confidence });
});

app.Run();
```

The 64-bit requirement of the old .NET Framework note does not apply here, the NuGet package
carries the native library for each platform (win-x64, linux-x64, osx-arm64 and others). Only an
x86 runtime would need a different setup.

### Tests

`GenderClassifier` was run on .NET 9 with ONNX Runtime 1.x: all eight models, ten names each
(80 predictions), and the largest difference from the PyTorch probability was 3e-07. The tokenizer
and BLAKE2s were checked against the Python output (131 checks, all match, including words
longer than one 64-byte block and non-ASCII text). The `Program.cs` snippet above is not part of
that run.

To repeat the tokenizer check without ONNX Runtime, create a console project that includes
`Blake2s.cs`, `NameTokenizer.cs` and `TokenizerCheck.cs`, then run it with the path to
`onnx/models`. On .NET Framework, compile with
`csc -r:System.Web.Extensions.dll Blake2s.cs NameTokenizer.cs TokenizerCheck.cs`.

### .NET Framework 4.x (Web Forms)

The same files work. `NameTokenizer.cs` switches to `JavaScriptSerializer`, so add a reference to
`System.Web.Extensions`. ONNX Runtime is native, so the application pool must be 64-bit
(IIS, Enable 32-Bit Applications = False) and the project must not use "Prefer 32-bit". Create the
`GenderClassifier` once in `Application_Start` and dispose it in `Application_End`.
`GenderClassifier.cs` was not run on .NET Framework, only the tokenizer files were.

### Notes

- Characters outside the Basic Multilingual Plane (for example emoji) are two UTF-16 units in
  .NET and one code point in Python, so such input can tokenize differently. Names do not
  contain them in practice.
- Names are personal data. Keep them out of logs and exception messages.
- `Predict` is a per-name call. To score many names at once, build one `[n, max_len]` tensor and
  read `n` outputs.

## Re-export

```bash
python onnx/export_onnx.py
```

It needs `torch`, `onnx`, `onnxruntime` and `onnxscript`, the checkpoints in `models/` and the
tokenizers in `tokenizers/`. If `data/splits/val_2024_2026.csv` exists locally, 2,000 of its names
are used for the parity check, otherwise only the ten synthetic names are used.
