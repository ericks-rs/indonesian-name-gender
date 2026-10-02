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

## C# (ASP.NET Web Forms, .NET Framework 4.x)

`csharp/` holds four files with no dependency except ONNX Runtime:

| File | Purpose |
|---|---|
| `GenderClassifier.cs` | loads a model, `Predict(name)` returns label, probability, attention |
| `NameTokenizer.cs` | name to ids, same rules as the Python tokenizers |
| `Blake2s.cs` | BLAKE2s for the Word vocabulary keys (.NET Framework has none built in) |
| `TokenizerCheck.cs` | console test that compares the C# ids and hashes with `parity_cases.json` and `hash_cases.json` |

Setup:

1. Install the NuGet package `Microsoft.ML.OnnxRuntime` and add a reference to
   `System.Web.Extensions`.
2. ONNX Runtime is a native library, so the application pool must be **64-bit**
   (IIS: Advanced Settings, Enable 32-Bit Applications = False) and the project platform x64
   or Any CPU without "Prefer 32-bit".
3. Copy the `.onnx` files and the two `*_vocab.json` files to a folder, for example
   `App_Data/models`.
4. Create one `GenderClassifier` for the lifetime of the application and share it. `Run` is
   thread-safe.

`Global.asax.cs`:

```csharp
using System;
using System.Web;
using IndoNameGender;

public class Global : HttpApplication
{
    public static GenderClassifier Classifier;

    protected void Application_Start(object sender, EventArgs e)
    {
        string dir = Server.MapPath("~/App_Data/models");
        Classifier = new GenderClassifier(dir, "CharBiLSTM");
    }

    protected void Application_End(object sender, EventArgs e)
    {
        if (Classifier != null) Classifier.Dispose();
    }
}
```

`Predict.ashx` (a JSON endpoint, `Predict.ashx?name=banowati larasati`):

```csharp
<%@ WebHandler Language="C#" Class="PredictHandler" %>
using System.Web;
using System.Web.Script.Serialization;

public class PredictHandler : IHttpHandler
{
    public void ProcessRequest(HttpContext context)
    {
        string name = context.Request.QueryString["name"];
        if (string.IsNullOrWhiteSpace(name) || name.Length > 200)
        {
            context.Response.StatusCode = 400;
            return;
        }
        var result = Global.Classifier.Predict(name);
        context.Response.ContentType = "application/json";
        context.Response.Write(new JavaScriptSerializer().Serialize(result));
    }

    public bool IsReusable { get { return true; } }
}
```

Tokenizer check, without ONNX Runtime:

```bash
csc -out:TokenizerCheck.exe -r:System.Web.Extensions.dll Blake2s.cs NameTokenizer.cs TokenizerCheck.cs
TokenizerCheck.exe ..\models
```

`Blake2s.cs` and `NameTokenizer.cs` were checked this way against the Python output (131 checks,
all match, including words longer than one 64-byte block and non-ASCII text).
`GenderClassifier.cs` and the handler above have not been compiled against ONNX Runtime yet.

Notes:

- Characters outside the Basic Multilingual Plane (for example emoji) are two UTF-16 units in
  .NET and one code point in Python, so such input can tokenize differently. Names do not
  contain them in practice.
- `Predict` is a per-name call. To score many names at once, build one `[n, max_len]` tensor and
  read `n` outputs.

## Re-export

```bash
python onnx/export_onnx.py
```

It needs `torch`, `onnx`, `onnxruntime` and `onnxscript`, the checkpoints in `models/` and the
tokenizers in `tokenizers/`. If `data/splits/val_2024_2026.csv` exists locally, 2,000 of its names
are used for the parity check, otherwise only the ten synthetic names are used.
