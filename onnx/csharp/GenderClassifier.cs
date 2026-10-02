using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace IndoNameGender
{
    public class Prediction
    {
        public string Name;
        public string Model;
        public string Label;          // "L" (male) or "P" (female)
        public double ProbFemale;
        public double Confidence;     // probability of the predicted label
        public string[] Tokens;       // characters (Char models) or words (Word models)
        public float[] Attention;     // one weight per token
    }

    /// <summary>
    /// Loads one ONNX model and its tokenizer. Create it once (for example in Application_Start)
    /// and share it, InferenceSession.Run is thread-safe.
    /// </summary>
    public class GenderClassifier : IDisposable
    {
        readonly InferenceSession _session;
        readonly NameTokenizer _tokenizer;
        readonly bool _isChar;
        public readonly string ModelName;

        /// <param name="modelsDir">folder holding the .onnx files and the two *_vocab.json files</param>
        /// <param name="modelName">for example "CharBiLSTM" or "WordTransformer"</param>
        public GenderClassifier(string modelsDir, string modelName = "CharBiLSTM")
        {
            ModelName = modelName;
            _isChar = modelName.StartsWith("Char", StringComparison.Ordinal);
            _tokenizer = NameTokenizer.Load(
                Path.Combine(modelsDir, _isChar ? "char_vocab.json" : "word_vocab.json"), _isChar);
            _session = new InferenceSession(Path.Combine(modelsDir, modelName + ".onnx"));
        }

        public Prediction Predict(string name)
        {
            long[] ids = _tokenizer.Encode(name);
            var input = new DenseTensor<long>(ids, new[] { 1, ids.Length });
            using (var results = _session.Run(
                new[] { NamedOnnxValue.CreateFromTensor("ids", input) }))
            {
                float pFemale = results.First(r => r.Name == "prob_female").AsEnumerable<float>().First();
                float[] attn = results.First(r => r.Name == "attention").AsEnumerable<float>().ToArray();

                var lower = (name ?? "").ToLowerInvariant();
                string[] tokens = _isChar
                    ? lower.Select(c => c.ToString()).ToArray()
                    : lower.Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
                int n = Math.Min(tokens.Length, _tokenizer.MaxLen);

                bool female = pFemale >= 0.5f;
                return new Prediction
                {
                    Name = name,
                    Model = ModelName,
                    Label = female ? "P" : "L",
                    ProbFemale = pFemale,
                    Confidence = female ? pFemale : 1.0 - pFemale,
                    Tokens = tokens.Take(n).ToArray(),
                    Attention = attn.Take(n).ToArray(),
                };
            }
        }

        public void Dispose() { _session.Dispose(); }
    }
}
