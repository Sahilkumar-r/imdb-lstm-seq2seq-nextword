# LSTM Next-Word Predictor — Encoder/Decoder with Luong Attention

A PyTorch next-word prediction model trained on the review text of the
[IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
The `sentiment` column is dropped; only the `review` text is used.

The notebook is built for **Google Colab** and downloads the dataset itself through the Kaggle API.

## Files

| File | Purpose |
|---|---|
| `lstm_next_word_luong.ipynb` | Full pipeline: download, preprocessing, model, training, evaluation, prediction |
| `requirements.txt` | Dependencies (only needed outside Colab) |
| `README.md` | This file |

## Architecture

```
context (20 tokens) ──► Embedding ──► Encoder: 12 × LSTM (residual from layer 2, dropout, LayerNorm)
                                            │ outputs h̄_1..h̄_S          │ final (h, c) of all 12 layers
                                            ▼                              ▼
<start> + shifted target ──► Embedding ──► Decoder: 12 × LSTM (residual, dropout, LayerNorm)
                                            │ h_t
                                            ▼
                              Luong "general" attention: score = h_tᵀ W_a h̄_s  (padding masked)
                              c_t = Σ a_ts h̄_s ;  h̃_t = tanh(W_c [c_t ; h_t])
                                            ▼
                              Linear (tied with embedding) ──► softmax over vocabulary
```

| Setting | Default |
|---|---|
| Layers | 12 encoder + 12 decoder (`num_layers`) |
| Embedding / hidden size | 256 / 256 |
| Vocabulary | 20,000 (train split only, min frequency 3) |
| Special tokens | `<pad>`, `<unk>`, `<start>`, `<end>` |
| Context / target length | 20 / 5 tokens, stride 4 |
| Optimiser | AdamW, lr 1e-3, grad clip 1.0, mixed precision |
| Parameters | ≈ 18M (with weight tying) |

### Design decisions

- **Data split by review, after removing duplicates.** The dataset has duplicate reviews; without dedup the same text can appear in train and test. Splitting by review (90/5/5) means no window from a test review is seen during training. The vocabulary is built from the training split only.
- **Sliding windows.** Each review becomes `<start> w1 … wn <end>`. A window gives the encoder 20 context tokens and the decoder the next 5. Reviews are left-padded so the model also learns from very short contexts (e.g. only `<start>`). Windows never cross review boundaries.
- **Teacher forcing.** The decoder input is `<start>` + target shifted right, so every decoder position is a genuine next-word prediction.
- **Residual connections in the 12-layer stacks.** Plain deep LSTM stacks train poorly because gradients must pass through every layer; GNMT (Wu et al., 2016) used residual connections for exactly this reason.
- **Left padding + attention mask.** The encoder's final state always sits on the last real token, so no sequence packing is needed; padding is excluded from attention.
- **No input feeding.** Luong's optional input-feeding requires a step-by-step decoder loop, which is much slower with 12 layers. Global attention without it is a standard Luong variant.
- **Weight tying.** The output layer shares the embedding matrix, cutting about 5M parameters and usually improving language-model perplexity.
- **Baseline.** A unigram (word-frequency) model is evaluated on the same windows. The LSTM's numbers only mean something relative to it.

### Known limitations

- For pure next-word prediction, a **decoder-only** language model is the standard design; the encoder/decoder framing here is a deliberate choice, not a requirement of the task.
- A 12-layer stack is unlikely to beat a 2–4 layer stack on 50K reviews and is several times slower. Set `num_layers = 2` and compare before drawing conclusions.
- `<unk>` counts as a normal target in the metrics, so accuracy is slightly optimistic.

## Setting up the Kaggle API key in Google Colab

### Step 1 — Get your credentials from Kaggle
1. Sign in at [kaggle.com](https://www.kaggle.com).
2. Click your profile picture → **Settings**.
3. Scroll to the **API** section and click **Create New Token**.
4. A file `kaggle.json` downloads. It looks like `{"username":"your_name","key":"abc123..."}`.
   Treat the key like a password and never commit it to GitHub.

### Step 2 (recommended) — Store it as Colab Secrets
1. Open the notebook in Colab.
2. Click the **key icon** (🔑 Secrets) in the left sidebar.
3. Add two secrets:
   - Name `KAGGLE_USERNAME`, value = `username` from `kaggle.json`
   - Name `KAGGLE_KEY`, value = `key` from `kaggle.json`
4. Turn on **Notebook access** for both.

Secrets are stored in your Google account, so you set them once and every notebook can use them.

> If Kaggle's settings page gives you a single token string instead of a `kaggle.json` file, store it as a secret named `KAGGLE_API_TOKEN`. The notebook passes it through; this requires a recent `kaggle` package, which the notebook installs.

### Step 2 (alternative) — Upload `kaggle.json`
If no secrets are found, the download cell shows an upload button. Select your `kaggle.json`; the notebook copies it to `~/.kaggle/kaggle.json` with permissions `600`. You have to repeat this every new Colab session.

### What the notebook then does
```bash
kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews -p /content/data --unzip
```
If the CSV already exists in `/content/data`, the download is skipped.

## How to run

1. Upload `lstm_next_word_luong.ipynb` to Colab (File → Upload notebook) or open it from GitHub/Drive.
2. **Runtime → Change runtime type → T4 GPU.**
3. Set up the Kaggle secrets (above).
4. Optionally edit the `Config` cell (`num_layers`, `epochs`, `windows_per_epoch`, `save_to_drive`).
5. **Runtime → Run all.**

Training time depends on the GPU; check the speed of the progress bar in the first epoch and reduce `windows_per_epoch` or `epochs` if needed. Set `save_to_drive = True` so the best checkpoint survives a Colab disconnect.

### Using the trained model
```python
predict_next("this movie was one of the", k=5)
# -> [(word, probability), ...]

generate("the acting was", n_words=25, temperature=0.8, top_k=40)
```

## Results

> To be added after training.

| Model | Layers (enc/dec) | Val ppl | Test ppl | Test top-1 | Test top-5 | Time / epoch | GPU |
|---|---|---|---|---|---|---|---|
| Unigram baseline | – | | | | | – | – |
| LSTM seq2seq + Luong | 12 / 12 | | | | | | |
| LSTM seq2seq + Luong | 2 / 2 | | | | | | |

**Sample predictions:** _to be added_

**Training curves:** _to be added_

## Troubleshooting

| Problem | Fix |
|---|---|
| `401 Unauthorized` from Kaggle | Wrong username/key, or Notebook access is off for the secrets. Create a new token and update the secrets. |
| `Could not find kaggle.json` | Neither secrets nor file were found; re-run the download cell and upload `kaggle.json`. |
| `CUDA out of memory` | Lower `batch_size` (e.g. 128) or `num_layers`. |
| Very slow training | Check that a GPU runtime is selected; lower `windows_per_epoch`. |
| Loss becomes `nan` | Lower `lr` to 5e-4; gradient clipping is already on. |

## References

- Luong, Pham, Manning (2015). *Effective Approaches to Attention-based Neural Machine Translation.*
- Sutskever, Vinyals, Le (2014). *Sequence to Sequence Learning with Neural Networks.*
- Wu et al. (2016). *Google's Neural Machine Translation System* (residual connections in deep LSTM stacks).
- Press, Wolf (2017). *Using the Output Embedding to Improve Language Models* (weight tying).
