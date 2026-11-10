# Speaker notes for Ropeformer_Final.pdf

Format (course handout): **20-minute presentation + 10-minute Q&A**; all members present and must be able to answer questions on any part.

Main deck: pages 1–27 (19 content slides, about 19.5 min). Backup: pages 28–35, shown only when asked.

Suggested split:

- **Liu Yuncong**, pages 1–8, about 6 min: context, problem, proposal, RoPE.
- **Zhang Yihang**, pages 9–17, about 6.5 min: method, intuition, setup, first results.
- **Ma Xiaoyu**, pages 18–27, about 7 min: mechanism, multi-horizon training, finance, limitations, takeaways.

Page numbers are PDF pages. Section divider pages take no time.

| Page | Slide | Time | What to say |
|---|---|---|---|
| 1 | Title | 0:20 | Names. One-line message: in our experiments RoPE degrades beyond the training horizon, and the evidence points to its rotation being out of step with the daily cycle; learnable frequencies reduce the error by 38–66%. |
| 2 | Contents | 0:10 | Five parts. |
| 4 | Context and goal | 0:50 | Finance needs several horizons: 1- and 10-day VaR, next day and next week in energy. Usual practice is one model per horizon. Goal: train once at 96 steps and forecast up to 720. A step is one observation; on hourly prices 96 steps = 4 days and 720 = 30 days. |
| 5 | The problem | 1:00 | What happens: beyond the training horizon the error rises sharply, up to 6.4× that of a model trained for 720. Why it matters: the failure is silent, RoPE is a standard component, and the alternative is one model per horizon. Our explanation: RoPE's fixed periods do not match the daily cycle (evidence on pages 12, 18–19). |
| 6 | Where we started | 0:50 | The proposal gave each head a learnable time scale. At the training horizon the scales hardly moved and the heads learned similar scales. The surprise came beyond the horizon. |
| 7 | RoPE | 1:00 | Attention needs position information. RoPE rotates each query/key pair by an angle proportional to position, so attention depends only on the lag. (Background on Transformers and positional encodings: backup pages 29–31.) |
| 8 | Frequency interpretation | 0:50 | Each pair is a sinusoid. The fastest pair turns once every 6.28 tokens. Red line: in our experiments the learned scale did not make heads specialise; it moved this fastest pair close to the daily cycle. |
| 10 | Model | 1:10 | Patches of 16 steps. One placeholder per future patch. All placeholders are identical, so only RoPE tells them apart, which is why RoPE decides extrapolation. The mask makes the forecast independent of H. |
| 11 | Learnable scales | 0:50 | ω̂ = ω·e^a, 4 parameters per layer, starting at standard RoPE. Ablation variants. Second remedy: multi-horizon training. |
| 12 | Clock intuition | 1:10 | ETTm1: 1 day = 6 tokens, but standard RoPE takes 6.28, so it falls 0.28 tokens behind each day. After 7 days the hand points somewhere never seen in training. With alignment, every midnight looks the same. |
| 14 | Setup | 0:50 | 7 datasets, 3 of them financial. Baselines and why: fixed RoPE isolates the mechanism, linear is strong (DLinear), direct-720 is the real alternative. Significance: Diebold–Mariano tests. |
| 15 | Result 1 | 0:40 | At 96 steps the model is competitive and linear models stay strong. Fixed and learnable RoPE differ by about one seed standard deviation or less. |
| 16 | Result 2: collapse | 1:10 | The orange line rises sharply after the dashed line. Up to 6.4× the error of a directly trained model; the scales reduce it by 38–66%; all 6 runs improve, each significant by DM. |
| 17 | Result 2: all datasets | 0:40 | Significant on Electricity, ETTh1, ETTm1 and both electricity markets. No gain on Weather or exchange rates, explained next. |
| 18 | Why: alignment | 1:20 | Key slide. Learned scale 1.04 vs aligned 1.047. Fixing 1.047 without learning matches the learned model; 5% off degrades like standard RoPE (1.047 beats 0.955 in all 6 runs). Backup page 28: head diversity, grouping and clipping do not explain the gain. |
| 19 | Attention | 1:00 | Each future patch looks at history at the same time of day ("same time yesterday"). Fixed RoPE loses this after about 2 days (0.64 → 0.05, below chance); aligned 1.047 keeps it at 0.53 for all 7.5 days. |
| 20 | Gain vs daily cycle | 0:50 | Stronger daily cycle, larger gain. In Weather only solar radiation has a cycle, so only those channels gain. |
| 21 | Multi-horizon | 1:00 | One model on mixed horizons ≈ per-horizon models on all 7 datasets, at a small cost at 96. When long-horizon training overfits (ETTh1, EPF-NP), learnable scales plus extrapolation do best. |
| 23 | Finance | 1:00 | Electricity prices: significant gains. Exchange rates: close to a random walk with no daily cycle; no consistent effect; linear model best; RevIN needed. Three practical rules. |
| 25 | Limitations | 0:40 | No evidence for the proposal's head-specialisation hypothesis; mechanism analyses cover ETTh1, ETTm1 and Weather only; small model; point tokens noisier. Future work: FFT-based initialisation, several cycles, larger models. |
| 26 | Takeaways | 0:40 | Problem → cause → fix → finance. |
| 27 | Thank you | – | Q&A. |

Total ≈ 19.5 min. If you run short of time, compress page 7 to 30 s and page 15 to 20 s.

## Backup pages (for Q&A)

| Page | Use it when asked about |
|---|---|
| 28 | What else could explain the gain? (head diversity, grouping, gradient clipping) |
| 29–31 | Transformer and positional-encoding basics |
| 32–34 | Derivation of RoPE's relative-position property |
| 35 | Replication and hyper-parameters |

## Likely questions

1. **Isn't learnable RoPE already known?** RoPE is Su et al. (2021), and tunable RoPE frequencies appear in later work. Our contributions are: an explanation of the extrapolation failure supported by experiments (alignment with the daily cycle); controlled evidence (fixed 1.047 vs 0.955/1.10, DM tests, attention analysis); and when it helps (seasonal vs random-walk data).
2. **Why not just train at 720?** Multi-horizon training is the stronger fix when long targets exist. Learnable scales help when they do not, or when long-horizon training overfits (ETTh1, EPF-NP).
3. **Why does linear win at 96 on some datasets?** This is a known result (DLinear, AAAI 2023). Our claim is about extrapolation, not about beating linear models.
4. **Are 3 seeds enough?** Each run also gets a Diebold–Mariano test over thousands of test windows, with a Newey–West variance for overlapping windows.
5. **Why patches?** With point tokens the direction is the same but noisier. Patches also make one day an integer number of tokens on ETTm1.
6. **What happened to the oil and attention-heatmap results of the earlier deck?** They come from a previous prototype and are not part of this project's evidence. Those heatmaps suggested head specialisation, which our controlled ablations did not find. Our own attention analysis (page 19) shows something else: a same-time-of-day lookup that only an aligned RoPE keeps beyond the training horizon.
7. **Bitcoin?** A supplementary run, not in the report: very weak daily cycle (seasonal strength 0.01) and a small gain (about 6–10% at 720 steps), consistent with the mechanism.
