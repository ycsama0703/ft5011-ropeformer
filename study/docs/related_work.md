# Related-work brief: learnable RoPE frequency scaling for time-series forecasting

Compiled 2026-11-09. Every title, author list, year and ID below was checked against the arXiv abstract page, the arXiv HTML full text, or an official proceedings page (PMLR / ICLR / ICML / CVPR) on the day of writing. Where a venue could not be confirmed from an official page this is stated. Numbers quoted from papers are the authors' own claims.

Notation used here: our ladder has 8 rotary pairs with base b = 1e4, so omega_j = 1e4^(-j/8) for j = 0..7, i.e. wavelengths 2*pi/omega_j of about 6, 20, 63, 199, 628, 1987, 6283, 19869 positions. With lookback L = 336, pairs 0-4 complete at least half a rotation inside the window; pairs 5-7 ("slow group" in our two-group split, if split 4/4 the slow group is pairs 4-7) rotate 0.17, 0.05 and 0.017 of a turn respectively, which is where the "about 6x needed" figure comes from (336/1987 ~ 1/6).

---

## 1. RoFormer (origin of RoPE)

**Su, J., Lu, Y., Pan, S., et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding." arXiv:2104.09864 (2021); journal version Neurocomputing 568:127063 (2024).**
Six authors in total (Su, Lu, Pan, Murtadha, Wen, Liu). Introduces RoPE: query/key pairs are rotated by angle m*theta_j with theta_j = 10000^(-2j/d), so absolute positions are encoded while attention scores depend only on relative offsets; the paper also derives a long-term decay of the score with distance.
*Relation:* our variant A and the default base 1e4 come directly from this paper; the "decay with distance" property is exactly what Barbero et al. (Sec. 5) later argue is not the mechanism that matters.

## 2. LeRoPE (closest prior work; ID verified)

**Karypis, P., O'Brien, S., Kadekodi, S., Zhu, R., McAuley, J. "LeRoPE: Learnable RoPE Frequencies Improve Language Modeling." arXiv:2607.10134 (submitted 11 Jul 2026, cs.LG).**
The ID resolves and the paper exists. Method: one learnable scalar alpha_m per frequency band (32 for head dim 64), multiplicative in log space (omega_m * e^{alpha_m}), initialised at 0 so training starts at standard RoPE; **shared across all layers and heads** (the authors tried per-layer, per-head scalars and found no benefit: 2.8996 vs 2.8978 nats). No weight decay on the scalars, separate gradient clipping, no separate learning rate (LR was swept per method and the optimum coincided). Trained on C4 at length 2048, 52M-2.52B params. Findings: LeRoPE beats RoPE at every scale (e.g. 10.14 vs 10.17 ppl at 2.52B; RoPE needs about 3.4% more compute to match); a single high-norm "positional band" emerges whose wavelength is about 2.2x the training length; learned high frequencies barely move, mid frequencies slow down, and the lowest frequencies are pushed continuously toward zero (not truncated). p-RoPE (p = 0.75) captures only about 10% of LeRoPE's gain.
*Relation:* this is the language-model analogue of our variant B/C. It supports two of our findings (learnable scales beat fixed RoPE by a small but robust margin; per-head vs shared is within noise) and contrasts with a third: in LMs the slow frequencies move the most and move *down*, whereas ours move little and move *up* (1.1-1.4x). A plausible reconciliation: with L = 336 our pair j = 4 (wavelength 628 ~ 1.9 L) already sits near LeRoPE's "band" wavelength of about 2.2 L, so the ladder is close to where LeRoPE's models end up, and pairs 5-7 are below the band where LeRoPE also finds little useful positional signal.

## 3. YaRN and NTK-by-parts (fixed group-wise frequency scaling)

**Peng, B., Quesnelle, J., Fan, H., Shippole, E. "YaRN: Efficient Context Window Extension of Large Language Models." arXiv:2309.00071 (2023).** (Also appeared at ICLR 2024; venue not shown on the arXiv page, so cite the arXiv ID.)
NTK-by-parts (Sec. 3.2 of the paper) is explicitly group-wise: for each dimension d it computes r(d) = L / lambda_d, the number of rotations the dimension completes inside the original context L, and a ramp gamma(r) = 0 for r < alpha, 1 for r > beta, linear in between (alpha = 1, beta = 32 for LLaMA). The new frequency is h(theta_d) = (1 - gamma) * theta_d / s + gamma * theta_d, where s is the context extension ratio. So dimensions with wavelength longer than the context (r < 1) are interpolated by 1/s, dimensions that rotate more than 32 times are left untouched, and the band between is blended. YaRN = NTK-by-parts plus an attention temperature sqrt(1/t) = 0.1 ln(s) + 1 applied by scaling q and k.
*Relation:* our variant D is a learnable, two-group relaxation of NTK-by-parts (fast group vs slow group, boundary fixed, scale learned instead of hand-set); the r(d) < 1 criterion is exactly the "does the pair rotate within the window" criterion we use to define the slow group.

## 4. LongRoPE (non-uniform per-dimension rescale by search)

**Ding, Y., Zhang, L. L., Zhang, C., et al. "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens." ICML 2024, PMLR 235:11091-11104; arXiv:2402.13753.**
Eight authors (Microsoft Research). Exploits two non-uniformities: (i) each RoPE dimension gets its own rescale factor lambda_i, (ii) the first n_hat tokens are not interpolated. The lambda_i are found by evolutionary search on PG19 validation perplexity over [1.0, 1.25 s] in steps of 0.01, with a monotonic non-decreasing constraint lambda_i <= lambda_{i+1} across dimensions; the population is initialised from PI, NTK and YaRN factors.
*Relation:* LongRoPE is the search-based counterpart of our gradient-trained per-pair scales (the G = 8 option); its monotonicity constraint is a cheap prior we could impose (parameterise log-scales as cumulative sums of softplus increments) if G = 8 turns out noisy.

## 5. Round and Round We Go (low frequencies as semantic channels; p-RoPE)

**Barbero, F., Vitvitskyi, A., Perivolaropoulos, C., Pascanu, R., Velickovic, P. "Round and Round We Go! What makes Rotary Positional Encodings useful?" ICLR 2025; arXiv:2410.06205 (2024).** (ICLR 2025 listing confirmed via DBLP; OpenReview id nA464tCGR5.)
Analysing Gemma 7B, the authors find that attention heads build sharp positional patterns (diagonal / previous-token heads) from the *highest* RoPE frequencies, while the model's query/key norms concentrate on the *lowest* frequencies, which rotate so little over the context that they behave as distance-robust "semantic" channels. They argue the "decay with distance" explanation is not the real mechanism. p-RoPE keeps a fraction p of frequencies and removes the lowest (1 - p); 0.75-RoPE (lowest 25% removed, i.e. NoPE on those dims) slightly improves Gemma 2B (Wiki ppl 4.441 vs 4.463) and beats both theta = 500k and removing the *highest* frequencies instead.
*Implication for our slow-group result:* with b = 1e4 and L = 336, pairs 5-7 rotate at most 1/6 of a turn over the whole window, so from the attention's point of view they are near-identity rotations: content can pass through them with almost no positional modulation. If the model uses them that way (as Barbero et al. observe in LMs), then the gradient pushing their scale toward "rotate meaningfully" is weak or even negative, and a learned scale of 1.1-1.4x instead of about 6x is what one should expect. In other words, the small movement is not a failure of the learnable-scale mechanism; it is evidence that the slow pairs are being used as semantic channels, which p-RoPE-style truncation can test directly (see experiments).

## 6. Other 2024-2026 work on adaptive / learnable RoPE frequencies (3 most relevant, all verified)

**6a. Wang, S., Zheng, Y., Zhu, T., et al. "AdaRoPE: Not All Attention Heads Should Rotate and Scale Equally." ICML 2026; arXiv:2607.19363 (2026).** Seven authors. Gives every attention head its own learnable rotation frequencies (AdaFreq) and a length-dependent attention scaling factor (AdaScale), arguing heads with different roles need different frequency ranges; reports gains over RoPE, partial RoPE and NoPE and better context extension than YaRN.
*Relation:* essentially our variant B (per-head scale) at LM scale, with the YaRN temperature made learnable too. Contrasts with LeRoPE, which found per-head frequencies gave no benefit; our "grouped vs per-head within noise" lands between the two.

**6b. Hua, E., Jiang, C., Lv, X., et al. "Fourier Position Embedding: Enhancing Attention's Periodic Extension for Length Generalization." ICML 2025, PMLR 267:24932-24949; arXiv:2412.17739.** Ten authors. Treats each RoPE dimension as a Fourier series (dominant frequency plus weighted harmonics) and, importantly for us, sets a floor frequency 2*pi/N (N = training length) and clips every frequency below it to zero, because such components are under-trained and carry no usable positional information.
*Relation:* a principled, fixed version of "truncate the slow group": FoPE's floor rule applied to our ladder would zero pairs 5-7 (wavelength > 336), exactly the pairs whose learned scales barely move.

**6c. Oka, Y., Saito, I., Nishida, K., Saito, K. "Frequency Bands in RoPE: Base Frequency and Context Length Shape the Interpolation-Extrapolation Trade-Off." ICLR 2026 (OpenReview PR1PPxvG9Q).** Identifies a consistent high-norm RoPE dimension (the "frequency band") whose location depends on base theta and training length; replacing the dimensions *below* the band with NoPE at inference barely changes performance; setting theta roughly equal to the training length moves the band to lower frequencies and improves extrapolation, while larger theta favours interpolation.
*Relation:* gives a zero-training-cost diagnostic (ablate slow pairs at inference) and motivates our "initialise the ladder to the window" experiment; also independently confirms LeRoPE's band finding.

Briefly noted (verified, less central): **Resonance RoPE** (Wang, S., Kobyzev, I., Lu, P., et al., ACL 2024 Findings, arXiv:2403.00071) rounds each wavelength to the nearest integer so features are periodic on integer positions; **ComRoPE** (Yu, H., Jiang, T., Jia, S., et al., CVPR 2025, arXiv:2506.03737) makes the whole angle matrix trainable subject to pairwise commutativity, a strict generalisation of learnable scalar frequencies; **VideoRoPE** (Wei, X., Liu, X., Zang, Y., et al., arXiv:2502.05173, 2025) assigns the temporal axis to the *low* frequencies to suppress periodic oscillations; **Wu, X., Liu, S., Jadbabaie, A. "How Data Shapes RoPE Frequency Usage: From Positional Scale Matching to Length Generalization", arXiv:2607.07678 (2026)** argues trained models select frequencies to match the dependency width W of the data (optimal frequency ~ 1/W), which predicts that for periodic series the learned scales should track the dominant period rather than the window.

## 7. Time-series Transformers that use RoPE

- **Moirai.** Woo, G., Liu, C., Kumar, A., et al. "Unified Training of Universal Time Series Forecasting Transformers." ICML 2024 (oral), PMLR 235:53140-53164; arXiv:2402.02592. Masked-encoder foundation model (LOTSA, 27B observations). Uses RoPE on the time axis plus learned binary attention biases (two scalars per head, same-variate vs different-variate) for variate identity. **RoPE base not stated in the paper.**
- **Time-MoE.** Shi, X., Wang, S., Nie, Y., et al. "Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts." ICLR 2025; arXiv:2409.16040. Decoder-only, point-wise tokens, sparse MoE, trained up to 4096 context; "instead of using absolute positional encoding, we adopt rotary positional embeddings." **Base not stated.**
- **Timer-XL.** Liu, Y., Qin, G., Huang, X., Wang, J., Long, M. "Timer-XL: Long-Context Transformers for Unified Time Series Forecasting." ICLR 2025; arXiv:2410.04803. Decoder-only, patch tokens flattened over variables; TimeAttention uses RoPE on the temporal dimension and two learnable per-head scalar biases for same/different variable. **Base not stated.**
*Relation:* all three adopt RoPE with (as far as the papers say) the LM default and no time-series-specific tuning of the frequency ladder, which is the gap our project targets. None of them reports learnable or rescaled frequencies. (Moirai and Timer-XL's per-head variate-bias scalars are an unrelated but similarly cheap per-head learnable attention term.)

## 8. Baselines

**Nie, Y., Nguyen, N. H., Sinthong, P., Kalagnanam, J. "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." ICLR 2023; arXiv:2211.14730.** Patching plus channel independence, RevIN, learned absolute positional embedding (not RoPE); our channel-independent backbone follows this design.

**Li, Z., Qi, S., Li, Y., Xu, Z. "Revisiting Long-term Time Series Forecasting: An Investigation on Linear Mapping." arXiv:2305.10721 (2023).** (A 2026 journal version exists under the title "... an Investigation on Affine Mapping", Academia AI and Applications 2(2); cite the arXiv version for RLinear.) Defines RLinear = RevIN + a single linear layer over time (Appendix A, Algorithm 1) and shows it beats PatchTST on ETTh1/ETTh2/ETTm2 but not ETTm1, Weather or ECL at input 336.
*Relation:* our "RLinear wins on 5 of 9 datasets" is consistent with their mixed table; it argues for framing our contribution as a positional-encoding study rather than a new SOTA forecaster.

## 9. Positional encoding in time-series forecasting (verified)

**Zhang, J., Wang, J., Qiang, W., et al. "Intriguing Properties of Positional Encoding in Time Series Forecasting." arXiv:2404.10337v1 (Apr 2024).** Seven authors on v1. Note: the current v4 (Oct 2025) is retitled "Understanding Token-level Topological Structures in Transformer-based Time Series Forecasting" with a changed author list (Zhang, Qiang, Wang, Zhou, Zheng, Xiong) and a different framing (Topology Enhancement Method); no venue is listed. v1 finds, via mutual information between PE and layer inputs, that positional information injected by standard PE fades with depth in forecasting Transformers and that re-injecting it in deeper layers helps.
*Relation:* independent evidence that positional information is a live design variable in TS Transformers; RoPE, which is applied inside every attention layer, side-steps the "fading" problem they describe, which is one argument for RoPE-based TS models. Cite v1 explicitly if used.

---

## Recommended cheap experiments (each <= 1 GPU-hour on the d_model 64 model), ranked

1. **p-RoPE / FoPE-style truncation of the slow group (highest priority).** Two sub-steps: (a) zero cost: take already-trained A/B/D checkpoints and at *inference* replace pairs 5-7 (or the whole slow group) by identity rotation, following Oka et al.'s "NoPE below the band" ablation; (b) retrain once with those pairs set to zero frequency (FoPE floor rule, wavelength > 336 -> 0). If MSE is unchanged or improves, the slow pairs are content channels and the 1.1-1.4x result is explained by Barbero et al.; if MSE degrades, they carry positional signal despite barely rotating. Either outcome sharpens the story, and the test is literally what three verified papers (Barbero, Hua, Oka) do in LMs.

2. **Initialise the ladder to match the window.** Train fixed and learnable variants at base b in {336, 1000, 1e4} (Oka et al.: theta ~ training length shifts the band; LeRoPE: the useful band sits at wavelength ~2.2 L; Wu et al.: frequency ~ 1/W). This separates two readings of our result: "learnable scales recover what a well-chosen base would give" (gain shrinks at matched base) versus "learnable scales add something beyond base tuning" (gain persists). It also tests whether the slow scales fail to reach 6x because of optimisation (they should move much less when the init is already matched).

3. **Higher learning rate / no weight decay for the scale parameters.** LeRoPE uses no weight decay and separate gradient clipping on the scalars but no separate LR; AdaRoPE learns per-head frequencies successfully. A 10x-100x LR sweep on a_{l,h} only (plus removing weight decay if we currently apply it) is the control that lets us claim the 1.1-1.4x movement is a converged equilibrium rather than an under-trained parameter. Cheap, but the outcome is less informative than 1 or 2 because LeRoPE's evidence already suggests the scales are not LR-limited.

4. **Per-pair scales (G = 8), lowest priority.** LeRoPE found per-head/per-layer freedom gave nothing, and our own grouped-vs-per-head comparison is within noise, so G = 8 is likely also within noise and adds little narrative. If run, impose LongRoPE's monotonic constraint to reduce variance and report the learned ladder as a curve (that plot, if it shows the LeRoPE pattern of slow pairs moving most, would be the one interesting output).

---

## LaTeX bibitems (short format, ready to paste)

```latex
\bibitem{su2021roformer} J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, Y. Liu. RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864, 2021 (Neurocomputing 568:127063, 2024).
\bibitem{karypis2026lerope} P. Karypis, S. O'Brien, S. Kadekodi, R. Zhu, J. McAuley. LeRoPE: Learnable RoPE Frequencies Improve Language Modeling. arXiv:2607.10134, 2026.
\bibitem{peng2023yarn} B. Peng, J. Quesnelle, H. Fan, E. Shippole. YaRN: Efficient Context Window Extension of Large Language Models. arXiv:2309.00071, 2023.
\bibitem{ding2024longrope} Y. Ding, L. L. Zhang, C. Zhang, et al. LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens. ICML 2024 (PMLR 235), 2024.
\bibitem{barbero2025round} F. Barbero, A. Vitvitskyi, C. Perivolaropoulos, R. Pascanu, P. Veli\v{c}kovi\'{c}. Round and Round We Go! What makes Rotary Positional Encodings useful? ICLR 2025 (arXiv:2410.06205), 2025.
\bibitem{wang2026adarope} S. Wang, Y. Zheng, T. Zhu, et al. AdaRoPE: Not All Attention Heads Should Rotate and Scale Equally. ICML 2026 (arXiv:2607.19363), 2026.
\bibitem{hua2025fope} E. Hua, C. Jiang, X. Lv, et al. Fourier Position Embedding: Enhancing Attention's Periodic Extension for Length Generalization. ICML 2025 (PMLR 267), 2025.
\bibitem{oka2026bands} Y. Oka, I. Saito, K. Nishida, K. Saito. Frequency Bands in RoPE: Base Frequency and Context Length Shape the Interpolation--Extrapolation Trade-Off. ICLR 2026, 2026.
\bibitem{wang2024resonance} S. Wang, I. Kobyzev, P. Lu, M. Rezagholizadeh, B. Liu. Resonance RoPE: Improving Context Length Generalization of Large Language Models. Findings of ACL 2024 (arXiv:2403.00071), 2024.
\bibitem{yu2025comrope} H. Yu, T. Jiang, S. Jia, et al. ComRoPE: Scalable and Robust Rotary Position Embedding Parameterized by Trainable Commuting Angle Matrices. CVPR 2025 (arXiv:2506.03737), 2025.
\bibitem{wei2025videorope} X. Wei, X. Liu, Y. Zang, et al. VideoRoPE: What Makes for Good Video Rotary Position Embedding? arXiv:2502.05173, 2025.
\bibitem{wu2026data} X. Wu, S. Liu, A. Jadbabaie. How Data Shapes RoPE Frequency Usage: From Positional Scale Matching to Length Generalization. arXiv:2607.07678, 2026.
\bibitem{woo2024moirai} G. Woo, C. Liu, A. Kumar, C. Xiong, S. Savarese, D. Sahoo. Unified Training of Universal Time Series Forecasting Transformers. ICML 2024 (PMLR 235), 2024.
\bibitem{shi2025timemoe} X. Shi, S. Wang, Y. Nie, et al. Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts. ICLR 2025 (arXiv:2409.16040), 2025.
\bibitem{liu2025timerxl} Y. Liu, G. Qin, X. Huang, J. Wang, M. Long. Timer-XL: Long-Context Transformers for Unified Time Series Forecasting. ICLR 2025 (arXiv:2410.04803), 2025.
\bibitem{nie2023patchtst} Y. Nie, N. H. Nguyen, P. Sinthong, J. Kalagnanam. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. ICLR 2023, 2023.
\bibitem{li2023rlinear} Z. Li, S. Qi, Y. Li, Z. Xu. Revisiting Long-term Time Series Forecasting: An Investigation on Linear Mapping. arXiv:2305.10721, 2023.
\bibitem{zhang2024intriguing} J. Zhang, J. Wang, W. Qiang, et al. Intriguing Properties of Positional Encoding in Time Series Forecasting. arXiv:2404.10337v1, 2024.
```

## Verification notes / caveats

- arXiv:2607.10134 resolves to LeRoPE (Karypis et al., 11 Jul 2026); details above come from the arXiv HTML full text.
- YaRN's ICLR 2024 acceptance is not shown on the arXiv page and was not independently confirmed here; the arXiv ID is safe to cite.
- Moirai, Time-MoE and Timer-XL state that they use RoPE but none states the base in the paper text we checked; do not write "base 10000" for them without checking their code.
- arXiv:2404.10337 changed title and author list between v1 and v4; cite v1 explicitly if you use the "Intriguing Properties" title.
- RoFormer author order differs slightly between the arXiv page and the Neurocomputing record; the order above follows arXiv.
