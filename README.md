# Dialogue Summarization on SAMSum
**By:** Alejandro Silva  
**Data:** [Hugging Face — SAMSum (`knkarthick/samsum`)](https://huggingface.co/datasets/knkarthick/samsum)

---

## Overview
This project builds a reliable baseline to convert **multi-speaker chat threads** into **concise, faithful summaries** that speed support triage, improve CRM notes, and capture meeting outcomes.  
We follow a practical, experiment-driven process:

- **Backbone comparison:** BERT2BERT (encoder–decoder) vs GPT-2 (decoder-only) → **BERT2BERT wins** on validation ROUGE.  
- **Phase A (Training HPO):** tune lr/weight decay/label smoothing/warmup on a representative subset.  
- **Phase B (Decoding only):** optimize generation settings (beams, length penalty, no-repeat n-gram, max new tokens) on the **full validation set** to pick deployable defaults.

**Best config (Phase B):** `num_beams=6, length_penalty=1.0, no_repeat_ngram_size=3, max_new_tokens=64`  
**Validation ROUGE-Lsum:** ≈ **0.2483** (Phase A best ≈ **0.2475**)  
**Avg length:** ~**44.7** tokens

**Business impact (reading @ 200 wpm):**  
Full dialogue (~91.5 words) ≈ **27.5 s** → model summary (~43.3 words) ≈ **13.0 s** → **~14.5 s saved/chat (~53%)**  
(~**24 min per 100 chats**). With brevity controls (human-like length), savings can approach **~79%**.

---

## Project Goal
Deliver a **deployable summarization baseline** that produces **short, faithful** notes from chat transcripts—balancing **quality, latency, and cost**—and document a clear path for improvement.

---

## Business Problem & Stakeholders
- **Stakeholders:** Support/Success teams, CRM operations, analysts reviewing chat threads.
- **Need:** Faster, consistent summaries for case triage and record-keeping; predictable latency and low inference cost.
- **Non-goals:** Long-form abstractive writing or creative paraphrasing beyond factual summaries.

---

## Data
- **Source:** SAMSum (train/validation/test with human summaries).  
- **Quality:** One empty dialogue detected in **test** → removed. **Validation/test clean** thereafter.  
- **Scale:** Train has **14,731** dialogues; **28,935** unique dialogue words. Typical chat: **~94 words**; **~8 words/turn** (median 10).  
- **Human style:** Median summary **~18 words** (compression ≈ **0.28–0.30×** of source).

---

## Notebooks Overview
Three separated notebooks are in this repository

### 1) EDA, Data Load & Model Selection  
`notebooks/01_ACME Project.ipynb`
- Loads **SAMSum** via 🤗 Datasets: `knkarthick/samsum`.
- Cleans data (removes the single empty dialogue in **test**; validation/test otherwise clean).
- EDA: dialogue/summary length distributions, number of turns, compression ratios, vocabulary stats.
- **Backbone comparison**: **BERT2BERT** (encoder–decoder) vs **GPT-2** (decoder-only). BERT2BERT wins on validation ROUGE → chosen as baseline.

### 2) Training & Decoding (Phase A & B)  
`notebooks/02_ACME Project_Variant 2.ipynb`
- **Phase A (Training HPO)**: sweep `learning_rate`, `weight_decay`, `label_smoothing`, `warmup_ratio` on a representative subset; select best checkpoint by **ROUGE-Lsum**.  
  - Best Phase A: **ROUGE-Lsum ≈ 0.2475**, avg gen_len ≈ **47.7** tokens.
- **Phase B (Decoding Optimization)**: inference-only tuning on **full validation**: `num_beams`, `length_penalty`, `no_repeat_ngram_size`, `max_new_tokens`.  
  - Best Phase B: **ROUGE-Lsum ≈ 0.2483** with `beams=6, len_pen=1.0, no_repeat=3, max_new=64`, avg **44.7** tokens.
- Saves artifacts: `phaseA_results.csv`, `phaseB_grid.csv`, best checkpoint folder.

### 3) Sampling, Human Checks & Extra Metrics  
`notebooks/03_ACME Project_Variant 3.ipynb`
- Loads best Phase-B decoding and runs targeted samples (e.g., idx **654**, **114**, **25**).
- Side-by-side: dialogue, human reference, model summary; brief notes on humor, numeric drift, role attribution.
- Builds `validation_with_generated.csv` and computes aggregated stats (compression, lengths).
- (Optional) Snapshot latency (p50/p95) and extra semantic metrics if time permits.

---
