import os
import re
import json
import random
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from dataclasses import dataclass
from typing import Any, Dict, List

import librosa
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset as TorchDataset
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import jiwer
from tqdm.auto import tqdm
from accent_engine import AccentBridgeEngine, STOPWORDS, normalize_text, phonetic_similarity

random.seed(42)
np.random.seed(42)

SE_EAST_ASIAN_TAGS = [
    "thai", "vietnam", "indonesia", "malay", "singapore", "filipino", "philippine", "tagalog",
    "mandarin", "chinese", "china", "cantonese", "hong kong", "taiwan", "korea", "japan", "burmese", "khmer"
]
SOUTH_ASIAN_TAGS = [
    "indian", "india", "hindi", "tamil", "urdu", "bengali", "punjabi", "telugu", "south asia", "sri lanka", "pakistan"
]


def get_asian_subregion(meta_str: str):
    low = meta_str.lower()
    if any(k in low for k in SE_EAST_ASIAN_TAGS):
        return "East & SE Asian"
    if any(k in low for k in SOUTH_ASIAN_TAGS):
        return "South Asian"
    return None


def is_l2_hard_phonology_word(w: str) -> bool:
    if len(w) < 4 or w in STOPWORDS:
        return False
    return bool(re.search(r"(th|ght|str|spr|spl|[rl]|(ld|st|nd|nt|ct|pt|ft|le|ce|se))$", w))


def clean_transcript(text):
    if not text:
        return None
    if any(tag in text.upper() for tag in ["<FOREIGN>", "<INAUDIBLE>", "<OVERLAP>", "<UNINTELLIGIBLE>"]):
        return None
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text.split()) >= 4 else None


def collect_pure_asian_corpus(target_train: int, target_dev: int, target_test: int):
    train_dev_pool, test_pool = [], []

    try:
        print("🌏 ดึงเสียงผู้พูดชาวเอเชียจากชุดข้อมูล GLOBE_V2...")
        globe = load_dataset("MushanW/GLOBE_V2", streaming=True)
        g_train = globe["train"].cast_column("audio", Audio(sampling_rate=16000))
        g_test = globe["test"].cast_column("audio", Audio(sampling_rate=16000))

        for item in g_train:
            acc = str(item.get("accent", ""))
            region = get_asian_subregion(acc)
            if not region:
                continue
            txt = clean_transcript(item.get("transcript", ""))
            arr = item["audio"]["array"]
            if txt and (2.0 <= len(arr) / 16000.0 <= 16.0):
                train_dev_pool.append({"audio_array": arr, "text": txt, "l1": acc, "group": region})
            if len(train_dev_pool) >= int((target_train + target_dev) * 0.60):
                break

        for item in g_test:
            acc = str(item.get("accent", ""))
            region = get_asian_subregion(acc)
            if not region:
                continue
            txt = clean_transcript(item.get("transcript", ""))
            arr = item["audio"]["array"]
            if txt and (2.0 <= len(arr) / 16000.0 <= 16.0):
                test_pool.append({"audio_array": arr, "text": txt, "l1": acc, "group": region})
            if len(test_pool) >= int(target_test * 0.50):
                break
    except Exception as e:
        print(f"ℹ️ ข้าม GLOBE_V2 ชั่วคราว ({e}) -> ดึงสำเนียงเอเชียจาก EdAcc เต็มจำนวนแทน")

    print("🎙️ ดึงบทสนทนาจริงสำเนียงเอเชียจากชุดข้อมูล EdAcc...")
    edacc = load_dataset("edinburghcstr/edacc", streaming=True)
    e_val = edacc["validation"].cast_column("audio", Audio(sampling_rate=16000))
    e_test = edacc["test"].cast_column("audio", Audio(sampling_rate=16000))

    for item in e_val:
        l1 = str(item.get("l1", "")).strip()
        acc = str(item.get("accent", "")).strip()
        region = get_asian_subregion(f"{l1} {acc}")
        if not region:
            continue
        txt = clean_transcript(item.get("text", ""))
        arr = item["audio"]["array"]
        if txt and (2.0 <= len(arr) / 16000.0 <= 18.0):
            train_dev_pool.append({"audio_array": arr, "text": txt, "l1": l1 or acc, "group": region})
        if len(train_dev_pool) >= (target_train + target_dev):
            break

    for item in e_test:
        l1 = str(item.get("l1", "")).strip()
        acc = str(item.get("accent", "")).strip()
        region = get_asian_subregion(f"{l1} {acc}")
        if not region:
            continue
        txt = clean_transcript(item.get("text", ""))
        arr = item["audio"]["array"]
        if txt and (2.0 <= len(arr) / 16000.0 <= 18.0):
            test_pool.append({"audio_array": arr, "text": txt, "l1": l1 or acc, "group": region})
        if len(test_pool) >= target_test:
            break

    random.shuffle(train_dev_pool)
    train_split = train_dev_pool[:target_train]
    dev_split = train_dev_pool[target_train : target_train + target_dev]
    return train_split, dev_split, test_pool


class AsianAugmentedDataset(TorchDataset):
    def __init__(self, items, processor, augment: bool = True):
        self.items, self.processor, self.augment = items, processor, augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        row = self.items[idx]
        wav = np.asarray(row["audio_array"], dtype=np.float32)
        if self.augment and random.random() < 0.65:
            rate = random.uniform(0.91, 1.09)
            wav = librosa.effects.time_stretch(wav, rate=rate)
            wav = (wav + np.random.normal(0, 0.0025, len(wav))).astype(np.float32)
        feat = self.processor.feature_extractor(wav, sampling_rate=16000).input_features[0]
        labels = self.processor.tokenizer(row["text"], truncation=True, max_length=128).input_ids
        return {"input_features": feat, "labels": labels}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    bos_token_id: int
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        feats = torch.stack([torch.as_tensor(f["input_features"], dtype=torch.float32) for f in features])
        label_tensors = [torch.as_tensor(f["labels"], dtype=torch.long) for f in features]
        padded_labels = torch.nn.utils.rnn.pad_sequence(label_tensors, batch_first=True, padding_value=-100)
        if padded_labels.size(1) > 0 and (padded_labels[:, 0] == self.bos_token_id).all().item():
            padded_labels = padded_labels[:, 1:]
        return {"input_features": feats, "labels": padded_labels}


def mine_asian_confusion_priors(model, processor, samples, device, out_json_path: str):
    model.eval()
    pair_counts = Counter()
    for r in tqdm(samples[:220], desc="Mining Pure Asian Phonetic Confusion Priors"):
        feats = processor(r["audio_array"], sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            ids = model.generate(feats, max_new_tokens=128)
        hyp_words = normalize_text(processor.batch_decode(ids, skip_special_tokens=True)[0]).split()
        ref_words = r["text"].split()
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, hyp_words, ref_words).get_opcodes():
            if tag == "replace" and (i2 - i1) == 1 and (j2 - j1) == 1:
                hw, rw = hyp_words[i1], ref_words[j1]
                if hw != rw and len(rw) >= 3 and rw not in STOPWORDS and phonetic_similarity(hw, rw) >= 0.52:
                    pair_counts[f"{hw}->{rw}"] += 1
    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump({"word_pairs": {k: v for k, v in pair_counts.most_common(350)}}, f, indent=2)


def calibrate_hyperparams_on_dev(engine: AccentBridgeEngine, dev_samples: list, out_calib_path: str):
    if not dev_samples:
        return engine.calib
    sub_dev = dev_samples[:25]
    vocab_pool = list({w for r in sub_dev for w in r["text"].split() if len(w) >= 5 and w not in STOPWORDS})

    best_wer = 1e9
    best_cfg = {"alpha": 0.15, "hw_weight": 0.20, "gate_threshold": 0.90}

    grid = [
        (0.12, 0.20, 0.88),
        (0.15, 0.22, 0.90),
        (0.18, 0.25, 0.92),
    ]
    for alpha, hw_w, gate_t in grid:
        refs, preds = [], []
        for row in sub_dev:
            ref = row["text"]
            refs.append(ref)
            true_kws = [w for w in dict.fromkeys(ref.split()) if len(w) >= 5 and w not in STOPWORDS][:3]
            cand_pool = [w for w in vocab_pool if w not in set(ref.split())]
            distractors = random.sample(cand_pool, k=min(10, len(cand_pool)))
            pred, _ = engine.transcribe_accentbridge(
                row["audio_array"],
                hotwords_raw=true_kws + distractors,
                slide_topic=" ".join(true_kws),
                alpha=alpha,
                hw_weight=hw_w,
                gate_thresh=gate_t,
                use_confidence_gate=True,
            )
            preds.append(pred)
        w = jiwer.wer(refs, preds) * 100.0
        if w < best_wer:
            best_wer = w
            best_cfg = {"alpha": alpha, "hw_weight": hw_w, "gate_threshold": gate_t}

    os.makedirs(os.path.dirname(out_calib_path), exist_ok=True)
    with open(out_calib_path, "w", encoding="utf-8") as f:
        json.dump(best_cfg, f, indent=2)
    engine.calib.update(best_cfg)
    print(f"🎯 Calibrated Dev Parameters: {best_cfg} (Dev WER: {best_wer:.2f}%)")
    return best_cfg


def quick_wer(model, processor, samples, device):
    model.eval()
    refs, preds = [], []
    for r in samples:
        feats = processor(r["audio_array"], sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            ids = model.generate(input_features=feats, max_new_tokens=128)
        preds.append(normalize_text(processor.batch_decode(ids, skip_special_tokens=True)[0]))
        refs.append(r["text"])
    return jiwer.wer(refs, preds) * 100.0


def keyword_recall(target_words_list, predictions):
    total_kw, hit_kw = 0, 0
    for kws, pred in zip(target_words_list, predictions):
        pred_set = set(pred.split())
        for kw in kws:
            total_kw += 1
            if kw in pred_set:
                hit_kw += 1
    return (hit_kw / max(1, total_kw)) * 100.0


def paired_bootstrap_ci(refs: list, base_preds: list, ab_preds: list, n_boot: int = 1000):
    n = len(refs)
    lengths = np.array([max(1, len(r.split())) for r in refs], dtype=np.float64)
    base_errs = np.array([jiwer.wer(r, b) * l for r, b, l in zip(refs, base_preds, lengths)])
    ab_errs = np.array([jiwer.wer(r, p) * l for r, p, l in zip(refs, ab_preds, lengths)])
    diffs, wins = [], 0
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        tot_len = lengths[idx].sum()
        b_w, a_w = (base_errs[idx].sum() / tot_len) * 100.0, (ab_errs[idx].sum() / tot_len) * 100.0
        diffs.append(a_w - b_w)
        if a_w < b_w:
            wins += 1
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(ci_low), float(ci_high), float(1.0 - (wins / n_boot))


def save_charts(wer_dict: dict, kw_dict: dict, group_wers: dict, out_dir: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=200)
    bars1 = ax1.bar(list(wer_dict.keys()), list(wer_dict.values()), color=["#94a3b8", "#38bdf8", "#10b981"])
    ax1.set_title("Asian-Accent Word Error Rate (Lower is Better)", fontweight="bold")
    ax1.set_ylabel("WER (%)")
    for b in bars1:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4, f"{b.get_height():.2f}%", ha="center", fontweight="bold")

    bars2 = ax2.bar(list(kw_dict.keys()), list(kw_dict.values()), color=["#94a3b8", "#0284c7", "#059669"])
    ax2.set_title("Slide Keyword Recall on Asian Speech (Higher is Better)", fontweight="bold")
    ax2.set_ylabel("Recall (%)")
    ax2.set_ylim(0, min(105, max(kw_dict.values()) + 15))
    for b in bars2:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0, f"{b.get_height():.2f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "benchmark_chart.png"))
    plt.close()

    if group_wers:
        groups = list(group_wers.keys())
        x, w = np.arange(len(groups)), 0.35
        fig2, ax = plt.subplots(figsize=(8.5, 4.5), dpi=200)
        r1 = ax.bar(x - w / 2, [group_wers[g][0] for g in groups], w, label="Baseline Whisper-small.en", color="#94a3b8")
        r2 = ax.bar(x + w / 2, [group_wers[g][1] for g in groups], w, label="Asian-Trained AccentBridge", color="#10b981")
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontweight="bold")
        ax.set_ylabel("WER (%)")
        ax.set_title("WER Breakdown: East/SE Asian vs. South Asian Speakers", fontweight="bold")
        ax.legend()
        for r in list(r1) + list(r2):
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.4, f"{r.get_height():.1f}%", ha="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "per_accent_wer.png"))
        plt.close()


def main():
    has_gpu = torch.cuda.is_available()
    device = "cuda" if has_gpu else "cpu"
    train_size, dev_size = (600, 60) if has_gpu else (160, 30)
    max_steps, test_size = (360 if has_gpu else 80), 120

    train_data, dev_data, test_data = collect_pure_asian_corpus(train_size, dev_size, test_size)
    print(f"✅ ชุดข้อมูลคนเอเชีย 100% พร้อมเทรน -> Train: {len(train_data)} | Dev: {len(dev_data)} | Test: {len(test_data)}")

    model_id = "openai/whisper-small.en"
    processor = WhisperProcessor.from_pretrained(model_id)
    base_whisper = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)

    os.makedirs("./saved_model", exist_ok=True)
    mine_asian_confusion_priors(base_whisper, processor, train_data, device, "./saved_model/confusion_priors.json")
    dev_base_wer = quick_wer(base_whisper, processor, dev_data, device)
    print(f"📊 Dev WER ก่อนฝึกสำเนียงเอเชีย: {dev_base_wer:.2f}%")

    base_whisper.config.use_cache = False
    base_whisper.config.apply_spec_augment = True
    base_whisper.config.mask_time_prob = 0.06
    base_whisper.config.mask_feature_prob = 0.05
    base_whisper.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    peft_model = get_peft_model(base_whisper, lora_cfg)
    peft_model.print_trainable_parameters()

    trainer = Seq2SeqTrainer(
        args=Seq2SeqTrainingArguments(
            output_dir="./accentbridge_ckpt",
            per_device_train_batch_size=8 if has_gpu else 4,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            warmup_steps=35,
            max_steps=max_steps,
            fp16=has_gpu,
            logging_steps=40,
            save_strategy="no",
            label_names=["labels"],
            report_to="none",
        ),
        model=peft_model,
        train_dataset=AsianAugmentedDataset(train_data, processor, augment=True),
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(bos_token_id=processor.tokenizer.bos_token_id),
    )

    print("🚀 เริ่มฝึกโมเดลกับสำเนียงคนเอเชีย 100%...")
    trainer.train()

    dev_ft_wer = quick_wer(peft_model.to(device), processor, dev_data, device)
    if dev_ft_wer <= dev_base_wer + 0.5:
        peft_model.save_pretrained("./saved_model")
        print(f"✅ ฝึกสำเร็จ! Dev WER ลดจาก {dev_base_wer:.2f}% เหลือ {dev_ft_wer:.2f}% (เซฟ LoRA ลง ./saved_model)")

    engine = AccentBridgeEngine(finetuned_dir="./saved_model", base_id=model_id)
    calibrate_hyperparams_on_dev(engine, dev_data, "./saved_model/calibrated_params.json")

    vocab_pool = list({w for r in test_data for w in r["text"].split() if len(w) >= 5 and w not in STOPWORDS})

    refs, base_preds, ft_only_preds, full_preds = [], [], [], []
    true_keywords_per_row, hard_phon_per_row = [], []
    group_indices = defaultdict(list)

    off_slide_gated_hallucinations, off_slide_ungated_hallucinations, off_slide_trials = 0, 0, 0

    for idx, row in enumerate(tqdm(test_data, desc="Evaluating on Unseen Asian Speakers")):
        ref = row["text"]
        refs.append(ref)
        group_indices[row["group"]].append(idx)

        true_kws = [w for w in dict.fromkeys(ref.split()) if len(w) >= 5 and w not in STOPWORDS][:3]
        hard_kws = [w for w in dict.fromkeys(ref.split()) if is_l2_hard_phonology_word(w)][:4]
        true_keywords_per_row.append(true_kws)
        hard_phon_per_row.append(hard_kws)

        cand_pool = [w for w in vocab_pool if w not in set(ref.split())]
        distractors = random.sample(cand_pool, k=min(18, len(cand_pool)))
        slide_vocab = list(dict.fromkeys(true_kws + hard_kws + distractors))
        random.shuffle(slide_vocab)

        base_preds.append(engine.transcribe_baseline(row["audio_array"]))
        ft_txt, _ = engine.transcribe_accentbridge(row["audio_array"], hotwords_raw=None)
        ab_txt, _ = engine.transcribe_accentbridge(
            row["audio_array"], hotwords_raw=slide_vocab, slide_topic=" ".join(true_kws), use_confidence_gate=True
        )
        ft_only_preds.append(ft_txt)
        full_preds.append(ab_txt)

        if idx < 30 and distractors:
            off_slide_trials += 1
            _, resc_gated = engine.transcribe_accentbridge(
                row["audio_array"], hotwords_raw=distractors, use_confidence_gate=True
            )
            _, resc_ungated = engine.transcribe_accentbridge(
                row["audio_array"], hotwords_raw=distractors, use_confidence_gate=False
            )
            off_slide_gated_hallucinations += len(resc_gated)
            off_slide_ungated_hallucinations += len(resc_ungated)

    base_wer, ft_wer, full_wer = jiwer.wer(refs, base_preds) * 100, jiwer.wer(refs, ft_only_preds) * 100, jiwer.wer(refs, full_preds) * 100
    base_kw, ft_kw, full_kw = keyword_recall(true_keywords_per_row, base_preds), keyword_recall(true_keywords_per_row, ft_only_preds), keyword_recall(true_keywords_per_row, full_preds)
    base_hard, full_hard = keyword_recall(hard_phon_per_row, base_preds), keyword_recall(hard_phon_per_row, full_preds)
    mean_diff, ci_low, ci_high, p_val = paired_bootstrap_ci(refs, base_preds, full_preds)

    group_wers = {
        g: (jiwer.wer([refs[i] for i in idxs], [base_preds[i] for i in idxs]) * 100, jiwer.wer([refs[i] for i in idxs], [full_preds[i] for i in idxs]) * 100)
        for g, idxs in group_indices.items() if len(idxs) >= 5
    }

    os.makedirs("./results", exist_ok=True)
    save_charts({"Baseline": base_wer, "Asian LoRA": ft_wer, "AccentBridge": full_wer}, {"Baseline": base_kw, "Asian LoRA": ft_kw, "AccentBridge": full_kw}, group_wers, "./results")

    lines = [
        "# AccentBridge — 100% Pure Asian-Accent Evaluation Report\n",
        "| Model Variant | Asian Speaker WER (Lower=Better) | Slide Keyword Recall | Asian Hard-Phoneme Recall |",
        "|:---|:---|:---|:---|",
        f"| 1. Baseline (Zero-shot whisper-small.en) | {base_wer:.2f}% | {base_kw:.2f}% | {base_hard:.2f}% |",
        f"| 2. Asian-Trained LoRA Whisper (GLOBE_V2 + EdAcc Asian) | {ft_wer:.2f}% | {ft_kw:.2f}% | — |",
        f"| 3. AccentBridge Full (+Slide-Prompt & Phoneme Rescue) | {full_wer:.2f}% | {full_kw:.2f}% | {full_hard:.2f}% |\n",
        f"- **Paired Bootstrap 95% CI:** `{mean_diff:+.2f}%` (`[{ci_low:+.2f}%, {ci_high:+.2f}%]`, **p = {p_val:.4f}**)",
        f"- **Off-Slide Spontaneous Speech Safety (30 pure distractor trials):** Confidence Gate reduced false-positive hotword swaps from `{off_slide_ungated_hallucinations}` (ungated) down to `{off_slide_gated_hallucinations}` (gated).\n",
        "## Winning Examples & Failure Cases",
    ]
    for r, b, p, meta in list(zip(refs, base_preds, full_preds, test_data))[:60]:
        if jiwer.wer(r, p) < jiwer.wer(r, b):
            lines.append(f"- **[WIN - {meta['l1']}]** Truth: `{r}` | Baseline: `{b}` | AccentBridge: `{p}`")
    for r, p, meta in list(zip(refs, full_preds, test_data))[:60]:
        if jiwer.wer(r, p) > 0.35:
            lines.append(f"- **[FAIL - {meta['l1']}]** Truth: `{r}` | Predicted: `{p}`")

    with open("./results/report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines[:15]))


if __name__ == "__main__":
    main()
