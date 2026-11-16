import argparse
import ast
import json
import os
import re
import numpy as np
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOICE_DIR = os.path.join(BASE_DIR, "voice_dataset")
SAVED_HF_DIR = os.path.join(BASE_DIR, "saved_model")
SAVED_CT2_DIR = os.path.join(BASE_DIR, "saved_model_ct2")
TURBO_REPO = "deepdml/faster-whisper-large-v3-turbo-ct2"

# ประโยคฝึกออกเสียงเน้นจุดที่สำเนียงเอเชียมักเพี้ยนบ่อยที่สุด + คำว่า point / PowerPoint / ชื่อเฉพาะ
CORE_DRILL_SENTENCES = [
    "My main point on this slide is our real time accuracy and low latency.",
    "Let me point out three core API modules in our open source pipeline.",
    "Today please look at the audio file on the right slide in PowerPoint.",
    "Here we evaluate word error rate and target keyword recall on GPU.",
    "AccentBridge was built for the Veritas benchmark on unseen Asian speakers.",
    "At this point our confidence gate keeps spontaneous subtitles untouched.",
    "First point is speed and the second point is phonetic rescue.",
    "Please check the five files and script results on the next slide.",
    "Silero voice activity detection cuts sentences cleanly when I pause.",
    "Our baseline model reduces errors across Thai and Asian accents.",
    "This dataset measures recall without changing off script speech.",
    "Finally let us review the three key points of our system architecture.",
]


def _gpu_info() -> tuple:
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            name = torch.cuda.get_device_name(0)
            ctype = "float16" if vram_gb >= 7.5 else "int8_float16"
            return True, name, round(vram_gb, 1), ctype
    except Exception:
        pass
    return False, "CPU", 0.0, "int8"


def apply_turbo_upgrade():
    has_gpu, gpu_name, vram_gb, best_ctype = _gpu_info()
    print(f"🖥️ ตรวจพบอุปกรณ์: {gpu_name} (VRAM: {vram_gb} GB) -> โหมดคำนวณที่เหมาะสม: {best_ctype}")

    app_path = os.path.join(BASE_DIR, "app.py")
    if os.path.exists(app_path):
        with open(app_path, "r", encoding="utf-8") as f:
            src = f.read()
        updated = src
        # 1) เปลี่ยนโมเดลเริ่มต้นบนการ์ดจอเป็น Whisper Large-v3-Turbo CTranslate2
        old_m = '"small" if _has_cuda() else "base"'
        new_m = f'"{TURBO_REPO}" if _has_cuda() else "base"'
        if old_m in updated:
            updated = updated.replace(old_m, new_m)
        # 2) ปรับ compute_type อัตโนมัติตามขนาด VRAM ป้องกันแรมการ์ดจอเต็ม
        old_ct = 'self._compute_type = "float16" if self._device == "cuda" else "int8"'
        new_ct = f'self._compute_type = "{best_ctype}" if self._device == "cuda" else "int8"'
        if old_ct in updated:
            updated = updated.replace(old_ct, new_ct)
        # 3) ส่งคำศัพท์สไลด์และชื่อเฉพาะเป็น Lexicon Bias (initial_prompt) ให้ทุกภาษา
        if 'if slide_prompt and w_lang == "en":' in updated:
            updated = updated.replace('if slide_prompt and w_lang == "en":', "if slide_prompt:")

        if updated != src:
            ast.parse(updated, filename="app.py")
            with open(app_path, "w", encoding="utf-8") as f:
                f.write(updated)
            print(f"✅ อัปเกรด app.py เป็น {TURBO_REPO} ({best_ctype}) + Lexicon Biasing ทุกภาษาสำเร็จ!")
        else:
            print("✅ app.py ตั้งค่าเป็นโหมดความแม่นยำสูงไว้เรียบร้อยแล้ว")

    if has_gpu:
        try:
            from faster_whisper import WhisperModel
            print(f"⬇️ กำลังตรวจสอบ/ดาวน์โหลดโมเดล {TURBO_REPO} เตรียมไว้ในเครื่อง...")
            WhisperModel(TURBO_REPO, device="cuda", compute_type=best_ctype)
            print("🚀 โมเดล Whisper Large-v3-Turbo พร้อมใช้งานทันที!")
        except ImportError:
            print("⚠️ ยังไม่ได้ลง faster-whisper กรุณารัน: pip install faster-whisper")
        except Exception as e:
            print(f"ℹ️ หมายเหตุการโหลดโมเดลล่วงหน้า: {e}")


def collect_training_prompts(target_count: int) -> list:
    prompts = list(CORE_DRILL_SENTENCES)
    try:
        from slide_parser import extract_slides_from_file
        for fname in os.listdir(BASE_DIR):
            if fname.lower().endswith((".pptx", ".pdf", ".txt")) and fname != "requirements.txt":
                slides = extract_slides_from_file(os.path.join(BASE_DIR, fname))
                for s in slides:
                    for line in re.split(r"[.!?\n]+", s.get("text", "")):
                        cl = re.sub(r"\[Speaker Notes\]", "", line).strip()
                        words = re.findall(r"[A-Za-z0-9'\-]+", cl)
                        if 5 <= len(words) <= 18 and not re.search(r"[\u0E00-\u0E7F]", cl):
                            if cl not in prompts:
                                prompts.append(cl)
    except Exception:
        pass
    return prompts[:target_count]


def record_interactive_dataset(prompts: list) -> list:
    try:
        import sounddevice as sd
    except ImportError:
        print("❌ ต้องติดตั้ง sounddevice ก่อนอัดเสียง: pip install sounddevice")
        return []

    os.makedirs(VOICE_DIR, exist_ok=True)
    manifest_path = os.path.join(VOICE_DIR, "manifest.json")
    records = []
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            records = []

    print("\n" + "=" * 68)
    print("🎙️ โหมดอัดเสียงฝึก AI เฉพาะตัวคุณ (Personal Voice LoRA Studio)")
    print("   • กด Enter เพื่อเริ่มอัด -> อ่านประโยคด้วยสำเนียงปกติ -> กด Enter เพื่อหยุด")
    print("   • พิมพ์ s แล้ว Enter = ข้ามข้อนี้ | พิมพ์ q แล้ว Enter = พอแค่นี้แล้วเริ่มเทรนเลย!")
    print("=" * 68)

    sr = 16000
    idx = 0
    while idx < len(prompts):
        sent = prompts[idx]
        cmd = input(f"\n[{idx + 1}/{len(prompts)}] เตรียมอ่าน: \"{sent}\"\n   กด Enter เพื่อเริ่มอัด (s=ข้าม, q=เริ่มเทรน): ").strip().lower()
        if cmd == "q":
            break
        if cmd == "s":
            idx += 1
            continue

        frames = []
        def _cb(indata, _f, _t, _s):
            frames.append(indata[:, 0].copy())

        stream = sd.InputStream(samplerate=sr, channels=1, dtype="float32", callback=_cb)
        stream.start()
        input("   🔴 กำลังอัดเสียง... อ่านจบแล้วกด Enter เพื่อหยุด: ")
        stream.stop()
        stream.close()

        if not frames:
            continue
        wav = np.concatenate(frames).astype(np.float32)
        wav = wav - float(np.mean(wav))
        rms = float(np.sqrt(np.mean(wav * wav) + 1e-12))
        dur = len(wav) / sr
        if dur < 0.8 or rms < 0.004:
            print("   ⚠️ เสียงสั้นหรือเบาเกินไป ลองอัดข้อนี้ใหม่อีกครั้งนะครับ")
            continue

        peak = float(np.percentile(np.abs(wav), 99.2))
        if peak > 1e-5:
            wav = np.clip(wav / peak, -0.98, 0.98)

        fname = f"utt_{len(records) + 1:03d}.wav"
        fpath = os.path.join(VOICE_DIR, fname)
        sf.write(fpath, wav, sr)
        records.append({"audio": fpath, "text": sent})
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"   ✅ บันทึกแล้ว ({dur:.1f} วินาที) — รวมมีเสียงตัวอย่าง {len(records)} ประโยค")
        idx += 1

    return records


def load_saved_dataset() -> list:
    manifest_path = os.path.join(VOICE_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        return []
    with open(manifest_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    return [it for it in items if os.path.exists(it.get("audio", ""))]


def spec_augment_mel(feats, torch_mod):
    # สุ่มบังแถบความถี่และช่วงเวลาเล็กน้อย (SpecAugment) ให้โมเดลทนต่อเสียงรบกวนและจังหวะพูดจริง
    aug = feats.clone()
    _, n_mels, n_steps = aug.shape
    f_w = int(np.random.randint(2, 8))
    f0 = int(np.random.randint(0, max(1, n_mels - f_w)))
    aug[:, f0 : f0 + f_w, :] = 0.0
    if n_steps > 40:
        t_w = int(np.random.randint(4, 18))
        t0 = int(np.random.randint(0, n_steps - t_w))
        aug[:, :, t0 : t0 + t_w] = 0.0
    return aug


def train_personal_lora(records: list, epochs: int = 6):
    if len(records) < 3:
        print("⚠️ ต้องมีเสียงอัดอย่างน้อย 3 ประโยคขึ้นไปจึงจะเริ่มเทรนได้ครับ")
        return

    import torch
    from difflib import SequenceMatcher
    from jiwer import wer
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, WhisperForConditionalGeneration, WhisperProcessor
    from accent_engine import STOPWORDS, normalize_text

    has_gpu, _, vram_gb, best_ctype = _gpu_info()
    device = "cuda" if has_gpu else "cpu"
    base_id = "openai/whisper-small.en" if (has_gpu and vram_gb >= 3.8) else "openai/whisper-base.en"

    print(f"\n🧠 เริ่มกระบวนการเทรน Personal LoRA บน {base_id} ({device})...")
    proc = WhisperProcessor.from_pretrained(base_id)
    base_m = WhisperForConditionalGeneration.from_pretrained(base_id).to(device)
    base_m.eval()

    samples = []
    refs, before_hyps = [], []
    learned_pairs = {}

    with torch.no_grad():
        for it in records:
            wav, sr = sf.read(it["audio"], dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            ref_norm = normalize_text(it["text"])
            feats = proc(wav, sampling_rate=sr, return_tensors="pt").input_features.to(device)
            pred_ids = base_m.generate(feats, max_new_tokens=96)
            hyp_norm = normalize_text(proc.batch_decode(pred_ids, skip_special_tokens=True)[0])
            refs.append(ref_norm)
            before_hyps.append(hyp_norm)
            samples.append((wav, ref_norm))

            hw_l, rw_l = hyp_norm.split(), ref_norm.split()
            for tag, i1, i2, j1, j2 in SequenceMatcher(None, hw_l, rw_l).get_opcodes():
                if tag == "replace" and (i2 - i1) == 1 and (j2 - j1) == 1:
                    h_w, r_w = hw_l[i1], rw_l[j1]
                    if h_w != r_w and r_w not in STOPWORDS and h_w not in STOPWORDS:
                        learned_pairs[f"{h_w}->{r_w}"] = learned_pairs.get(f"{h_w}->{r_w}", 0) + 6

    before_wer = round(wer(refs, before_hyps) * 100, 2)
    print(f"📊 ค่าความคลาดเคลื่อน (WER) ก่อนเทรนเสียงคุณ: {before_wer}%")

    # ตั้งค่า Multi-Layer LoRA ครอบคลุมทั้ง Attention Query/Key/Value/Out
    base_m.config.use_cache = False
    base_m.enable_input_require_grads()
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    peft_m = get_peft_model(base_m, lora_cfg)
    peft_m.train()

    total_steps = len(samples) * epochs
    opt = torch.optim.AdamW(peft_m.parameters(), lr=1.2e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps))

    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        order = np.random.permutation(len(samples))
        for idx in order:
            wav, ref_norm = samples[int(idx)]
            # ผสมเสียงรบกวนเบาๆ ให้โมเดลเก่งในสภาพห้องจริง
            noisy = (wav + np.random.normal(0, 0.002, len(wav))).astype(np.float32)
            feats = proc(noisy, sampling_rate=16000, return_tensors="pt").input_features.to(device)
            if ep > 1:
                feats = spec_augment_mel(feats, torch)
            lbl = proc.tokenizer(ref_norm, return_tensors="pt", truncation=True, max_length=128).input_ids.to(device)
            opt.zero_grad()
            loss = peft_m(input_features=feats, labels=lbl).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(peft_m.parameters(), 1.0)
            opt.step()
            sched.step()
            ep_loss += float(loss.item())
        print(f"   • Epoch {ep}/{epochs} — Loss: {ep_loss / len(samples):.4f}")

    peft_m.eval()
    merged = peft_m.merge_and_unload()

    after_hyps = []
    with torch.no_grad():
        for wav, _ in samples:
            feats = proc(wav, sampling_rate=16000, return_tensors="pt").input_features.to(device)
            pred_ids = merged.generate(feats, max_new_tokens=96, no_repeat_ngram_size=3, repetition_penalty=1.18)
            after_hyps.append(normalize_text(proc.batch_decode(pred_ids, skip_special_tokens=True)[0]))
    after_wer = round(wer(refs, after_hyps) * 100, 2)

    os.makedirs(SAVED_HF_DIR, exist_ok=True)
    save_m = merged.half() if device == "cuda" else merged
    save_m.save_pretrained(SAVED_HF_DIR)
    proc.save_pretrained(SAVED_HF_DIR)
    # บันทึก tokenizer.json แบบ Fast เพื่อให้ CTranslate2 นำไปใช้ต่อได้ทันที 100%
    AutoTokenizer.from_pretrained(base_id, use_fast=True).save_pretrained(SAVED_HF_DIR)

    # อัปเดตคู่คำสับสนเฉพาะตัวคุณลงทั้งสองไฟล์
    for p_path in (os.path.join(BASE_DIR, "confusion_priors.json"), os.path.join(SAVED_HF_DIR, "confusion_priors.json")):
        cur_p = {"word_pairs": {}, "thai_translit_stems": {}}
        if os.path.exists(p_path):
            try:
                with open(p_path, "r", encoding="utf-8") as f:
                    cur_p.update(json.load(f))
            except Exception:
                pass
        cur_p.setdefault("word_pairs", {}).update(learned_pairs)
        with open(p_path, "w", encoding="utf-8") as f:
            json.dump(cur_p, f, ensure_ascii=False, indent=2)

    # แปลงโมเดลที่เทรนเสร็จเป็นฟอร์แมต CTranslate2 ความเร็วสูง (./saved_model_ct2)
    try:
        import ctranslate2
        copy_f = [ fn for fn in ("tokenizer.json", "preprocessor_config.json", "vocab.json", "merges.txt", "normalizer.json") if os.path.exists(os.path.join(SAVED_HF_DIR, fn)) ]
        conv = ctranslate2.converters.TransformersConverter(SAVED_HF_DIR, copy_files=copy_f, load_as_float16=has_gpu)
        q_type = "float16" if has_gpu else "int8"
        conv.convert(SAVED_CT2_DIR, quantization=q_type, force=True)
        print(f"⚡ แปลงโมเดลเสียงของคุณเป็น CTranslate2 ที่ {SAVED_CT2_DIR} สำเร็จ!")
    except Exception as e:
        print(f"ℹ️ บันทึกโมเดลหลักที่ ./saved_model แล้ว (หมายเหตุ CTranslate2 converter: {e})")

    print("\n" + "=" * 68)
    print(f"🎉 เทรนสำเร็จ! WER ก่อนเทรน: {before_wer}%  -->  หลังเทรนเสียงคุณ: {after_wer}%")
    if learned_pairs:
        print(f"🎯 จับคู่คำเพี้ยนเฉพาะตัวคุณเพิ่มอัตโนมัติ: {list(learned_pairs.keys())[:10]}")
    print("👉 สั่งรัน `python app.py` เพื่อใช้งานโมเดลเสียงของคุณได้ทันที!")
    print("=" * 68)


def main():
    parser = argparse.ArgumentParser(description="AccentBridge Turbo & Personal LoRA Trainer")
    parser.add_argument("--turbo", action="store_true", help="อัปเกรดเป็น Whisper Large-v3-Turbo ทันทีโดยไม่ต้องอัดเสียง")
    parser.add_argument("--retrain-saved", action="store_true", help="เทรนซ้ำจากไฟล์เสียงที่เคยอัดไว้ใน voice_dataset/")
    parser.add_argument("--count", type=int, default=15, help="จำนวนประโยคที่ต้องการอ่านอัดเสียง (ค่าเริ่มต้น 15)")
    parser.add_argument("--epochs", type=int, default=6, help="จำนวนรอบการเทรน LoRA (ค่าเริ่มต้น 6)")
    args = parser.parse_args()

    apply_turbo_upgrade()
    if args.turbo:
        return

    if args.retrain_saved:
        recs = load_saved_dataset()
    else:
        prompts = collect_training_prompts(args.count)
        recs = record_interactive_dataset(prompts)

    if recs:
        train_personal_lora(recs, epochs=args.epochs)


if __name__ == "__main__":
    main()
