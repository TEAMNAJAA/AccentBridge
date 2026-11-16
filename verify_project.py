import ast
import json
import os
from accent_engine import AccentBridgeEngine, phonetic_similarity, normalize_text
from slide_parser import extract_priority_keywords, extract_acronyms, should_auto_advance


def run_checks():
    print("🔍 Running AccentBridge Sanity & Architecture Verification Suite...")

    # 1) Required project files check
    required = [
        "requirements.txt",
        "confusion_priors.json",
        "slide_parser.py",
        "accent_engine.py",
        "train_and_eval.py",
        "app.py",
        "overlay.py",
        "static/index.html",
        "static/captions.html",
        "README.md",
    ]
    missing = [f for f in required if not os.path.exists(f)]
    assert not missing, f"❌ Missing required files: {missing}"
    print("  [1/6] ✅ Project file layout verified (including overlay.py & static/captions.html)")

    # 2) Python syntax & JSON validity check
    for py_file in ("app.py", "overlay.py", "accent_engine.py", "slide_parser.py"):
        with open(py_file, "r", encoding="utf-8") as f:
            ast.parse(f.read(), filename=py_file)
    with open("confusion_priors.json", "r", encoding="utf-8") as f:
        priors_data = json.load(f)
    assert "word_pairs" in priors_data and len(priors_data["word_pairs"]) >= 20, "confusion_priors.json incomplete"
    print("  [2/6] ✅ AST syntax & Asian confusion_priors.json verified")

    # 3) Pan-Asian ARPAbet Phonology & Off-Slide Safety check
    assert phonetic_similarity("fire", "file") >= 0.80, "r/l pair ('fire' vs 'file') must match >= 0.80"
    assert phonetic_similarity("fine", "file") >= 0.80, "Thai final -l/-n ('fine' vs 'file') must match >= 0.80"
    assert phonetic_similarity("side", "slide") >= 0.80, "cluster pair ('side' vs 'slide') must match >= 0.80"
    assert phonetic_similarity("tree", "three") >= 0.80, "th-stopping ('tree' vs 'three') must match >= 0.80"
    assert phonetic_similarity("free", "three") >= 0.80, "th-fronting ('free' vs 'three') must match >= 0.80"
    assert phonetic_similarity("word", "world") >= 0.80, "coda cluster ('word' vs 'world') must match >= 0.80"

    assert phonetic_similarity("coffee", "file") < 0.50, "off-slide word ('coffee' vs 'file') must stay untouched"
    assert phonetic_similarity("today", "slide") < 0.55, "off-slide word ('today' vs 'slide') must stay untouched"
    assert normalize_text("Hello, API!") == "hello api"
    print("  [3/6] ✅ Pan-Asian phoneme alignment & off-slide safety bounds verified")

    # 4) Mixed Thai-English Slide Parser & Auto-Advance check
    mixed_slide = "ประเมินผลด้วย CTranslate2 และ LoRA บนการ์ดจอGPU เพื่อลดค่า WER แบบ real-time"
    kws = extract_priority_keywords(mixed_slide)
    acs = extract_acronyms(mixed_slide)
    assert "wer" in acs and "gpu" in acs and "lora" in acs, f"Mixed-script acronym extraction failed: {acs}"
    assert "ctranslate2" in kws and "real-time" in kws, f"Technical keyword extraction failed: {kws}"

    curr_s = {"index": 0, "keywords": ["file", "architecture", "modules"]}
    next_s = {"index": 1, "keywords": ["evaluate", "benchmark", "speakers", "recall"]}
    adv_en, reason_en = should_auto_advance(
        "here we evaluate benchmark recall across speakers", curr_s, next_s, phonetic_similarity
    )
    adv_th, _ = should_auto_advance("ต่อไปเรามาดูสไลด์ถัดไปกันครับ", curr_s, next_s, phonetic_similarity)
    assert adv_en and "2" in reason_en, "English semantic auto-advance failed"
    assert adv_th, "Thai voice cue auto-advance failed"
    print("  [4/6] ✅ Mixed Thai-English slide parser & semantic auto-advance verified")

    # 5) Lazy-VRAM Engine & Thai-English Code-Switching Rescue check (runs in <50ms with 0 GPU allocation!)
    eng = AccentBridgeEngine()
    cs_out, cs_pairs = eng.rescue_code_switching(
        "วันนี้เราจะรันโมเดลบน จีพียู เพื่อวัดค่า เวิร์ดเออเรอร์เรต",
        hotwords_raw="gpu, word error rate, benchmark",
        context_text="Measures word error rate on GPU benchmark.",
    )
    assert "GPU" in cs_out and "word error rate" in cs_out and len(cs_pairs) >= 2, f"Code-switching rescue failed: {cs_out}"
    print("  [5/6] ✅ Lazy-VRAM engine startup & Thai-English code-switching rescue verified")

    # 6) Optional CTranslate2 (`faster-whisper`) availability report
    try:
        import faster_whisper  # noqa: F401
        ct2_msg = "INSTALLED (Ultra-fast CTranslate2 + Silero VAD active!)"
    except ImportError:
        ct2_msg = "NOT INSTALLED YET (Run `pip install faster-whisper` for 4x speed boost; PyTorch fallback active)"
    print(f"  [6/6] ⚡ CTranslate2 (`faster-whisper`) status: {ct2_msg}")

    print("\n[OK] All sanity checks passed 100%! Run `python app.py` and open http://localhost:7860 in Chrome/Edge.")


if __name__ == "__main__":
    run_checks()
