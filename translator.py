"""
แปลซับหลายภาษาแบบออฟไลน์ด้วย facebook/nllb-200-distilled-600M (รองรับ 200 ภาษา)
โหลดโมเดลครั้งแรกตอนเปิดใช้การแปลเท่านั้น (ดาวน์โหลดประมาณ 2.5 GB ครั้งเดียว)
"""
import threading
from functools import lru_cache

MODEL_ID = "facebook/nllb-200-distilled-600M"

# รหัสภาษาที่หน้าเว็บใช้ -> รหัสของ NLLB
LANGS = {
    "en": ("English", "eng_Latn"),
    "th": ("ไทย", "tha_Thai"),
    "zh": ("中文", "zho_Hans"),
    "ja": ("日本語", "jpn_Jpan"),
    "ko": ("한국어", "kor_Hang"),
    "vi": ("Tiếng Việt", "vie_Latn"),
    "id": ("Bahasa Indonesia", "ind_Latn"),
    "ms": ("Bahasa Melayu", "zsm_Latn"),
    "fr": ("Français", "fra_Latn"),
    "es": ("Español", "spa_Latn"),
}


class Translator:
    def __init__(self):
        self.tok = None
        self.model = None
        self.device = "cpu"
        self.lock = threading.Lock()

    def _load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🌐 กำลังโหลดโมเดลแปลภาษา ({MODEL_ID}) ครั้งแรก อาจใช้เวลาหลายนาที...")
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(self.device).eval()
        print("✅ โมเดลแปลภาษาพร้อมใช้งาน")

    def translate(self, text: str, src: str, tgt: str) -> str:
        text = (text or "").strip()
        if not text or src == tgt or src not in LANGS or tgt not in LANGS:
            return ""
        return self._cached(text, src, tgt)

    @lru_cache(maxsize=512)
    def _cached(self, text: str, src: str, tgt: str) -> str:
        import torch
        with self.lock:
            self._load()
            self.tok.src_lang = LANGS[src][1]
            enc = self.tok(text, return_tensors="pt", truncation=True, max_length=200).to(self.device)
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    forced_bos_token_id=self.tok.convert_tokens_to_ids(LANGS[tgt][1]),
                    max_new_tokens=160,
                    num_beams=1,
                )
            return self.tok.batch_decode(out, skip_special_tokens=True)[0].strip()
