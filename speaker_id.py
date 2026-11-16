"""
จำเสียงผู้พูด (Speaker Verification) ด้วยโมเดล microsoft/wavlm-base-plus-sv
- enroll(): บันทึกลายเสียงจากเสียงตัวอย่าง (ควรยาว 5 วินาทีขึ้นไป)
- verify(): เทียบเสียงก้อนใหม่กับลายเสียงที่บันทึกไว้ (cosine similarity)
โหลดโมเดลตอนใช้งานครั้งแรกเท่านั้น จึงไม่ทำให้ app.py เปิดช้าลง
"""
import os
import threading

import numpy as np

MODEL_ID = "microsoft/wavlm-base-plus-sv"
THRESHOLDS = {"strict": 0.86, "normal": 0.80, "lenient": 0.72}


class SpeakerGate:
    def __init__(self, profile_path: str):
        self.profile_path = profile_path
        self.model = None
        self.fe = None
        self.lock = threading.Lock()
        self.profile = None
        self.level = "normal"
        if os.path.exists(profile_path):
            try:
                self.profile = np.load(profile_path)
            except Exception:
                self.profile = None

    def _load(self):
        if self.model is not None:
            return
        import torch  # noqa: F401
        from transformers import AutoFeatureExtractor, WavLMForXVector
        print(f"🔊 กำลังโหลดโมเดลจำเสียง ({MODEL_ID}) ครั้งแรก...")
        self.fe = AutoFeatureExtractor.from_pretrained(MODEL_ID)
        self.model = WavLMForXVector.from_pretrained(MODEL_ID).eval()

    def _embed(self, audio_16k: np.ndarray) -> np.ndarray:
        import torch
        self._load()
        inputs = self.fe(audio_16k.astype(np.float32), sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            emb = self.model(**inputs).embeddings[0]
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.cpu().numpy()

    @staticmethod
    def _speech_only(audio_16k: np.ndarray) -> np.ndarray:
        import librosa
        if len(audio_16k) == 0:
            return audio_16k
        parts = librosa.effects.split(audio_16k.astype(np.float32), top_db=30)
        if len(parts) == 0:
            return audio_16k[:0]
        return np.concatenate([audio_16k[s:e] for s, e in parts])

    def enroll(self, audio_16k: np.ndarray) -> dict:
        speech = self._speech_only(audio_16k)
        if len(speech) < 16000 * 3:
            return {"ok": False, "message": "เสียงสั้นเกินไป กรุณาพูดต่อเนื่องอย่างน้อย 5 วินาที"}
        with self.lock:
            win = 16000 * 3
            embs = [self._embed(speech[i:i + win]) for i in range(0, max(1, len(speech) - win + 1), 16000 * 2)]
            prof = np.mean(embs, axis=0)
            prof = prof / (np.linalg.norm(prof) + 1e-9)
            self.profile = prof
            os.makedirs(os.path.dirname(self.profile_path) or ".", exist_ok=True)
            np.save(self.profile_path, prof)
        return {"ok": True, "message": f"จำเสียงสำเร็จ (ใช้เสียง {len(speech) / 16000:.1f} วินาที)"}

    def verify(self, audio_16k: np.ndarray) -> dict:
        if self.profile is None:
            return {"ok": True, "score": None, "reason": "no_profile"}
        speech = self._speech_only(audio_16k)
        if len(speech) < int(16000 * 0.6):
            return {"ok": None, "score": None, "reason": "too_short"}
        with self.lock:
            emb = self._embed(speech[-16000 * 4:])
        score = float(np.dot(emb, self.profile))
        return {"ok": score >= THRESHOLDS[self.level], "score": round(score, 3), "reason": ""}

    def clear(self):
        self.profile = None
        if os.path.exists(self.profile_path):
            os.remove(self.profile_path)
