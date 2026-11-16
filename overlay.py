"""
AccentBridge Overlay — ซับลอยโปร่งใส อยู่บนสุดทับทุกโปรแกรม (Windows)
• อัปเกรดระบบไมค์ตรงเป็น Breath-Synced Adaptive VAD (ตัดประโยคตามจังหวะหยุดหายใจ 300ms + พรีวิวสดระหว่างพูดทุก 650ms)
• เชื่อมต่อตรงกับเอนจิน CTranslate2 (`faster-whisper`) + Silero Neural VAD ความเร็วระดับ ~50–90ms
• เลือกภาษาแปลเองได้โดยตรงจากปุ่ม [🌐 แปล: ...] บนซับลอย + ปุ่ม [🎙️ ปิดไมค์ชั่วคราว] กันซับรวนบน localhost
"""
import io
import json
import re
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
import unicodedata
import urllib.parse
import urllib.request
import wave

BASE_URLS = ["http://127.0.0.1:7860", "http://localhost:7860"]
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

TRANSPARENT = "#010203"
BOX = "#111111"
TEXT = "#ffffff"
HIGHLIGHT = "#ffd84d"
INTERIM = "#b8d4ff"
ORIG = "#d6d6d6"
MUTE_BG = "#6e1b26"
MUTE_BORDER = "#f85149"
UNMUTE_BG = "#16202c"
UNMUTE_BORDER = "#2ea043"
LANG_BG = "#172336"
LANG_BORDER = "#388bfd"
VAD_BG = "#0f5132"
VAD_BORDER = "#3fb950"
MAX_CHARS = 46
SIZES = {"S": 26, "M": 34, "L": 44, "XL": 56}
LEADING_VOWELS = set("เแโใไ")

TRANSLATE_TARGETS = [
    ("sync", "🔄 ตามหน้าเว็บ (Sync with Web)"),
    ("none", "🚫 ไม่แปล (แสดงต้นฉบับ)"),
    ("en", "🇬🇧 English (อังกฤษ)"),
    ("th", "🇹🇭 ภาษาไทย (Thai)"),
    ("zh", "🇨🇳 中文 (จีน)"),
    ("ja", "🇯🇵 日本語 (ญี่ปุ่น)"),
    ("ko", "🇰🇷 한국어 (เกาหลี)"),
    ("vi", "🇻🇳 Tiếng Việt (เวียดนาม)"),
    ("id", "🇮🇩 Bahasa Indonesia (อินโดนีเซีย)"),
    ("ms", "🇲🇾 Bahasa Melayu (มาเลย์)"),
    ("hi", "🇮🇳 हिन्दी (ฮินดี)"),
    ("es", "🇪🇸 Español (สเปน)"),
    ("fr", "🇫🇷 Français (ฝรั่งเศส)"),
    ("de", "🇩🇪 Deutsch (เยอรมัน)"),
    ("ru", "🇷🇺 Русский (รัสเซีย)"),
]


def tokenize(text):
    toks = []
    for piece in re.findall(r"\s+|[A-Za-z0-9'\-.,!?;:]+|.", text or ""):
        if piece.isspace():
            toks.append((" ", True))
            continue
        if len(piece) == 1 and toks and not toks[-1][1]:
            prev = toks[-1][0]
            if unicodedata.category(piece) in ("Mn", "Mc") or piece == "ำ" or prev[-1] in LEADING_VOWELS:
                toks[-1] = (prev + piece, False)
                continue
        toks.append((piece, False))
    return toks


def _clean_word_key(tok: str) -> str:
    return re.sub(r"^[^A-Za-z0-9']+|[^A-Za-z0-9']+$", "", (tok or "").lower())


def _encode_multipart_wav(wav_bytes: bytes, fields: dict) -> tuple[bytes, str]:
    boundary = f"----AccentBridgeBoundary{int(time.time() * 1000)}"
    buf = io.BytesIO()
    for k, v in fields.items():
        buf.write(f"--{boundary}\r\n".encode("utf-8"))
        buf.write(f'Content-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode("utf-8"))
    buf.write(f"--{boundary}\r\n".encode("utf-8"))
    buf.write(b'Content-Disposition: form-data; name="audio"; filename="vad.wav"\r\n')
    buf.write(b"Content-Type: audio/wav\r\n\r\n")
    buf.write(wav_bytes)
    buf.write(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return buf.getvalue(), f"multipart/form-data; boundary={boundary}"


class BreathVADSegmenter:
    """
    ตัวตัดประโยคตามจังหวะพูดและหยุดหายใจอัจฉริยะ (30ms Frame Adaptive VAD):
      - เก็บ Pre-Roll ล่วงหน้า 240ms พยัญชนะต้นคำไม่แหว่ง
      - ส่งภาพรวมเสียงระหว่างพูดทุก 650ms (is_final=False) ให้ซับไหลสดๆ ขณะกำลังพูด
      - ตัดจบประโยคทันทีเมื่อเว้นจังหวะหายใจ ~300ms (is_final=True) ประโยคไม่ขาดครึ่ง!
    """

    def __init__(
        self,
        sr: int = 16000,
        frame_ms: int = 30,
        preroll_ms: int = 240,
        silence_end_ms: int = 300,
        min_speech_ms: int = 210,
        stream_interval_ms: int = 650,
        max_utterance_s: float = 8.0,
    ):
        self.sr = sr
        self.frame_len = int(sr * frame_ms / 1000)
        self.preroll_frames = max(1, preroll_ms // frame_ms)
        self.silence_end_frames = max(2, silence_end_ms // frame_ms)
        self.min_speech_frames = max(2, min_speech_ms // frame_ms)
        self.stream_interval_samples = int(sr * stream_interval_ms / 1000)
        self.max_samples = int(sr * max_utterance_s)

        self.noise_rms = 0.0035
        self.ring = []
        self.speech_buf = []
        self.in_speech = False
        self.speech_frames = 0
        self.silence_frames = 0
        self.last_stream_samples = 0

    def reset(self):
        self.ring.clear()
        self.speech_buf.clear()
        self.in_speech = False
        self.speech_frames = 0
        self.silence_frames = 0
        self.last_stream_samples = 0

    def push_frame(self, frame, np_mod):
        rms = float(np_mod.sqrt(np_mod.mean(frame * frame) + 1e-12))
        thresh = max(0.0075, min(0.035, self.noise_rms * 2.8))
        is_voiced = rms >= thresh

        if not self.in_speech:
            if not is_voiced:
                self.noise_rms = 0.96 * self.noise_rms + 0.04 * rms
                self.ring.append(frame)
                if len(self.ring) > self.preroll_frames:
                    self.ring.pop(0)
                return None
            self.in_speech = True
            self.speech_buf = list(self.ring) + [frame]
            self.ring.clear()
            self.speech_frames = 1
            self.silence_frames = 0
            self.last_stream_samples = 0
            return None

        self.speech_buf.append(frame)
        total_samples = len(self.speech_buf) * self.frame_len
        if is_voiced:
            self.speech_frames += 1
            self.silence_frames = 0
        else:
            self.silence_frames += 1

        if self.silence_frames >= self.silence_end_frames or total_samples >= self.max_samples:
            valid = self.speech_frames >= self.min_speech_frames
            arr = np_mod.concatenate(self.speech_buf).astype(np_mod.float32) if valid else None
            self.reset()
            return (arr, True) if arr is not None else None

        if self.speech_frames >= self.min_speech_frames and (total_samples - self.last_stream_samples) >= self.stream_interval_samples:
            self.last_stream_samples = total_samples
            arr = np_mod.concatenate(self.speech_buf).astype(np_mod.float32)
            return arr, False

        return None


class Overlay:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AccentBridge Overlay (CTranslate2 + Breath VAD)")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-transparentcolor", TRANSPARENT)
        except tk.TclError:
            self.root.attributes("-alpha", 0.88)
        self.root.config(bg=TRANSPARENT)

        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.w, self.h = int(sw * 0.82), int(sh * 0.32)
        self.root.geometry(f"{self.w}x{self.h}+{(sw - self.w) // 2}+{sh - self.h - int(sh * 0.06)}")
        self.canvas = tk.Canvas(self.root, bg=TRANSPARENT, highlightthickness=0, width=self.w, height=self.h)
        self.canvas.pack(fill="both", expand=True)

        self.size_key, self.lines, self.style, self.highlight = "M", 2, "box", True
        self.align = "center"
        self.show, self.idle_sec = "tr", 3
        self.translate_to = "sync"
        self.extra_langs = []
        self.show_pills = True
        self.direct_mic = False
        self.active_base = BASE_URLS[0]

        self.font = tkfont.Font(family="Segoe UI Semibold", size=SIZES[self.size_key])
        self.small = tkfont.Font(family="Segoe UI", size=int(SIZES[self.size_key] * 0.62))
        self.pill_font = tkfont.Font(family="Segoe UI Semibold", size=11)

        self.state, self.connected = {}, False
        self.sig, self.last_activity = None, 0.0
        self.cleared_upto, self.cleared_interim = 0, ""
        self._dragging = False

        self.canvas.bind("<ButtonPress-1>", self._drag_start)
        self.canvas.bind("<B1-Motion>", self._drag_move)
        self.canvas.bind("<Button-3>", self._menu)

        threading.Thread(target=self._poll_unified_state, daemon=True).start()
        self._render()
        self.root.after(60, self._tick)

    def _is_muted(self) -> bool:
        return bool((self.state.get("ui") or {}).get("muted", False))

    def _effective_target_lang(self) -> str:
        if self.translate_to and self.translate_to != "sync":
            return self.translate_to.split("-")[0].lower()
        ui = self.state.get("ui") or {}
        return (ui.get("translate_to") or "none").split("-")[0].lower()

    def _set_translate_to(self, lang_code: str):
        self.translate_to = lang_code
        if lang_code not in ("sync", "none") and self.show == "orig":
            self.show = "tr"

        def _send():
            try:
                body = urllib.parse.urlencode({"translate_to": lang_code}).encode("utf-8")
                req = urllib.request.Request(
                    f"{self.active_base}/api/overlay_settings",
                    data=body,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    method="POST",
                )
                OPENER.open(req, timeout=2.0).close()
            except Exception as e:
                print(f"⚠️ ตั้งภาษาแปลไม่สำเร็จ: {e}")

        threading.Thread(target=_send, daemon=True).start()
        self._render()

    def _fetch_server_languages_once(self, base: str):
        if self.extra_langs:
            return
        try:
            with OPENER.open(f"{base}/api/languages", timeout=1.5) as r:
                d = json.loads(r.read().decode("utf-8"))
            known = {k for k, _ in TRANSLATE_TARGETS}
            extras = []
            for k, lbl in d.items():
                kc = k.split("-")[0].lower()
                if kc not in known and kc != "none":
                    known.add(kc)
                    extras.append((kc, f"🌐 {lbl} ({kc.upper()})"))
            self.extra_langs = extras
        except Exception:
            pass

    def _toggle_mute(self, _event=None):
        def _send():
            try:
                body = urllib.parse.urlencode({"muted": "toggle"}).encode("utf-8")
                req = urllib.request.Request(
                    f"{self.active_base}/api/mic_mute",
                    data=body,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    method="POST",
                )
                with OPENER.open(req, timeout=1.5) as r:
                    res = json.loads(r.read().decode("utf-8"))
                ui = self.state.setdefault("ui", {})
                ui["muted"] = bool(res.get("muted"))
            except Exception:
                pass

        threading.Thread(target=_send, daemon=True).start()
        return "break"

    def _popup_translate_menu(self, event):
        m = self._build_translate_submenu(tk.Menu(self.root, tearoff=0), standalone=True)
        m.tk_popup(event.x_root, event.y_root)
        return "break"

    def _poll_unified_state(self):
        last_err, idx, cur_ver = None, 0, -1
        while True:
            base = BASE_URLS[idx % len(BASE_URLS)]
            url = f"{base}/api/captions_state?since_version={cur_ver}&wait_ms=700&client=desktop_overlay"
            try:
                with OPENER.open(url, timeout=2.5) as r:
                    data = json.loads(r.read().decode("utf-8"))
                self.state = data
                self.active_base = base
                cur_ver = int(data.get("version", -1))
                if not self.connected:
                    print(f"✅ เชื่อมต่อ app.py สำเร็จที่ {base} ({data.get('detector', 'ready')})")
                    threading.Thread(target=self._fetch_server_languages_once, args=(base,), daemon=True).start()
                self.connected, last_err = True, None
            except Exception as e:
                self.connected = False
                cur_ver = -1
                idx += 1
                msg = f"{type(e).__name__}: {e}"
                if msg != last_err:
                    print(f"⚠️ ต่อ {base} ไม่ได้ ({msg}) — เช็กว่า python app.py ยังรันอยู่")
                    last_err = msg
                time.sleep(0.6)

    def _toggle_direct_mic(self, _event=None):
        if self.direct_mic:
            self.direct_mic = False
            self._render()
            return "break"
        try:
            import sounddevice  # noqa: F401
        except Exception:
            print("⚠️ โหมดไมค์ตรง VAD ต้องติดตั้ง sounddevice (`pip install sounddevice`) หรือกดปุ่ม '⚡ เริ่มฟัง VAD' บนหน้าเว็บได้เลยครับ")
            return "break"
        self.direct_mic = True
        threading.Thread(target=self._direct_mic_vad_worker, daemon=True).start()
        self._render()
        return "break"

    def _direct_mic_vad_worker(self):
        """
        ไมค์ตรงระดับ 30ms Frame-Level Breath VAD -> ยิงเข้า `/api/transcribe_vad` (CTranslate2):
        พูดสั้นตัดไว พูดยาวไม่ขาดครึ่ง พร้อมพรีวิวคำแปลสดระหว่างพูด!
        """
        import numpy as np
        import sounddevice as sd

        sr = 16000
        vad = BreathVADSegmenter(sr=sr)
        preview_busy = {"flag": False}

        def _post_vad_audio(audio_arr, is_fin: bool):
            try:
                pcm16 = np.clip(audio_arr * 32767.0, -32768, 32767).astype(np.int16)
                wav_io = io.BytesIO()
                with wave.open(wav_io, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(pcm16.tobytes())

                ui = self.state.get("ui") or {}
                fields = {
                    "is_final": "1" if is_fin else "0",
                    "slide_title": ui.get("slide_title") or self.state.get("slide", ""),
                    "source_lang": ui.get("source_lang") or "auto",
                    "translate_to": self._effective_target_lang(),
                    "speaker_gate": "1" if ui.get("speaker_gate") else "0",
                    "trigger_word": ui.get("trigger_word") or "next",
                }
                body, ctype = _encode_multipart_wav(wav_io.getvalue(), fields)
                req = urllib.request.Request(
                    f"{self.active_base}/api/transcribe_vad",
                    data=body,
                    headers={"Content-Type": ctype},
                    method="POST",
                )
                with OPENER.open(req, timeout=6.0) as _:
                    pass
            except Exception:
                pass
            finally:
                if not is_fin:
                    preview_busy["flag"] = False

        print("⚡ เปิดไมค์ตรง CTranslate2 + Breath VAD (ตัดตามจังหวะหยุดหายใจ 300ms) แล้ว!")
        try:
            with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=vad.frame_len) as stream:
                while self.direct_mic:
                    chunk, _ = stream.read(vad.frame_len)
                    if self._is_muted():
                        vad.reset()
                        continue
                    frame = chunk[:, 0].copy()
                    ev = vad.push_frame(frame, np)
                    if ev is not None:
                        arr, is_final = ev
                        if is_final:
                            threading.Thread(target=_post_vad_audio, args=(arr, True), daemon=True).start()
                        elif not preview_busy["flag"]:
                            preview_busy["flag"] = True
                            threading.Thread(target=_post_vad_audio, args=(arr, False), daemon=True).start()
        except Exception as e:
            print(f"⚠️ Direct VAD Mic หยุดทำงาน: {e}")
            self.direct_mic = False

    def _visible(self):
        segs = [s for s in self.state.get("segs", []) if s.get("id", 0) > self.cleared_upto]
        interim = self.state.get("interim", "")
        interim_tr = self.state.get("overlay_interim_tr") or self.state.get("interim_tr") or ""
        if interim == self.cleared_interim:
            interim, interim_tr = "", ""
        return segs[-1:], interim, interim_tr

    def _tick(self):
        segs = self.state.get("segs", [])
        sig = (
            self.state.get("version", 0),
            segs[-1]["id"] if segs else 0,
            self.state.get("interim", ""),
            self.state.get("overlay_interim_tr", ""),
            self.state.get("interim_tr", ""),
            self._is_muted(),
            self.translate_to,
            self.connected,
            self.direct_mic,
        )
        if sig != self.sig:
            self.sig = sig
            self.last_activity = time.time()
            self._render()
        elif self.idle_sec and self.last_activity and time.time() - self.last_activity > self.idle_sec:
            if segs or self.state.get("interim"):
                self.cleared_upto = segs[-1]["id"] if segs else self.cleared_upto
                self.cleared_interim = self.state.get("interim", "")
                self._render()
            self.last_activity = 0
        self.root.after(60, self._tick)

    def _tidy(self, tok, first):
        m = re.match(r"^([^A-Za-z']*)([A-Za-z']+)([^A-Za-z']*)$", tok)
        if not m:
            return tok
        pre, word, post = m.group(1), m.group(2), m.group(3)
        lw = word.lower()
        if lw in {a.lower() for a in self.state.get("acronyms", [])} or (word.isupper() and len(word) >= 2):
            return f"{pre}{lw.upper()}{post}"
        if lw == "i" or lw.startswith("i'"):
            return f"{pre}I{lw[1:]}{post}"
        core = lw[:1].upper() + lw[1:] if first else lw
        return f"{pre}{core}{post}"

    def _tokenize_to_items(self, text: str, kind: str = "final", fixed_tokens: set | None = None):
        items = []
        fixed_tokens = fixed_tokens or set()
        first = True
        for t, sp in tokenize(text):
            if sp:
                items.append((" ", True, kind))
            else:
                wkey = _clean_word_key(t)
                item_kind = "fixed" if (kind == "final" and wkey and wkey in fixed_tokens) else kind
                items.append((self._tidy(t, first), False, item_kind))
                first = False
        return items

    def _wrap(self, items, font, n, max_chars=MAX_CHARS):
        max_w = self.w - 60
        lines, cur = [], []
        txt = lambda arr: "".join(t for t, _, _ in arr).strip()
        for it in items:
            if it[1] and not cur:
                continue
            trial = cur + [it]
            s = txt(trial)
            if not it[1] and cur and (len(s) > max_chars or font.measure(s) > max_w):
                while cur and cur[-1][1]:
                    cur.pop()
                lines.append(cur)
                cur = [it]
            else:
                cur = trial
        while cur and cur[-1][1]:
            cur.pop()
        if cur:
            lines.append(cur)
        return lines[-n:] if n else lines

    def _tail(self, items, font, max_chars):
        out = []
        for it in reversed(items):
            s = "".join(t for t, _, _ in [it] + out).strip()
            if out and (len(s) > max_chars or font.measure(s) > self.w - 60):
                break
            out.insert(0, it)
        while out and out[0][1]:
            out.pop(0)
        return [out] if out else []

    def _draw_lines(self, lines, font, y_bottom, color_override=None):
        if not lines:
            return y_bottom
        c = self.canvas
        lh = font.metrics("linespace")
        pad_x, pad_y, gap = 14, 4, 6
        total_h = len(lines) * (lh + pad_y * 2) + max(0, len(lines) - 1) * gap
        y = y_bottom - total_h
        top = y
        win_x = self.root.winfo_x()
        screen_w = self.root.winfo_screenwidth()
        for line in lines:
            txt = "".join(t for t, _, _ in line)
            lw = font.measure(txt)
            if self.align == "left":
                x0 = 24
            else:
                x0 = (self.w - lw) // 2
            if win_x + x0 - pad_x < 8:
                x0 = max(pad_x + 4, 8 - win_x + pad_x)
            elif win_x + x0 + lw + pad_x > screen_w - 8:
                x0 = min(self.w - lw - pad_x - 4, screen_w - 8 - win_x - lw - pad_x)

            if self.style == "box":
                c.create_rectangle(x0 - pad_x, y, x0 + lw + pad_x, y + lh + pad_y * 2, fill=BOX, outline=BOX)
            x = x0
            for t, _, kind in line:
                color = color_override or (INTERIM if kind == "interim" else (HIGHLIGHT if kind == "fixed" and self.highlight else TEXT))
                if self.style == "outline":
                    for dx, dy in ((-2, -2), (2, -2), (-2, 2), (2, 2), (0, 2), (0, -2), (2, 0), (-2, 0)):
                        c.create_text(x + dx, y + pad_y + dy, text=t, anchor="nw", font=font, fill="#000000")
                c.create_text(x, y + pad_y, text=t, anchor="nw", font=font, fill=color)
                x += font.measure(t)
            y += lh + pad_y * 2 + gap
        return top

    def _draw_control_pills(self, y_anchor):
        if not self.show_pills and not self._is_muted():
            return
        muted = self._is_muted()
        mute_lbl = "🔇 ไมค์ปิดอยู่ (คลิกพูดต่อ)" if muted else "🎙️ ปิดไมค์ชั่วคราว"

        eff_tgt = self._effective_target_lang()
        if self.translate_to == "sync":
            tr_lbl = f"🌐 แปล: ตามเว็บ ({eff_tgt.upper() if eff_tgt != 'none' else 'ไม่แปล'}) ▾"
        elif eff_tgt == "none":
            tr_lbl = "🌐 แปล: ไม่แปล ▾"
        else:
            tr_lbl = f"🌐 แปลเป็น: {eff_tgt.upper()} ▾"

        vad_lbl = "⏹ VAD สด: เปิดอยู่" if self.direct_mic else "⚡ ไมค์ตรง VAD"

        mw = self.pill_font.measure(mute_lbl)
        tw = self.pill_font.measure(tr_lbl)
        vw = self.pill_font.measure(vad_lbl)
        bh = self.pill_font.metrics("linespace")
        gap_between = 8
        total_w = (mw + 18) + gap_between + (tw + 18) + gap_between + (vw + 18)

        win_x = self.root.winfo_x()
        screen_w = self.root.winfo_screenwidth()
        x_start = (self.w - total_w) // 2
        if win_x + x_start < 10:
            x_start = max(6, 10 - win_x)
        elif win_x + x_start + total_w > screen_w - 10:
            x_start = min(self.w - total_w - 6, screen_w - 10 - win_x - total_w)

        y0 = max(6, y_anchor - bh - 14)

        # 1) Mute Pill
        m_bg = MUTE_BG if muted else UNMUTE_BG
        m_bd = MUTE_BORDER if muted else UNMUTE_BORDER
        m_fg = "#ffffff" if muted else "#a5d6ff"
        r1 = self.canvas.create_rectangle(x_start, y0 - 3, x_start + mw + 18, y0 + bh + 3, fill=m_bg, outline=m_bd, width=1, tags=("pill_btn",))
        t1 = self.canvas.create_text(x_start + 9, y0, text=mute_lbl, anchor="nw", font=self.pill_font, fill=m_fg, tags=("pill_btn",))
        self.canvas.tag_bind(r1, "<ButtonPress-1>", self._toggle_mute)
        self.canvas.tag_bind(t1, "<ButtonPress-1>", self._toggle_mute)

        # 2) Translate-To Pill
        x2 = x_start + mw + 18 + gap_between
        r2 = self.canvas.create_rectangle(x2, y0 - 3, x2 + tw + 18, y0 + bh + 3, fill=LANG_BG, outline=LANG_BORDER, width=1, tags=("pill_btn",))
        t2 = self.canvas.create_text(x2 + 9, y0, text=tr_lbl, anchor="nw", font=self.pill_font, fill="#ffd84d" if eff_tgt != "none" else "#c9d1d9", tags=("pill_btn",))
        self.canvas.tag_bind(r2, "<ButtonPress-1>", self._popup_translate_menu)
        self.canvas.tag_bind(t2, "<ButtonPress-1>", self._popup_translate_menu)

        # 3) Direct Breath-VAD Mic Pill
        x3 = x2 + tw + 18 + gap_between
        v_bg = VAD_BG if self.direct_mic else "#1b222c"
        v_bd = VAD_BORDER if self.direct_mic else "#484f58"
        v_fg = "#7ee787" if self.direct_mic else "#c9d1d9"
        r3 = self.canvas.create_rectangle(x3, y0 - 3, x3 + vw + 18, y0 + bh + 3, fill=v_bg, outline=v_bd, width=1, tags=("pill_btn",))
        t3 = self.canvas.create_text(x3 + 9, y0, text=vad_lbl, anchor="nw", font=self.pill_font, fill=v_fg, tags=("pill_btn",))
        self.canvas.tag_bind(r3, "<ButtonPress-1>", self._toggle_direct_mic)
        self.canvas.tag_bind(t3, "<ButtonPress-1>", self._toggle_direct_mic)

    def _render(self):
        self.canvas.delete("all")
        if not self.connected and not self.state:
            self._draw_lines([[("รอเชื่อมต่อ CTranslate2 Server ที่ localhost:7860 …", False, "final")]], self.font, self.h - 8)
            return

        if self._is_muted():
            self._draw_control_pills(self.h - 8)
            return

        segs, interim, interim_tr = self._visible()
        eff_tgt = self._effective_target_lang()
        tr_enabled = eff_tgt not in ("none", "")

        if not segs and not interim:
            if not self.state.get("segs") and not self.state.get("interim"):
                tgt_disp = eff_tgt.upper() if tr_enabled else "ต้นฉบับ"
                msg = f"✓ CTranslate2 + VAD พร้อมแล้ว · แปลเป็น: {tgt_disp} (คลิก ⚡ ไมค์ตรง VAD หรือเริ่มจากหน้าเว็บได้เลย)"
                top = self._draw_lines([[(msg, False, "final")]], self.small, self.h - 10)
                self._draw_control_pills(top)
            else:
                self._draw_control_pills(self.h - 8)
            return

        if interim:
            if not tr_enabled or self.show == "orig":
                items = self._tokenize_to_items(interim, kind="interim")
                top = self._draw_lines(self._wrap(items, self.font, self.lines), self.font, self.h - 8)
            elif self.show == "tr":
                if interim_tr:
                    items = self._tokenize_to_items(interim_tr, kind="interim")
                    top = self._draw_lines(self._wrap(items, self.font, self.lines), self.font, self.h - 8)
                else:
                    top = self.h - 8
            else:
                if interim_tr:
                    t1 = self._draw_lines(self._wrap(self._tokenize_to_items(interim_tr, kind="interim"), self.font, self.lines), self.font, self.h - 8)
                    top = self._draw_lines(self._tail(self._tokenize_to_items(interim, kind="interim"), self.small, 78), self.small, t1 - 6, color_override=ORIG)
                else:
                    top = self._draw_lines(self._tail(self._tokenize_to_items(interim, kind="interim"), self.small, 78), self.small, self.h - 8, color_override=ORIG)
            self._draw_control_pills(top)
            return

        sg = segs[-1]
        orig_txt = sg.get("text", "")
        tr_txt = (sg.get("overlay_tr") if self.translate_to != "sync" else sg.get("tr")) or sg.get("tr") or ""
        seg_lang = (sg.get("lang") or "auto").lower()
        same_lang = tr_enabled and (seg_lang == eff_tgt)

        fixed_set = {_clean_word_key(sub_w) for fw in sg.get("fixed", []) for sub_w in str(fw).lower().split()}

        if not tr_enabled or self.show == "orig" or same_lang:
            items = self._tokenize_to_items(orig_txt, kind="final", fixed_tokens=fixed_set)
            top = self._draw_lines(self._wrap(items, self.font, self.lines), self.font, self.h - 8)
        elif self.show == "tr":
            if tr_txt:
                items = self._tokenize_to_items(tr_txt, kind="final")
                top = self._draw_lines(self._wrap(items, self.font, self.lines), self.font, self.h - 8)
            else:
                top = self.h - 8
        else:
            if tr_txt:
                t1 = self._draw_lines(self._wrap(self._tokenize_to_items(tr_txt, kind="final"), self.font, self.lines), self.font, self.h - 8)
                top = self._draw_lines(self._tail(self._tokenize_to_items(orig_txt, kind="final", fixed_tokens=fixed_set), self.small, 78), self.small, t1 - 6, color_override=ORIG)
            else:
                items = self._tokenize_to_items(orig_txt, kind="final", fixed_tokens=fixed_set)
                top = self._draw_lines(self._wrap(items, self.font, self.lines), self.font, self.h - 8)
        self._draw_control_pills(top)

    def _drag_start(self, e):
        items = self.canvas.find_withtag("current")
        if items and "pill_btn" in self.canvas.gettags(items[0]):
            self._dragging = False
            return
        self._dragging = True
        self._dx, self._dy = e.x, e.y

    def _drag_move(self, e):
        if not self._dragging:
            return
        self.root.geometry(f"+{self.root.winfo_x() + e.x - self._dx}+{self.root.winfo_y() + e.y - self._dy}")

    def _set(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self.font.configure(size=SIZES[self.size_key])
        self.small.configure(size=int(SIZES[self.size_key] * 0.62))
        self._render()

    def _clear_now(self):
        try:
            req = urllib.request.Request(
                f"{self.active_base}/api/caption_event",
                data=json.dumps({"type": "clear"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            OPENER.open(req, timeout=1.5).close()
        except Exception:
            pass

    def _build_translate_submenu(self, tr_menu: tk.Menu, standalone: bool = False) -> tk.Menu:
        all_targets = list(TRANSLATE_TARGETS) + self.extra_langs
        for code, lbl in all_targets:
            checked = "✓ " if self.translate_to == code else "   "
            tr_menu.add_command(label=checked + lbl, command=lambda c=code: self._set_translate_to(c))
            if code == "none":
                tr_menu.add_separator()
        return tr_menu

    def _menu(self, e):
        m = tk.Menu(self.root, tearoff=0)
        muted = self._is_muted()
        m.add_command(
            label=("🎙️ เปิดไมค์ต่อ (Unmute)" if muted else "🔇 ปิดไมค์ชั่วคราว (กันซับรวน)"),
            command=self._toggle_mute,
        )
        m.add_command(
            label=("⏹ ปิดไมค์ตรง VAD" if self.direct_mic else "⚡ เปิดไมค์ตรง CTranslate2 + VAD (ตัดตามจังหวะพูด)"),
            command=self._toggle_direct_mic,
        )

        tr_sub = self._build_translate_submenu(tk.Menu(m, tearoff=0))
        m.add_cascade(label="🌐 แปลซับลอยเป็นภาษา (Translate to)", menu=tr_sub)

        show_sub = tk.Menu(m, tearoff=0)
        for k, lbl in (("tr", "เฉพาะคำแปล (ไม่โชว์ต้นฉบับก่อน)"), ("both", "สองภาษา (ต้นฉบับ + คำแปล)"), ("orig", "เฉพาะต้นฉบับ")):
            show_sub.add_command(label=("✓ " if self.show == k else "   ") + lbl, command=lambda k=k: self._set(show=k))
        m.add_cascade(label="รูปแบบภาษาที่แสดง", menu=show_sub)

        m.add_separator()
        idle = tk.Menu(m, tearoff=0)
        for sec in (1, 2, 3, 4, 6, 0):
            lbl = f"{sec} วินาที" if sec else "ไม่หาย"
            idle.add_command(label=("✓ " if self.idle_sec == sec else "   ") + lbl, command=lambda s=sec: self._set(idle_sec=s))
        m.add_cascade(label="ซับหายเมื่อหยุดพูด", menu=idle)

        sm = tk.Menu(m, tearoff=0)
        for k in SIZES:
            sm.add_command(label=("✓ " if self.size_key == k else "   ") + k, command=lambda k=k: self._set(size_key=k))
        m.add_cascade(label="ขนาดตัวอักษร", menu=sm)

        lm = tk.Menu(m, tearoff=0)
        for n in (1, 2, 3):
            lm.add_command(label=("✓ " if self.lines == n else "   ") + f"{n} บรรทัด", command=lambda n=n: self._set(lines=n))
        m.add_cascade(label="จำนวนบรรทัด", menu=lm)

        al = tk.Menu(m, tearoff=0)
        al.add_command(label=("✓ " if self.align == "center" else "   ") + "กึ่งกลาง", command=lambda: self._set(align="center"))
        al.add_command(label=("✓ " if self.align == "left" else "   ") + "ชิดซ้าย", command=lambda: self._set(align="left"))
        m.add_cascade(label="การจัดวางข้อความ", menu=al)

        m.add_command(label=("✓ " if self.style == "box" else "   ") + "รูปแบบ: กล่องดำ", command=lambda: self._set(style="box"))
        m.add_command(label=("✓ " if self.style == "outline" else "   ") + "รูปแบบ: ขอบดำ (ไม่มีกล่อง)", command=lambda: self._set(style="outline"))
        m.add_command(label=("✓ " if self.highlight else "   ") + "ไฮไลต์คำสำเนียงเอเชียที่ AI แก้ (สีเหลือง)", command=lambda: self._set(highlight=not self.highlight))
        m.add_command(label=("✓ " if self.show_pills else "   ") + "แสดงแถบปุ่มลัดด้านบนซับลอย", command=lambda: self._set(show_pills=not self.show_pills))
        m.add_separator()
        m.add_command(label="🧹 ล้างข้อความซับทั้งหมด", command=self._clear_now)
        m.add_separator()
        m.add_command(label="ปิดซับลอย", command=self.root.destroy)
        m.tk_popup(e.x_root, e.y_root)


if __name__ == "__main__":
    print("🪟 เปิดซับลอยแล้ว (CTranslate2 + Breath VAD): คลิกปุ่ม [⚡ ไมค์ตรง VAD] หรือ [🌐 แปล: ...] ได้เลย!")
    Overlay().root.mainloop()
