# \# AccentBridge: CTranslate2 Breath-VAD, Phoneme-Aligned \& Slide-Synced Live Subtitle AI

# \*\*Track:\*\* AI + Languages — \*Veritas AI International High School Hackathon 2026\*

# 

# \## 1. Problem \& Overview

# Non-native English and multilingual presenters—especially East, Southeast, and South Asian speakers—routinely suffer from severe live subtitle errors and translation lag on technical domain terms due to systematic L2 phonological shifts and prosodic chunking mismatches:

# \- \*\*Onset \& coda consonant-cluster simplification \& unreleased final stops\*\* (e.g., \*slide\* $\\rightarrow$ \*side/sly\*, \*script\* $\\rightarrow$ \*skrip\*, \*file\* $\\rightarrow$ \*fine/fire\*, \*modules\* $\\rightarrow$ \*modu\*)

# \- \*\*Liquid \& labial conflation\*\* (\*/r/\* $\\leftrightarrow$ \*/l/\*, \*/v/\* $\\leftrightarrow$ \*/w, b/\*, \*/f/\* $\\leftrightarrow$ \*/p/\*, e.g., \*right\* $\\leftrightarrow$ \*light\*, \*files\* $\\leftrightarrow$ \*piles\*)

# \- \*\*Dental fricative stopping \& fronting\*\* (\*/θ, ð/\* $\\rightarrow$ \*/t, d, f/\*, e.g., \*three\* $\\rightarrow$ \*tree/free\*)

# \- \*\*Epenthetic coda vowels \& non-rhotic vowel coloring\*\* (e.g., \*test\* $\\rightarrow$ \*testo\*, \*word error rate\* $\\rightarrow$ \*werd era late\*, \*open source\* $\\rightarrow$ \*open sauce\*)

# \- \*\*Rigid fixed-window chunking \& code-switching failures:\*\* Fixed 2–3s audio timers slice words in half mid-syllable, while English-only decoders loop or hallucinate when presenters switch between Asian languages (e.g., Thai, Mandarin, Japanese, Korean) and English slide terminology.

# 

# \*\*AccentBridge\*\* solves these challenges end-to-end by combining \*\*CTranslate2 (`faster-whisper`) + Breath-Synced Adaptive VAD\*\*, \*\*Asian-accent LoRA Whisper adaptation\*\*, \*\*CMU ARPAbet \& Pan-Asian Phonetic Skeleton Alignment\*\*, \*\*Thai–English Code-Switching Rescue\*\*, \*\*Speaker Verification Gating\*\*, and \*\*Zero-Flash Streaming Translation\*\* across both a \*\*Web Eye-Line Studio\*\* and a \*\*Click-Through Windows Floating Overlay (`overlay.py`)\*\*.

# 

# \---

# 

# \## 2. Key Technical Contributions

# 1\. \*\*Ultra-Low-Latency CTranslate2 Engine + Breath-Synced Adaptive VAD (`app.py` + `overlay.py`):\*\*

# &#x20;  - Accelerates inference by \*\*3–4× (\~45–90 ms on CUDA FP16 / INT8 on CPU)\*\* while cutting VRAM usage in half via \*\*CTranslate2 (`faster-whisper`)\*\* and \*\*Silero Neural VAD\*\*, automatically converting fine-tuned `./saved\_model` checkpoints to `./saved\_model\_ct2`.

# &#x20;  - Replaces blind fixed-interval slicing with a \*\*30 ms Frame-Level Breath-Synced VAD\*\* featuring a \*\*240 ms Pre-Roll Ring Buffer\*\* (preserving soft Asian initial consonants), \*\*650 ms mid-sentence streaming previews\*\*, and \*\*300 ms breath-pause endpointing\*\* so sentences never cut in half and old utterances never bleed into new ones.

# 2\. \*\*Pan-Asian Articulatory Phoneme Alignment \& Code-Switching Rescue (`accent\_engine.py` + `confusion\_priors.json`):\*\*

# &#x20;  - Aligns decoded tokens against active slide vocabulary in \*\*CMU ARPAbet\*\* space and \*\*Pan-Asian Phonetic Skeletons\*\* (modeling L/R, V/W/B, F/P, TH-stopping/fronting, coda-cluster deletion, non-rhotic vowels, and Asian spelled-out acronyms like `ay pee eye` $\\rightarrow$ `API`, `gee pee you` $\\rightarrow$ `GPU`).

# &#x20;  - Supports \*\*Multilingual Auto-Detection (99 languages)\*\* and \*\*Thai–English Code-Switching Rescue\*\*, converting spoken Thai technical loanwords (e.g., \*เวิร์ดเออเรอร์เรต\*, \*จีพียู\*) into exact slide terminology (\*word error rate\*, \*GPU\*) only when grounded on the current slide.

# 3\. \*\*10-Second Personal Voice LoRA Calibration \& Cross-Lingual Speaker Gate (`speaker\_id.py`):\*\*

# &#x20;  - Interactive \*\*10-Second Personal Voice Calibration\*\* (`/api/calibrate\_voice`) performs few-shot LoRA micro-adaptation (`q\_proj`, `v\_proj`) and mines speaker-specific confusion pairs live before a presentation.

# &#x20;  - Optional \*\*Enrolled Speaker Verification Gate\*\* rejects off-mic crosstalk and audience noise with cross-lingual tolerance so bilingual presenters are never falsely blocked.

# 4\. \*\*Unified Zero-Flash Streaming Translation, Broadcast View \& Desktop Overlay (`overlay.py` + `static/captions.html`):\*\*

# &#x20;  - Persistent \*\*Keep-Alive HTTPS + 1,024-entry LRU Cached Translator\*\* streams live translated captions (`interim\_tr`) in \*\*\~70–110 ms\*\* without flashing raw source text first in translation-only mode.

# &#x20;  - Synchronizes a single unified detector state (`SSE` + `Condition` long-polling `<15 ms`) across the \*\*Presenter Eye-Line Studio (`/`)\*\*, \*\*Broadcast/OBS Judge View (`/captions`)\*\*, and a \*\*Transparent Click-Through Windows Desktop Overlay (`overlay.py`)\*\* with independent target-language selection and instant hardware-level \*\*Mute Gate (`Alt+M`)\*\*.

# 

# \---

# 

# \## 3. Project Structure

# ```text

# AccentBridge/

# ├── app.py                  # Unified FastAPI server: CTranslate2 + Breath-VAD, SSE sync, Keep-Alive translator \& Mute gate

# ├── overlay.py              # Transparent click-through Windows desktop subtitle overlay with Direct VAD Mic \& Translate-To menu

# ├── accent\_engine.py        # Lazy-VRAM Whisper LoRA engine, ARPAbet DP alignment \& Thai-English code-switching rescue

# ├── slide\_parser.py         # Recursive PPTX/PDF/DOCX parser, mixed-script acronym/phrase extractor \& auto-advance

# ├── speaker\_id.py           # Enrolled speaker verification gate (filters background/audience voices)

# ├── translator.py           # Multilingual translation language registry \& fallback translation module

# ├── confusion\_priors.json   # Pan-Asian L2 phonetic minimal pairs \& Thai-English technical transliteration stems

# ├── train\_and\_eval.py       # Pure Asian corpus curation, LoRA training, Dev grid calibration \& paired bootstrap CI

# ├── make\_demo\_slides.py     # Generates a sample 3-slide PowerPoint deck (demo\_slides.pptx) with Speaker Notes

# ├── verify\_project.py       # Automated sanity test suite verifying phonology, parser \& file layout

# ├── static/

# │   ├── index.html          # Eye-line teleprompter, 30ms Breath-VAD WebAudio studio, Mute (Alt+M) \& Overlay launcher

# │   └── captions.html       # Broadcast/OBS judge caption window with SSE live translation \& Chroma/Transparent modes

# ├── saved\_model/

# │   ├── confusion\_priors.json   # Speaker-calibrated \& data-mined Asian phonetic confusion priors

# │   └── calibrated\_params.json  # Held-out Dev calibrated rescoring \& acoustic confidence gate parameters

# └── requirements.txt        # Reproducible Python dependencies (CTranslate2, faster-whisper, PyTorch, FastAPI)



