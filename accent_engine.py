import json
import os
import re
from difflib import SequenceMatcher
import numpy as np
from slide_parser import (
    extract_priority_keywords,
    extract_context_bigrams,
    extract_acronyms,
)

try:
    import pronouncing
    HAS_CMUDICT = True
except ImportError:
    HAS_CMUDICT = False

STOPWORDS = {
    "a", "an", "the", "to", "in", "on", "at", "of", "for", "and", "or",
    "is", "am", "are", "was", "were", "be", "it", "he", "she", "we", "they",
    "i", "you", "my", "me", "do", "go", "so", "no", "up", "by", "as", "not",
    "this", "that", "with", "from", "have", "has", "will", "can", "our", "your",
}

HALLUCINATION_PHRASES = {
    "thank you", "thank you for watching", "thanks for watching",
    "you", "bye", "subtitles by", "amara org",
}

POINT_PREV = {
    "a", "the", "this", "that", "my", "our", "your", "good", "key", "main",
    "important", "first", "second", "third", "fourth", "last", "next", "final",
    "one", "another", "bullet", "data", "starting", "turning", "focal",
    "floating", "decimal", "strong", "valid", "fair", "whole", "crucial",
    "critical", "single", "every", "each", "at", "on", "power", "check",
    "end", "pin", "view", "access", "entry", "pain", "weak",
}
POINT_NEXT = {"out", "of", "is", "was", "to", "that", "here", "where", "in"}

COMMON_NON_PROPER = STOPWORDS | {
    "today", "here", "check", "finally", "evaluated", "evaluate", "measures",
    "measure", "project", "overview", "file", "files", "architecture",
    "dataset", "benchmark", "intent", "preservation", "future", "work",
    "spontaneous", "speech", "stays", "untouched", "phoneme", "rescue",
    "activates", "only", "when", "close", "slide", "terms", "three", "core",
    "modules", "open", "source", "pipeline", "system", "word", "error",
    "rate", "target", "keyword", "recall", "asian", "accents", "unseen",
    "speakers", "normal", "subtitles", "while", "rescuing", "please", "look",
    "where", "present", "using", "test", "speak", "freely", "script",
    "confidence", "gate", "keeps", "model", "baseline", "verify", "right",
    "these", "those", "next", "summary", "introduction", "conclusion",
    "method", "methods", "results", "discussion", "point", "points",
    "hello", "welcome", "team", "name",
}

ARPABET_PAIR_COSTS = {
    frozenset({"R", "L"}): 0.08, frozenset({"R", "ER"}): 0.10,
    frozenset({"L", "ER"}): 0.14, frozenset({"L", "N"}): 0.14,
    frozenset({"TH", "T"}): 0.08, frozenset({"TH", "F"}): 0.12,
    frozenset({"TH", "S"}): 0.14, frozenset({"DH", "D"}): 0.08,
    frozenset({"DH", "Z"}): 0.14, frozenset({"V", "W"}): 0.08,
    frozenset({"V", "B"}): 0.12, frozenset({"F", "P"}): 0.08,
    frozenset({"V", "F"}): 0.14, frozenset({"T", "D"}): 0.10,
    frozenset({"P", "B"}): 0.10, frozenset({"K", "G"}): 0.12,
    frozenset({"S", "Z"}): 0.10, frozenset({"SH", "CH"}): 0.14,
    frozenset({"SH", "S"}): 0.14, frozenset({"CH", "TS"}): 0.14,
    frozenset({"JH", "CH"}): 0.16, frozenset({"N", "NG"}): 0.12,
    frozenset({"M", "N"}): 0.14, frozenset({"IY", "IH"}): 0.12,
    frozenset({"UW", "UH"}): 0.12, frozenset({"EY", "EH"}): 0.16,
    frozenset({"AE", "EH"}): 0.16, frozenset({"AA", "AO"}): 0.12,
    frozenset({"OW", "AO"}): 0.14, frozenset({"AH", "ER"}): 0.14,
    frozenset({"OY", "AY"}): 0.10, frozenset({"OY", "AO"}): 0.12,
}
ARPABET_VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
    "IH", "IY", "OW", "OY", "UH", "UW",
}
DROPPABLE_CODA = {
    "T", "D", "K", "G", "P", "B", "S", "Z", "L", "R", "F", "V", "N", "M", "NG",
}
CLUSTER_REDUCIBLE = {"L", "R", "S", "Z", "T", "D", "K", "P", "N", "M", "NG"}

DEFAULT_TH_STEMS = {
    "พอยต์": "point", "พ้อยท์": "point", "พ้อย": "point",
    "พาวเวอร์พอยต์": "PowerPoint", "แอคเซนต์บริดจ์": "AccentBridge",
    "เวอริทัส": "Veritas", "เอดแอค": "EdAcc", "ซิเลโร": "Silero",
    "เอพีไอ": "API", "จีพียู": "GPU", "ซีพียู": "CPU", "ดับเบิลยูอีอาร์": "WER",
    "โมเดล": "model", "โมดูล": "modules", "สไลด์": "slide", "ไฟล์": "file",
    "ซอร์ส": "source", "โอเพนซอร์ส": "open source", "ซิสเต็ม": "system",
    "ไปป์ไลน์": "pipeline", "ดาต้าเซ็ต": "dataset", "เบนช์มาร์ก": "benchmark",
    "เบสไลน์": "baseline", "รีคอล": "recall", "เออเรอร์": "error",
    "เวิร์ดเออเรอร์เรต": "word error rate", "คีย์เวิร์ด": "keyword",
    "สคริปต์": "script", "ซับไตเติล": "subtitles", "คอนฟิเดนซ์": "confidence",
    "เรียลไทม์": "real-time", "โปรเจกต์": "project",
}

TH_CONS = {
    "ก": "k", "ข": "k", "ฃ": "k", "ค": "k", "ฅ": "k", "ฆ": "k", "ง": "n",
    "จ": "c", "ฉ": "c", "ช": "c", "ฌ": "c", "ซ": "s", "ศ": "s", "ษ": "s",
    "ส": "s", "ญ": "y", "ย": "y", "ฎ": "d", "ด": "d", "ฏ": "t", "ฐ": "t",
    "ฑ": "t", "ฒ": "t", "ต": "t", "ถ": "t", "ท": "t", "ธ": "t", "ณ": "n",
    "น": "n", "บ": "b", "ป": "p", "ผ": "p", "พ": "p", "ภ": "p", "ฝ": "f",
    "ฟ": "f", "ม": "m", "ร": "l", "ล": "l", "ฬ": "l", "ว": "w", "ห": "h",
    "ฮ": "h", "อ": "a",
}
TH_VOW = {
    "ะ": "a", "ั": "a", "า": "a", "ำ": "am", "ิ": "i", "ี": "i", "ึ": "u",
    "ื": "u", "ุ": "u", "ู": "u", "เ": "e", "แ": "e", "โ": "o", "ใ": "ay",
    "ไ": "ay", "็": "", "ๆ": "",
}


def normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s']", " ", (text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _init_grp(w: str) -> str:
    if not w:
        return ""
    if w.startswith("th"):
        return "t"
    c = w[0]
    if c in "fpb":
        return "p"
    if c in "rl":
        return "l"
    if c in "vw":
        return "w"
    if c in "csz":
        return "s"
    return c


def latin_name_skeleton(s: str) -> str:
    w = re.sub(r"[^a-z0-9]", "", (s or "").lower())
    w = re.sub(r"cc([eiy])", r"ks\1", w)
    w = w.replace("tch", "C").replace("dge", "C").replace("dg", "C").replace("ch", "C")
    w = re.sub(r"c([eiy])", r"s\1", w).replace("ck", "k").replace("qu", "kw").replace("c", "k")
    w = w.replace("ph", "p").replace("th", "t").replace("sh", "s").replace("jh", "C").replace("gh", "")
    w = w.replace("ee", "i").replace("ea", "i").replace("oo", "u").replace("ou", "o").replace("ow", "o")
    w = w.replace("oy", "ay").replace("oi", "ay").replace("ai", "ay")
    w = re.sub(r"([bcdfghjklmnpqrstvwxz])y", r"\1i", w)
    w = re.sub(r"([aeiou])r(?![aeiouy])", r"\1", w)
    tmap = str.maketrans({"r": "l", "v": "w", "z": "s", "j": "c", "C": "c", "q": "k", "x": "ks"})
    w = re.sub(r"(.)\1+", r"\1", w.translate(tmap))
    if len(w) >= 4:
        w = re.sub(r"nt([bcdfghklmnpqstvwxz])", r"n\1", w)
        w = re.sub(r"(s|n|l)[tkp]$", r"\1", w)
        w = re.sub(r"[dtc]$", "t", re.sub(r"[gk]$", "k", re.sub(r"[bp]$", "p", w)))
    return w


def thai_to_latin_skeleton(th: str) -> str:
    s = re.sub(r"[\u0E48-\u0E4E]", "", re.sub(r"[ก-ฮ]์", "", th or ""))
    s = re.sub(r"([เแโใไ])([ก-ฮ][รลว]?)", r"\2\1", s)
    chars = [TH_CONS.get(c, TH_VOW.get(c, c.lower() if c.isalnum() else "")) for c in s]
    res = re.sub(r"(.)\1+", r"\1", "".join(chars))
    return re.sub(r"[dtc]$", "t", res) if len(res) >= 4 else res


def extract_proper_names(slide_title: str = "", slide_text: str = "", raw_keywords: str = "") -> list:
    names, seen = [], set()

    def _add(n: str):
        nc = n.strip(" ,.;:!?()[]{}\"'")
        nl = nc.lower()
        if len(nc) >= 3 and nl not in seen and nl not in COMMON_NON_PROPER:
            seen.add(nl)
            names.append(nc)

    kw_iter = (
        [str(x) for x in raw_keywords]
        if isinstance(raw_keywords, (list, tuple, set))
        else re.split(r"[,;\n]+", str(raw_keywords or ""))
    )
    for kw in kw_iter:
        ks = kw.strip().lstrip("@")
        if ks and (any(c.isupper() for c in ks) or kw.strip().startswith("@")):
            _add(ks)

    comb = f"{slide_title}\n{slide_text}".replace("[Speaker Notes]", " ")
    for m in re.finditer(r"(?<![A-Za-z])([A-Z][A-Za-z0-9]*(?:[A-Z0-9][A-Za-z0-9]*)+)(?![A-Za-z])", comb):
        _add(m.group(1))
    for m in re.finditer(r"(?<![A-Za-z])([A-Z][a-z]{2,})(?![A-Za-z])", comb):
        _add(m.group(1))
    return names[:35]


def rescue_collocations_and_point(text: str) -> tuple:
    if not text:
        return text, []
    toks, rescued, out, i = text.split(), [], [], 0
    bigram_fixes = {
        ("open", "sauce"): ("open", "source"),
        ("sauce", "code"): ("source", "code"),
        ("word", "era"): ("word", "error"),
        ("werd", "era"): ("word", "error"),
        ("era", "rate"): ("error", "rate"),
        ("era", "late"): ("error", "rate"),
        ("error", "late"): ("error", "rate"),
    }
    while i < len(toks):
        m = re.match(r"^([^A-Za-z]*)([A-Za-z]+)([^A-Za-z]*)$", toks[i])
        if not m:
            out.append(toks[i])
            i += 1
            continue
        pre, core, post = m.group(1), m.group(2), m.group(3)
        lw = core.lower()
        if i + 1 < len(toks):
            m2 = re.match(r"^([^A-Za-z]*)([A-Za-z]+)([^A-Za-z]*)$", toks[i + 1])
            if m2:
                lw2 = m2.group(2).lower()
                if lw == "power" and lw2 in ("fight", "fights", "fine", "find", "boy", "poy"):
                    out.append(f"{pre}PowerPoint{m2.group(3)}")
                    rescued.append((f"{core} {m2.group(2)}", "PowerPoint"))
                    i += 2
                    continue
                if (lw, lw2) in bigram_fixes:
                    r1, r2 = bigram_fixes[(lw, lw2)]
                    if core[0].isupper():
                        r1 = r1.capitalize()
                    out.append(f"{pre}{r1}{post}")
                    out.append(f"{m2.group(1)}{r2}{m2.group(3)}")
                    rescued.append((f"{core} {m2.group(2)}", f"{r1} {r2}"))
                    i += 2
                    continue
        if lw in ("fight", "fights"):
            pw = re.sub(r"[^a-z]", "", toks[i - 1].lower()) if i > 0 else ""
            nw = re.sub(r"[^a-z]", "", toks[i + 1].lower()) if i + 1 < len(toks) else ""
            if pw in POINT_PREV or nw in POINT_NEXT:
                rep = "points" if lw.endswith("s") else "point"
                if core[0].isupper():
                    rep = rep.capitalize()
                out.append(f"{pre}{rep}{post}")
                rescued.append((core, rep))
                i += 1
                continue
        out.append(toks[i])
        i += 1
    return " ".join(out), rescued


def rescue_proper_names_in_text(text: str, proper_names: list) -> tuple:
    if not text or not proper_names:
        return text, []
    entries = [(nm, latin_name_skeleton(nm)) for nm in proper_names if len(latin_name_skeleton(nm)) >= 3]
    if not entries:
        return text, []
    rescued = []

    if re.search(r"[\u0E00-\u0E7F]", text):
        prefs = ("โปรเจกต์", "ระบบ", "คุณ", "นาย", "นางสาว", "ดร.", "อาจารย์", "ทีม", "บริษัท", "งาน", "จาก", "ของ", "และ", "กับ", "บน", "ใน")
        new_p = []
        for pt in re.split(r"(\s+)", text):
            if not pt or pt.isspace() or not re.search(r"[\u0E00-\u0E7F]", pt):
                new_p.append(pt)
                continue
            pf_hit, body = "", pt
            for pf in sorted(prefs, key=len, reverse=True):
                if body.startswith(pf) and len(body) > len(pf) + 2:
                    pf_hit, body = pf, body[len(pf):]
                    break
            th_sk = thai_to_latin_skeleton(body)
            best_nm, best_sc = None, 0.0
            for nm, nm_sk in entries:
                if abs(len(th_sk) - len(nm_sk)) <= 4:
                    sc = SequenceMatcher(None, th_sk, nm_sk).ratio()
                    if sc > best_sc:
                        best_sc, best_nm = sc, nm
            if best_nm and best_sc >= 0.80:
                new_p.append(f"{pf_hit} {best_nm}".strip() if pf_hit else best_nm)
                rescued.append((body, best_nm))
            else:
                new_p.append(pt)
        text = re.sub(r"\s+", " ", "".join(new_p)).strip()

    toks, out_t, i = text.split(), [], 0
    while i < len(toks):
        best_nm, best_span, best_sc = None, 0, 0.0
        for span in (1, 2, 3):
            if i + span > len(toks):
                continue
            cores = [re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", t) for t in toks[i : i + span]]
            if not all(cores) or not all(re.fullmatch(r"[A-Za-z0-9'\-]+", c) for c in cores):
                continue
            bad_multi = span >= 2 and (cores[0].lower() in STOPWORDS or cores[-1].lower() in STOPWORDS)
            if bad_multi or (span == 1 and cores[0].lower() in COMMON_NON_PROPER):
                continue
            joined = "".join(cores)
            sk = latin_name_skeleton(joined)
            if len(sk) < 3:
                continue
            for nm, nm_sk in entries:
                if abs(len(sk) - len(nm_sk)) > 3:
                    continue
                if joined.lower() == nm.replace(" ", "").lower():
                    best_nm, best_span, best_sc = nm, span, 1.0
                    break
                sc = SequenceMatcher(None, sk, nm_sk).ratio()
                if sc >= (0.84 if span >= 2 else 0.86) and sc > best_sc:
                    best_sc, best_nm, best_span = sc, nm, span
            if best_sc == 1.0:
                break
        if best_nm and best_span > 0:
            pre = re.match(r"^([^A-Za-z0-9]*)", toks[i]).group(1)
            post = re.search(r"([^A-Za-z0-9]*)$", toks[i + best_span - 1]).group(1)
            orig_s = " ".join(re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", t) for t in toks[i : i + best_span])
            out_t.append(f"{pre}{best_nm}{post}")
            if orig_s != best_nm:
                rescued.append((orig_s, best_nm))
            i += best_span
        else:
            out_t.append(toks[i])
            i += 1
    return " ".join(out_t), rescued


def fallback_grapheme_phones(word: str) -> list:
    w = re.sub(r"[^a-z]", "", word.lower())
    if len(w) >= 5 and w[-1] in "uo" and w[-2] in "ptkbdgfsz":
        w = w[:-1]
    w = re.sub(r"tion$", "shn", re.sub(r"^kn", "n", re.sub(r"^wr", "r", w)))
    w = w.replace("oi", "oy").replace("igh", "ay").replace("ght", "t").replace("ph", "f").replace("ck", "k")
    if len(w) > 3 and w.endswith("e") and not w.endswith("ee"):
        w = w[:-1]
    tokens, i = [], 0
    vmap = {"A": "AE", "E": "EH", "I": "IH", "O": "AA", "U": "AH", "C": "K"}
    while i < len(w):
        if w[i : i + 2] in ("th", "sh", "ch", "ng", "ay", "oy"):
            tokens.append(w[i : i + 2].upper())
            i += 2
        elif w[i : i + 2] in ("ee", "ea"):
            tokens.append("IY")
            i += 2
        elif w[i : i + 2] == "oo":
            tokens.append("UW")
            i += 2
        else:
            ch = w[i].upper()
            tokens.append(vmap.get(ch, ch))
            i += 1
    return tokens or [word.upper()]


def word_to_arpabet(word: str) -> list:
    cw = word.lower().strip("'")
    if HAS_CMUDICT:
        pl = pronouncing.phones_for_word(cw)
        if not pl and len(cw) > 3 and cw.endswith("s"):
            bp = pronouncing.phones_for_word(cw[:-1])
            if bp:
                return [re.sub(r"\d", "", p) for p in bp[0].split()] + ["Z"]
        if pl:
            return [re.sub(r"\d", "", p) for p in pl[0].split()]
    return fallback_grapheme_phones(cw)


def _is_l2_droppable(phones: list, idx: int) -> bool:
    n, p = len(phones), phones[idx]
    if idx == n - 1 and p in DROPPABLE_CODA:
        return True
    if 1 <= idx <= 2 and p in {"L", "R"} and phones[idx - 1] not in ARPABET_VOWELS:
        return True
    return idx >= 1 and idx >= n - 3 and p in CLUSTER_REDUCIBLE and all(ph not in ARPABET_VOWELS for ph in phones[idx:])


def phonetic_similarity(w1: str, w2: str) -> float:
    c1, c2 = (w1 or "").lower().strip("'"), (w2 or "").lower().strip("'")
    if not c1 or not c2:
        return 0.0
    if c1 == c2:
        return 1.0
    ph1, ph2 = word_to_arpabet(c1), word_to_arpabet(c2)
    if ph1 == ph2:
        return 0.97
    m, n = len(ph1), len(ph2)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + (0.14 if _is_l2_droppable(ph1, i - 1) else 0.88)
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + (0.14 if _is_l2_droppable(ph2, j - 1) else 0.88)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dc = 0.14 if _is_l2_droppable(ph1, i - 1) else 0.88
            ic = 0.14 if _is_l2_droppable(ph2, j - 1) else 0.88
            p1, p2 = ph1[i - 1], ph2[j - 1]
            vow = 0.28 if (p1 in ARPABET_VOWELS and p2 in ARPABET_VOWELS) else 1.0
            sc = 0.0 if p1 == p2 else ARPABET_PAIR_COSTS.get(frozenset({p1, p2}), vow)
            dp[i][j] = min(dp[i - 1][j] + dc, dp[i][j - 1] + ic, dp[i - 1][j - 1] + sc)
    return float(max(0.0, 1.0 - (dp[m][n] / max(m, n, 1))))


def _dp_phrase_sim(w_low: list, win: list, learned_pairs: dict) -> tuple:
    m, n = len(w_low), len(win)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + (0.32 if w_low[i - 1] in STOPWORDS else 0.85)
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + (0.32 if win[j - 1] in STOPWORDS else 0.85)
    c_best = {}
    for i in range(1, m + 1):
        wl = w_low[i - 1]
        dc = 0.32 if wl in STOPWORDS else 0.85
        for j in range(1, n + 1):
            rl = win[j - 1]
            ic = 0.32 if rl in STOPWORDS else 0.85
            ws = 1.0 if wl == rl else (0.94 if f"{wl}->{rl}" in learned_pairs else phonetic_similarity(wl, rl))
            if rl not in STOPWORDS and abs(i - j) <= 3:
                c_best[j - 1] = max(c_best.get(j - 1, 0.0), ws)
            dp[i][j] = min(dp[i - 1][j] + dc, dp[i][j - 1] + ic, dp[i - 1][j - 1] + (1.0 - ws))
    sim = max(0.0, 1.0 - (dp[m][n] / max(m, n, 1)))
    c_vals = [c_best.get(j, 0.0) for j, (_, rl) in enumerate(win) if rl not in STOPWORDS]
    ok = len(c_vals) >= 2 and min(c_vals) >= 0.65 and sum(1 for v in c_vals if v >= 0.76) >= 2
    return sim, ok


def rescue_by_script_phrases(words: list, context_text: str, learned_pairs: dict) -> tuple:
    if len(words) < 3 or not context_text:
        return words, []
    clean_ctx = context_text.replace("[Speaker Notes]", " ")
    refs = [(r, r.lower().strip("'-")) for r in re.findall(r"[A-Za-z0-9'\-]+", clean_ctx) if r.lower().strip("'-")]
    if len(refs) < 3:
        return words, []

    L = len(words)
    w_low = [w.lower().strip("'-") for w in words]

    # 1) Full-Utterance Hybrid Alignment (>= 0.78 similarity with slide/script sentence)
    if L <= 18:
        best_sc, best_win = 0.0, None
        w_grps = [_init_grp(x) for x in w_low]
        for M in (L, L - 1, L + 1, L - 2, L + 2):
            if M < 3 or M > len(refs):
                continue
            for st in range(len(refs) - M + 1):
                win = refs[st : st + M]
                win_g = {_init_grp(rl) for _, rl in win}
                if sum(1 for g in w_grps if g in win_g) < max(2, int(L * 0.55)):
                    continue
                sim, ok = _dp_phrase_sim(w_low, win, learned_pairs)
                if ok and sim >= 0.78 and sim > best_sc:
                    best_sc, best_win = sim, win
        if best_win is not None:
            new_w = [r_raw for r_raw, _ in best_win]
            ref_l = [rl for _, rl in best_win]
            diffs = []
            for tag, i1, i2, j1, j2 in SequenceMatcher(None, w_low, ref_l).get_opcodes():
                if tag != "equal" and i1 < i2 and j1 < j2:
                    diffs.append((" ".join(words[i1:i2]), " ".join(new_w[j1:j2])))
            return new_w, diffs

    # 2) Sliding Sub-Phrase Alignment (4..8 word clauses inside mixed spontaneous speech)
    if L >= 5:
        out_w, used, sub_diffs = list(words), [False] * L, []
        for K in range(min(8, L), 3, -1):
            for i in range(L - K + 1):
                if any(used[i : i + K]):
                    continue
                sub_l = [out_w[i + d].lower().strip("'-") for d in range(K)]
                sub_g = [_init_grp(x) for x in sub_l]
                b_sc, b_win = 0.0, None
                for st in range(len(refs) - K + 1):
                    win = refs[st : st + K]
                    win_g = {_init_grp(rl) for _, rl in win}
                    if sum(1 for g in sub_g if g in win_g) < K - 1:
                        continue
                    sims = [
                        1.0 if wl == rl else (0.94 if f"{wl}->{rl}" in learned_pairs else phonetic_similarity(wl, rl))
                        for wl, (_, rl) in zip(sub_l, win)
                    ]
                    avg_s = sum(sims) / K
                    c_s = [s for s, (_, rl) in zip(sims, win) if rl not in STOPWORDS]
                    if len(c_s) >= 2 and avg_s >= 0.81 and min(c_s) >= 0.67 and min(sims) >= 0.50 and avg_s > b_sc:
                        b_sc, b_win = avg_s, win
                if b_win is not None:
                    for d in range(K):
                        r_raw, rl = b_win[d]
                        if out_w[i + d].lower().strip("'-") != rl:
                            sub_diffs.append((out_w[i + d], r_raw))
                        out_w[i + d] = r_raw
                        used[i + d] = True
        if sub_diffs:
            return out_w, sub_diffs

    return words, []


def normalize_audio_safe(raw_arr: np.ndarray, orig_sr: int) -> np.ndarray:
    arr = np.asarray(raw_arr)
    arr = (arr.mean(axis=1) if arr.ndim > 1 else arr).astype(np.float32)
    if len(arr) == 0:
        return arr
    if orig_sr != 16000:
        try:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(int(orig_sr), 16000)
            arr = resample_poly(arr, 16000 // g, int(orig_sr) // g).astype(np.float32)
        except Exception:
            import librosa
            arr = librosa.resample(arr, orig_sr=orig_sr, target_sr=16000).astype(np.float32)
    arr = np.append(arr[0], arr[1:] - 0.15 * arr[:-1]).astype(np.float32)
    p99 = float(np.percentile(np.abs(arr), 99.2))
    return np.clip(arr / p99, -1.0, 1.0) if p99 > 1e-6 else arr


def preprocess_and_chunk_audio(audio_16k: np.ndarray, sr: int = 16000, max_chunk_sec: float = 24.0) -> list:
    import librosa
    if audio_16k is None or len(audio_16k) < int(0.25 * sr):
        return []
    trimmed, _ = librosa.effects.trim(audio_16k.astype(np.float32), top_db=34)
    if len(trimmed) < int(0.25 * sr) or float(np.sqrt(np.mean(np.square(trimmed)))) < 8e-5:
        return []
    pad = np.zeros(int(0.15 * sr), dtype=np.float32)
    trimmed = np.concatenate([pad, trimmed, pad])
    max_len = int(max_chunk_sec * sr)
    if len(trimmed) <= max_len:
        return [trimmed]
    chunks, cur, min_len, win = [], 0, int(14.0 * sr), int(0.25 * sr)
    while cur < len(trimmed):
        if len(trimmed) - cur <= max_len:
            chunks.append(trimmed[cur:])
            break
        zone = trimmed[cur + min_len : cur + max_len]
        rng = range(0, max(1, len(zone) - win), win // 2)
        best_p = min(rng, key=lambda p: float(np.mean(np.square(zone[p : p + win]))))
        best_off = min_len + best_p + win // 2
        chunks.append(trimmed[cur : cur + best_off])
        cur += best_off
    return [c for c in chunks if len(c) >= int(0.25 * sr)]


class AccentBridgeEngine:
    def __init__(self, finetuned_dir: str = "./saved_model", base_id: str = "openai/whisper-base.en"):
        try:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        except Exception:
            self.device = "cpu"
        self.finetuned_dir, self.base_id = finetuned_dir, base_id
        self.adapter_cfg = os.path.join(finetuned_dir, "adapter_config.json")
        has_cfg = os.path.exists(os.path.join(finetuned_dir, "config.json"))
        self.using_finetuned = os.path.exists(self.adapter_cfg) or has_cfg
        self.processor = self.baseline_model = self.ft_model = self.lm_tok = self.lm_model = None
        self.learned_confusions = {
            "word_pairs": {"fight->point": 8, "fights->points": 8, "pint->point": 7, "poy->point": 8},
            "thai_translit_stems": dict(DEFAULT_TH_STEMS),
        }
        self.priors_path = os.path.join(finetuned_dir, "confusion_priors.json")
        base_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "confusion_priors.json")
        for p in (base_p, "confusion_priors.json", self.priors_path):
            if p and os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        d = json.load(f)
                    self.learned_confusions["word_pairs"].update(d.get("word_pairs", {}))
                    self.learned_confusions["thai_translit_stems"].update(d.get("thai_translit_stems", {}))
                except Exception:
                    pass
        self.calib = {"alpha": 0.15, "hw_weight": 0.22, "gate_threshold": 0.90}
        cp = os.path.join(finetuned_dir, "calibrated_params.json")
        if os.path.exists(cp):
            try:
                with open(cp, "r", encoding="utf-8") as f:
                    self.calib.update(json.load(f))
            except Exception:
                pass
        self.last_word_confidences = {}

    def _ensure_whisper(self, need_base: bool = False):
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        dt = torch.float16 if self.device == "cuda" else torch.float32
        if self.processor is None:
            self.processor = WhisperProcessor.from_pretrained(self.base_id)
        if need_base and self.baseline_model is None:
            m = WhisperForConditionalGeneration.from_pretrained(self.base_id, torch_dtype=dt)
            self.baseline_model = m.to(self.device).eval()
        if self.ft_model is None:
            try:
                if os.path.exists(self.adapter_cfg):
                    from peft import PeftModel
                    b = WhisperForConditionalGeneration.from_pretrained(self.base_id, torch_dtype=dt)
                    self.ft_model = PeftModel.from_pretrained(b, self.finetuned_dir).merge_and_unload().to(self.device)
                elif os.path.exists(os.path.join(self.finetuned_dir, "config.json")):
                    m = WhisperForConditionalGeneration.from_pretrained(self.finetuned_dir, torch_dtype=dt)
                    self.ft_model = m.to(self.device)
                else:
                    m = WhisperForConditionalGeneration.from_pretrained(self.base_id, torch_dtype=dt)
                    self.ft_model = self.baseline_model or m.to(self.device)
            except Exception:
                m = WhisperForConditionalGeneration.from_pretrained(self.base_id, torch_dtype=dt)
                self.ft_model = m.to(self.device)
            self.ft_model.eval()

    def _ensure_lm(self):
        if self.lm_model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        dt = torch.float16 if self.device == "cuda" else torch.float32
        self.lm_tok = AutoTokenizer.from_pretrained("distilgpt2")
        if self.lm_tok.pad_token is None:
            self.lm_tok.pad_token = self.lm_tok.eos_token
        m = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=dt)
        self.lm_model = m.to(self.device).eval()

    def score_sentences_batched(self, texts: list, slide_topic: str = "") -> list:
        import torch
        import torch.nn.functional as F
        if not texts:
            return []
        self._ensure_lm()
        tc = normalize_text(slide_topic)[:50]
        pref = f"topic {tc}: " if tc else ""
        prep = [(pref + t.strip() if len(t.strip().split()) >= 2 else t.strip()) or "empty" for t in texts]
        enc = self.lm_tok(prep, return_tensors="pt", padding=True, truncation=True, max_length=160).to(self.device)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        if ids.ndim < 2 or ids.size(1) < 2:
            return [-8.0] * len(texts)
        with getattr(torch, "inference_mode", torch.no_grad)():
            logits = self.lm_model(input_ids=ids, attention_mask=mask).logits.float()
            sl = logits[:, :-1, :].contiguous()
            sb = ids[:, 1:].contiguous()
            sm = mask[:, 1:].contiguous().float()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), sb.view(-1), reduction="none").view(sb.size())
            seq = (loss * sm).sum(dim=1) / sm.sum(dim=1).clamp(min=1.0)
        return [sc if len(t.strip().split()) >= 2 else -8.0 for sc, t in zip((-seq).cpu().tolist(), texts)]

    def parse_hotwords(self, hotwords_raw, context_text: str = "") -> list:
        comb = []
        if hotwords_raw:
            items = re.split(r"[,\n;]+", hotwords_raw) if isinstance(hotwords_raw, str) else list(hotwords_raw)
            for w in items:
                nw = normalize_text(w)
                if nw and nw not in comb:
                    comb.append(nw)
        if context_text:
            for aw in extract_priority_keywords(context_text, max_keywords=42):
                if aw not in comb:
                    comb.append(aw)
        return comb[:65]

    def _prior_bonus(self, mis: str, tgt: str) -> float:
        c = self.learned_confusions.get("word_pairs", {}).get(f"{mis}->{tgt}", 0)
        return min(0.20, 0.09 + 0.015 * float(c)) if c else 0.0

    def rescue_code_switching(self, text: str, hotwords_raw=None, context_text: str = "", slide_topic: str = "") -> tuple:
        if not text:
            return text, []
        p_names = extract_proper_names(slide_topic, context_text, str(hotwords_raw or ""))
        out_t, rescued = rescue_proper_names_in_text(text, p_names)
        if not re.search(r"[\u0E00-\u0E7F]", out_t):
            return out_t, rescued

        # Thai script phrase-snapping (>= 0.82 match with Thai speaker notes/slide clause)
        if context_text and len(out_t) >= 8:
            th_clauses = [ln.strip() for ln in re.split(r"[\n.!?]+", context_text.replace("[Speaker Notes]", "\n")) if re.search(r"[\u0E00-\u0E7F]", ln) and 8 <= len(ln.strip()) <= 80]
            best_cl, best_r = None, 0.0
            for cl in th_clauses:
                r = SequenceMatcher(None, out_t, cl).ratio()
                if r >= 0.82 and r > best_r:
                    best_r, best_cl = r, cl
            if best_cl and best_cl != out_t:
                rescued.append((out_t, best_cl))
                out_t = best_cl

        ac_raw = {a.lower() for a in extract_acronyms(context_text)}
        ctx_words = set(re.findall(r"[a-z0-9\-]+", (context_text or "").lower()))
        vocab = set(self.parse_hotwords(hotwords_raw, context_text)) | ac_raw | ctx_words
        acrs = ac_raw | {"api", "gpu", "cpu", "wer"}
        stems = self.learned_confusions.get("thai_translit_stems", DEFAULT_TH_STEMS)
        for th_s, en_t in sorted(stems.items(), key=lambda kv: len(kv[0]), reverse=True):
            el = en_t.lower()
            if th_s in out_t and (el in vocab or all(w in vocab for w in el.split()) or f"{el}s" in vocab):
                fmt = en_t.upper() if el in acrs else en_t
                out_t = out_t.replace(th_s, f" {fmt} ")
                rescued.append((th_s, fmt))
        return re.sub(r"\s+", " ", out_t).strip(), rescued

    def transcribe_baseline(self, audio_16k) -> str:
        import torch
        self._ensure_whisper(need_base=True)
        chunks = preprocess_and_chunk_audio(np.asarray(audio_16k, dtype=np.float32))
        if not chunks:
            return ""
        dt = torch.float16 if self.device == "cuda" else torch.float32
        parts = []
        with getattr(torch, "inference_mode", torch.no_grad)():
            for ch in chunks:
                f = self.processor(ch, sampling_rate=16000, return_tensors="pt").input_features.to(self.device, dtype=dt)
                ids = self.baseline_model.generate(f, max_new_tokens=96, no_repeat_ngram_size=3, repetition_penalty=1.18)
                t = normalize_text(self.processor.batch_decode(ids, skip_special_tokens=True)[0])
                if t and not (len(chunks) == 1 and len(ch) < 24000 and t in HALLUCINATION_PHRASES):
                    parts.append(t)
        return " ".join(parts).strip()

    def calibrate_personal_voice(self, audio_16k: np.ndarray, expected_sentence: str) -> dict:
        import librosa
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import WhisperForConditionalGeneration
        self._ensure_whisper(need_base=True)
        exp = normalize_text(expected_sentence)
        before = self.transcribe_baseline(audio_16k)
        learned = []
        pd = self.learned_confusions.setdefault("word_pairs", {})
        hw_l, rw_l = before.split(), exp.split()
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, hw_l, rw_l).get_opcodes():
            if tag == "replace" and (i2 - i1) == 1 and (j2 - j1) == 1:
                if hw_l[i1] != rw_l[j1] and rw_l[j1] not in STOPWORDS:
                    pd[f"{hw_l[i1]}->{rw_l[j1]}"] = pd.get(f"{hw_l[i1]}->{rw_l[j1]}", 0) + 5
                    learned.append(f"'{hw_l[i1]}' → '{rw_l[j1]}'")
        os.makedirs(self.finetuned_dir, exist_ok=True)
        with open(self.priors_path, "w", encoding="utf-8") as f:
            json.dump(self.learned_confusions, f, ensure_ascii=False, indent=2)
        chunks = preprocess_and_chunk_audio(audio_16k)
        if chunks:
            c = chunks[0]
            augs = [c, librosa.effects.time_stretch(c, rate=0.94), librosa.effects.time_stretch(c, rate=1.06)]
            ab = WhisperForConditionalGeneration.from_pretrained(self.base_id).to(self.device)
            ab.config.use_cache = False
            ab.enable_input_require_grads()
            lcfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0, bias="none")
            pm = get_peft_model(ab, lcfg)
            pm.train()
            opt = torch.optim.AdamW(pm.parameters(), lr=1e-4)
            lbl = self.processor.tokenizer(exp, return_tensors="pt", truncation=True, max_length=128).input_ids.to(self.device)
            for s in range(6 if self.device != "cpu" else 3):
                opt.zero_grad()
                ft = self.processor(augs[s % len(augs)], sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
                pm(input_features=ft, labels=lbl).loss.backward()
                opt.step()
            m = pm.eval().merge_and_unload()
            self.ft_model = (m.half() if self.device == "cuda" else m).to(self.device).eval()
            self.using_finetuned = True
        after, _ = self.transcribe_accentbridge(audio_16k, hotwords_raw=exp, context_text=exp)
        return {"before": before, "after": after, "learned_pairs": learned}

    def _apply_hotwords(self, sent: str, hotwords: list, acronyms: list, bigrams: set, w_confs: dict, topic: str, ctx: str, raw_hw: str, use_gate: bool, gate_th: float):
        if not sent:
            return sent, []
        sent, r_pt = rescue_collocations_and_point(sent)
        p_names = extract_proper_names(topic, ctx, raw_hw)
        sent, r_nm = rescue_proper_names_in_text(sent, p_names)
        p_set, p_low = set(p_names), {n.lower(): n for n in p_names}

        wp_dict = self.learned_confusions.get("word_pairs", {})
        ph_words, r_ph = rescue_by_script_phrases(sent.split(), f"{topic}\n{ctx}", wp_dict)

        words, r_ac, ac_set = [], [], {a.lower() for a in acronyms if 2 <= len(a) <= 5}
        idx = 0
        while idx < len(ph_words):
            hit_l, hit_a = 0, None
            for sp in (5, 4, 3, 2):
                seg = ph_words[idx : idx + sp]
                if idx + sp <= len(ph_words) and all(len(x) == 1 for x in seg) and "".join(seg).lower() in ac_set:
                    hit_l, hit_a = sp, "".join(seg).upper()
                    break
            if hit_a:
                r_ac.append((" ".join(ph_words[idx : idx + hit_l]), hit_a))
                words.append(hit_a)
                idx += hit_l
            else:
                words.append(ph_words[idx])
                idx += 1

        rescued = list(r_pt) + list(r_nm) + list(r_ph) + list(r_ac)
        prior_targets = [k.split("->", 1) for k, v in wp_dict.items() if "->" in k and v >= 6 and " " not in k]
        singles = list(dict.fromkeys([h for h in hotwords if " " not in h] + prior_targets))
        if not singles:
            return " ".join(words), rescued

        lm_trials = []
        for i, tok in enumerate(words):
            if tok in p_set:
                continue
            ow = tok.lower().strip(".,!?;:")
            if ow in STOPWORDS or ow in hotwords:
                if ow in p_low:
                    words[i] = p_low[ow]
                continue
            best_h, best_s, has_b, pb_v = None, 0.0, False, 0.0
            for hw in singles:
                if abs(len(hw) - len(ow)) > 3 or hw == ow:
                    continue
                pb = self._prior_bonus(ow, hw)
                if hw not in hotwords and pb == 0.0:
                    continue
                rp = phonetic_similarity(ow, hw)
                if rp < 0.78 and pb == 0.0:
                    continue
                prev_ok = i > 0 and (words[i - 1].lower(), hw) in bigrams
                next_ok = i < len(words) - 1 and (hw, words[i + 1].lower()) in bigrams
                bg_ok = prev_ok or next_ok
                es = rp + pb + (0.08 if bg_ok else 0.0)
                if es > best_s:
                    best_s, best_h, has_b, pb_v = es, hw, bg_ok, pb
            if use_gate and w_confs.get(ow, 0.65) >= gate_th and pb_v == 0.0:
                continue
            if best_h and best_s >= 0.78:
                nhw = p_low.get(best_h, best_h)
                if has_b or best_s >= 0.90:
                    rescued.append((tok, nhw))
                    words[i] = nhw
                else:
                    lm_trials.append((i, tok, nhw, best_s))

        # AI Contextual Sentence Probability Scoring for ambiguous off-script words
        if lm_trials and len(words) >= 3:
            try:
                base_s = " ".join(words)
                cands = []
                for i, _, nhw, _ in lm_trials:
                    tw = words.copy()
                    tw[i] = nhw
                    cands.append(" ".join(tw))
                scores = self.score_sentences_batched([base_s] + cands, slide_topic=topic)
                base_sc = scores[0]
                for (i, tok, nhw, bs), new_sc in zip(lm_trials, scores[1:]):
                    if (new_sc - base_sc) * 0.35 + (bs - 0.82) >= -0.04:
                        rescued.append((tok, nhw))
                        words[i] = nhw
            except Exception:
                for i, tok, nhw, bs in lm_trials:
                    if bs >= 0.85:
                        rescued.append((tok, nhw))
                        words[i] = nhw

        return " ".join(words), rescued

    def rescue_text(self, text: str, hotwords_raw=None, context_text: str = "", slide_topic: str = ""):
        rhw = hotwords_raw if isinstance(hotwords_raw, str) else ", ".join(hotwords_raw or [])
        if text and re.search(r"[\u0E00-\u0E7F]", text):
            return self.rescue_code_switching(text, hotwords_raw=rhw, context_text=context_text, slide_topic=slide_topic)
        sent = normalize_text(text)
        if not sent:
            return "", []
        hws = self.parse_hotwords(hotwords_raw, context_text=context_text)
        acs = extract_acronyms(context_text)
        bgs = extract_context_bigrams(context_text)
        return self._apply_hotwords(sent, hws, acs, bgs, {}, slide_topic, context_text, rhw, False, 0.90)

    def transcribe_accentbridge(self, audio_16k, hotwords_raw=None, context_text: str = "", slide_topic: str = "", alpha: float = None, hw_weight: float = None, gate_thresh: float = None, use_confidence_gate: bool = True):
        import torch
        self._ensure_whisper(need_base=False)
        chunks = preprocess_and_chunk_audio(np.asarray(audio_16k, dtype=np.float32))
        if not chunks:
            self.last_word_confidences = {}
            return "", []
        rhw = hotwords_raw if isinstance(hotwords_raw, str) else ", ".join(hotwords_raw or [])
        hws = self.parse_hotwords(hotwords_raw, context_text)
        acs = extract_acronyms(context_text)
        bgs = extract_context_bigrams(context_text)
        dt = torch.float16 if self.device == "cuda" else torch.float32
        sents, rescs = [], []
        gth = gate_thresh or self.calib["gate_threshold"]
        with getattr(torch, "inference_mode", torch.no_grad)():
            for ch in chunks:
                ft = self.processor(ch, sampling_rate=16000, return_tensors="pt").input_features.to(self.device, dtype=dt)
                ids = self.ft_model.generate(ft, max_new_tokens=96, no_repeat_ngram_size=3, repetition_penalty=1.18)
                raw = normalize_text(self.processor.batch_decode(ids, skip_special_tokens=True)[0])
                if raw and not (len(ch) < 24000 and raw in HALLUCINATION_PHRASES):
                    st, rc = self._apply_hotwords(raw, hws, acs, bgs, {}, slide_topic, context_text, rhw, use_confidence_gate, gth)
                    if st:
                        sents.append(st)
                        rescs.extend(rc)
        self.last_word_confidences = {}
        return " ".join(sents).strip(), rescs
