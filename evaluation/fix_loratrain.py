# fix_loratrain.py
# -- coding: utf-8 --

import json
import re
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# ======================
# Ayarlar
# ======================
IN_PATH = "lora_train_xgb.jsonl"
OUT_PATH = "loratrain_fixed.jsonl"
REPORT_PATH = "loratrain_fix_report.json"

SEED = 42
random.seed(SEED)

HYPER_THR = 140.0   # >= 140 -> hiperglisemi uyarı
HYPO_THR  = 70.0    # < 70  -> hipoglisemi uyarı

# tıbbi sınırlar: ilaç/insülin dozu önerme yok.
MSG_TEMPLATES = {
    "HYPER": [
        "Tahmin edilen değer yüksek aralıkta; öğün sonrası hiperglisemi riski olabilir. Trend yükselişi destekliyorsa daha sık takip etmek faydalı olur.",
        "Değer 2 saat sonrası için yüksek görünüyor; özellikle karbonhidrat yükü ve son trend bu sonucu destekleyebilir. Belirti olursa tıbbi destek düşünülmelidir.",
        "Tahmin yüksek aralıkta; öğün sonrası yükseliş bekleniyor olabilir. Kişisel değişkenlik olabileceği için takip önemlidir.",
        "Beklenen değer 140 mg/dL üzerinde; hiperglisemi riski açısından dikkatli olunmalı ve ölçümler izlenmelidir.",
    ],
    "NORMAL": [
        "Tahmin edilen değer hedef aralıkta; son 2 saatlik trend stabilse normal seyir beklenebilir. Yine de takip etmeyi sürdür.",
        "Değer normal aralıkta görünüyor; makro içeriği ve CGM trendi bu sonucu destekleyebilir. Kişisel farklılıklar için izlem önemlidir.",
        "Tahmin normal seviyede; öğün içeriği ve mevcut trendle uyumlu görünüyor. Düzenli takip önerilir.",
        "Beklenen değer normal aralıkta; herhangi bir belirti olursa durumu izlemek faydalı olur.",
    ],
    "HYPO": [
        "Tahmin edilen değer düşük aralıkta; hipoglisemi riski olabilir. Belirti olursa hızlıca destek almak ve ölçümü tekrar etmek önemlidir.",
        "Değer 70 mg/dL altı görünüyor; düşüş trendi varsa dikkatli olunmalı ve takip sıklaştırılmalıdır.",
        "Tahmin düşük aralıkta; özellikle aktivite/diğer etkenler varsa hipoglisemi riski artabilir. Belirti olursa tıbbi destek düşünülmelidir.",
        "Beklenen değer düşük seviyede; hipoglisemi açısından dikkatli olunmalı ve ölçümler izlenmelidir.",
    ],
}

PLACEHOLDER_PATTERNS = [
    r"<sayı>", r"<Normal veya Uyarı>", r"<kısa.?>", r"<1-2.?>", r"<etiket.*?>", r"1-2 cümle", r"etiket\+carbs",
    r"Bu bir örnek cümledir", r"yerine gerçek", r"<\.\.\.>"
]

RE_T120 = re.compile(r"t\+120[_ ]glukoz[_ ]mgdl\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE)
RE_CARBS = re.compile(r"carb(?:s)?\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE)
RE_PROT  = re.compile(r"protein\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE)
RE_FAT   = re.compile(r"fat|yağ|yag", re.IGNORECASE)

def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def detect_placeholders(text: str) -> bool:
    if not text:
        return True
    for p in PLACEHOLDER_PATTERNS:
        if re.search(p, text, flags=re.IGNORECASE):
            return True
    return False

def extract_from_user_text(user_text: str) -> Tuple[Optional[float], Optional[str], Optional[float], Optional[float], Optional[float]]:
    """
    Kullanıcı mesajından:
    - t+120 sayısı (XGB)
    - yemek etiketi/özeti
    - carbs, protein, fat
    Çıkarmaya çalışır. Bulamazsa None döner.
    """
    t120 = None
    m = RE_T120.search(user_text or "")
    if m:
        t120 = _safe_float(m.group(1))

    # food label/summary: "Yemek etiketi/özeti:" veya "Yemek etiketi:" satırı
    food = None
    for key in ["Yemek etiketi/özeti:", "Yemek etiketi:", "Yemek:", "Yemek etiketi/ozeti:"]:
        idx = (user_text or "").find(key)
        if idx != -1:
            line = (user_text or "")[idx:].splitlines()[0]
            food = line.split(":", 1)[-1].strip()
            if food:
                break

    # makrolar
    carbs = None
    prot = None
    fat = None

    # “- Karbonhidrat: 22.0 g” gibi
    m = re.search(r"Karbonhidrat\s*:\s*([-+]?\d+(?:\.\d+)?)", user_text or "", flags=re.IGNORECASE)
    if m:
        carbs = _safe_float(m.group(1))
    m = re.search(r"Protein\s*:\s*([-+]?\d+(?:\.\d+)?)", user_text or "", flags=re.IGNORECASE)
    if m:
        prot = _safe_float(m.group(1))
    m = re.search(r"Yağ\s*:\s*([-+]?\d+(?:\.\d+)?)", user_text or "", flags=re.IGNORECASE)
    if m:
        fat = _safe_float(m.group(1))

    return t120, food, carbs, prot, fat

def extract_from_meta(meta: Dict[str, Any]) -> Tuple[Optional[float], Optional[str], Optional[float], Optional[float], Optional[float]]:
    """
    meta içinden:
    - xgb_pred_t120 veya t+120
    - food / meal label (varsa)
    - carbs/protein/fat (varsa)
    """
    if not isinstance(meta, dict):
        return None, None, None, None, None

    t120 = _safe_float(meta.get("xgb_pred_t120"))
    if t120 is None:
        t120 = _safe_float(meta.get("t120_pred"))
    if t120 is None:
        t120 = _safe_float(meta.get("t+120_glukoz_mgdl"))

    food = meta.get("food") or meta.get("food_label") or meta.get("meal_type") or meta.get("meal") or meta.get("yemek")

    carbs = _safe_float(meta.get("carbs")) or _safe_float(meta.get("carbs_g"))
    prot  = _safe_float(meta.get("protein")) or _safe_float(meta.get("protein_g"))
    fat   = _safe_float(meta.get("fat")) or _safe_float(meta.get("fat_g"))

    return t120, (str(food) if food else None), carbs, prot, fat

def pick_status_and_msg(t120: float) -> Tuple[str, str]:
    if t120 < HYPO_THR:
        return "Uyarı", random.choice(MSG_TEMPLATES["HYPO"])
    if t120 >= HYPER_THR:
        return "Uyarı", random.choice(MSG_TEMPLATES["HYPER"])
    return "Normal", random.choice(MSG_TEMPLATES["NORMAL"])

def build_reference(food: str, carbs: float, prot: float, fat: float) -> str:
    # boşsa "unknown" yap
    food = (food or "unknown").strip()
    return f"{food} | carbs={carbs:.1f}g protein={prot:.1f}g fat={fat:.1f}g"

def force_4lines(t120: float, status: str, msg: str, ref: str) -> str:
    # kesin 4 satır
    msg = (msg or "").strip()
    ref = (ref or "").strip()

    # fazla satır/boşluk temizliği
    msg = " ".join(msg.split())
    ref = " ".join(ref.split())

    return (
        f"t+120_glukoz_mgdl: {t120:.1f}\n"
        f"durum: {status}\n"
        f"mesaj: {msg}\n"
        f"referans: {ref}\n"
    )

def fix_one_record(obj: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    returns: (fixed_obj or None, reason)
    """
    if not isinstance(obj, dict):
        return None, "not_a_dict"

    messages = obj.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None, "missing_messages"

    # user text
    user_msg = None
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            user_msg = m.get("content", "")
            break

    # assistant msg
    asst_idx = None
    for i, m in enumerate(messages):
        if isinstance(m, dict) and m.get("role") == "assistant":
            asst_idx = i
            break

    meta = obj.get("meta", {}) if isinstance(obj.get("meta", {}), dict) else {}

    # önce meta -> sonra user text’ten al
    t120_m, food_m, c_m, p_m, f_m = extract_from_meta(meta)
    t120_u, food_u, c_u, p_u, f_u = extract_from_user_text(user_msg or "")

    t120 = t120_m if t120_m is not None else t120_u
    food = food_u if food_u else food_m
    carbs = c_u if c_u is not None else c_m
    prot  = p_u if p_u is not None else p_m
    fat   = f_u if f_u is not None else f_m

    # zorunlu alanlar
    if t120 is None:
        return None, "t120_not_found"

    # makrolar yoksa 0 yap (ama referans yine dolsun)
    carbs = float(carbs) if carbs is not None else 0.0
    prot  = float(prot) if prot is not None else 0.0
    fat   = float(fat) if fat is not None else 0.0
    food  = food if food else "unknown"

    status, msg = pick_status_and_msg(float(t120))
    ref = build_reference(str(food), carbs, prot, fat)

    new_asst = force_4lines(float(t120), status, msg, ref)

    # assistant yoksa ekle
    if asst_idx is None:
        messages.append({"role": "assistant", "content": new_asst})
    else:
        messages[asst_idx]["content"] = new_asst

    # meta'ya referans/etiket ekle (opsiyonel ama faydalı)
    meta.setdefault("fixed_by", "fix_loratrain.py")
    meta.setdefault("hyper_thr", HYPER_THR)
    meta.setdefault("hypo_thr", HYPO_THR)
    meta["xgb_pred_t120"] = float(t120)
    meta["food_label_summary"] = str(food)
    meta["ref_line"] = ref
    obj["meta"] = meta
    obj["messages"] = messages

    return obj, "ok"

def main():
    in_path = Path(IN_PATH)
    out_path = Path(OUT_PATH)
    rep_path = Path(REPORT_PATH)

    if not in_path.exists():
        raise FileNotFoundError(f"Girdi bulunamadı: {in_path.resolve()}")

    n_total = 0
    n_ok = 0
    n_skipped = 0
    reasons = {}

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                obj = json.loads(line)
            except Exception:
                n_skipped += 1
                reasons["json_parse_error"] = reasons.get("json_parse_error", 0) + 1
                continue

            fixed, reason = fix_one_record(obj)
            if fixed is None:
                n_skipped += 1
                reasons[reason] = reasons.get(reason, 0) + 1
                continue

            # güvenlik: placeholder kalmış mı?
            # assistant içeriğini kontrol et
            asst_text = ""
            for m in fixed.get("messages", []):
                if isinstance(m, dict) and m.get("role") == "assistant":
                    asst_text = m.get("content", "")
                    break
            if detect_placeholders(asst_text):
                n_skipped += 1
                reasons["placeholder_left"] = reasons.get("placeholder_left", 0) + 1
                continue

            fout.write(json.dumps(fixed, ensure_ascii=False) + "\n")
            n_ok += 1

    report = {
        "in_path": str(in_path.resolve()),
        "out_path": str(out_path.resolve()),
        "n_total": n_total,
        "n_ok": n_ok,
        "n_skipped": n_skipped,
        "skip_reasons": reasons,
        "hyper_thr": HYPER_THR,
        "hypo_thr": HYPO_THR,
        "seed": SEED,
    }

    rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ FIX BİTTİ")
    print(json.dumps(report, ensure_ascii=False, indent=2))

