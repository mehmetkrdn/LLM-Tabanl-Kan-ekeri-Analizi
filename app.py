# app.py
# -- coding: utf-8 --

import os
import json
import csv
import re
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import xgboost as xgb

# joblib sadece .joblib için lazım
try:
    import joblib
except Exception:
    joblib = None

# =========================
# UI
# =========================
st.set_page_config(page_title="Diyabet Asistanı (Foto → RAG → XGB → LLM)", layout="centered")
st.title("🍽️ Diyabet Asistanı: Foto → RAG Makro → XGBoost t+120 → LLM Yorum")
st.caption("Uyarı: Bu uygulama tıbbi tavsiye vermez; ilaç/insülin dozu önermez. Tahmin + bilgilendirici yorum üretir.")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("⚙️ Ayarlar")

    st.subheader("📷 Food-101")
    FOOD_PTH = st.text_input("Food-101 .pth yolu", value="food101_resnet50_best.pth")
    conf_threshold = st.slider("Food-101 güven eşiği", 0.50, 0.95, 0.70, 0.01)
    topk_food = st.slider("Food-101 Top-K", 3, 10, 5, 1)

    st.divider()
    st.subheader("📚 RAG (besinbilgileri.json)")
    NUTR_JSON = st.text_input("Besin JSON yolu", value="besinbilgileri.json")
    rag_topk = st.slider("RAG Top-K", 3, 10, 5, 1)
    portion_mult = st.slider("Porsiyon çarpanı", 0.5, 3.0, 1.0, 0.1)
    min_sim = st.slider("Min similarity (coverage)", 0.10, 0.80, 0.35, 0.01)

    st.divider()
    st.subheader("📈 XGBoost")
    XGB_PATH = st.text_input("XGBoost model yolu (.joblib / .json / .ubj / .bst)", value="xgb_model.joblib")

    st.divider()
    st.subheader("🤖 LLM (LoRA)")
    BASE_MODEL = st.text_input("Base model", value="Qwen/Qwen2.5-3B-Instruct")
    ADAPTER_DIR = st.text_input("Adapter klasörü", value="/content/drive/MyDrive/QWEN2.5-3B")
    max_new_tokens = st.slider("Max new tokens", 60, 350, 200, 10)

    st.divider()
    st.subheader("🚨 Eşikler")
    hyper_thr = st.number_input("Hiperglisemi eşiği (mg/dL)", value=140.0, step=1.0)
    hypo_thr = st.number_input("Hipoglisemi eşiği (mg/dL)", value=70.0, step=1.0)

    st.divider()
    st.subheader("🧾 Log")
    ENABLE_LOG = st.checkbox("Log yaz", value=True)
    LOG_DIR = st.text_input("Log klasörü", value="logs")
    RUN_TAG = st.text_input("Run etiketi", value="run_001")

    st.divider()
    st.subheader("🧪 Debug")
    DEBUG = st.checkbox("Debug çıktıları göster", value=True)

    st.divider()
    preload = st.button("🚀 Ön yükle (1 kere)")

# =========================
# Helpers
# =========================
def ensure_dir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def append_jsonl(path: str, obj: Dict[str, Any]):
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def append_csv_row(path: str, header: List[str], row: Dict[str, Any]):
    ensure_dir(os.path.dirname(path))
    file_exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow(row)

# =========================
# Loaders (cached)
# =========================
@st.cache_resource
def load_food_model(pth_path: str):
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Food-101 pth bulunamadı: {pth_path}")

    ckpt = torch.load(pth_path, map_location=DEVICE)
    class_names = ckpt["class_names"]

    m = models.resnet50(pretrained=False)
    m.fc = nn.Linear(m.fc.in_features, len(class_names))
    m.load_state_dict(ckpt["model_state_dict"])
    m.to(DEVICE).eval()

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return m, class_names, tfm

@st.cache_resource
def load_rag_index(json_path: str):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Besin JSON bulunamadı: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    if isinstance(data, dict):
        for k, v in data.items():
            rec = {"key": k}
            if isinstance(v, dict):
                rec.update(v)
            records.append(rec)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                records.append(item.copy())
    else:
        raise ValueError("besinbilgileri.json formatı dict veya list olmalı.")

    def get_name(r: Dict[str, Any]) -> str:
        for cand in ["name", "title", "food", "label", "key", "yemek", "isim"]:
            if cand in r and r[cand]:
                return str(r[cand])
        return str(r.get("key", "unknown"))

    def get_macro(r: Dict[str, Any], candidates: List[str], default=0.0) -> float:
        for c in candidates:
            if c in r and r[c] is not None and str(r[c]).strip() != "":
                try:
                    return float(r[c])
                except Exception:
                    pass
        return float(default)

    docs = []
    for r in records:
        name = get_name(r)
        carbs = get_macro(r, ["carbs_g", "carb_g", "carbs", "carb", "karbonhidrat_g", "karbonhidrat"])
        protein = get_macro(r, ["protein_g", "protein", "protein_gr"])
        fat = get_macro(r, ["fat_g", "fat", "yag_g", "yağ", "yag"])
        text = f"{name} | carbs_g: {carbs} | protein_g: {protein} | fat_g: {fat}"
        docs.append({
            "name": name,
            "carbs_g": carbs,
            "protein_g": protein,
            "fat_g": fat,
            "text": text
        })

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode([d["text"] for d in docs], normalize_embeddings=True)
    return docs, embeddings, embedder

@st.cache_resource
def load_xgb_any(path: str):
    """
    Destek:
    - Booster formatları: .json, .ubj, .bst
    - joblib: xgboost.XGBRegressor / Booster / sklearn wrapper
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"XGBoost modeli bulunamadı: {path}")

    lower = path.lower()
    if lower.endswith((".json", ".ubj", ".bst")):
        booster = xgb.Booster()
        booster.load_model(path)
        return ("booster", booster)

    if lower.endswith(".joblib"):
        if joblib is None:
            raise RuntimeError("joblib yüklü değil. pip install joblib kur veya booster (.json/.ubj/.bst) kullan.")
        obj = joblib.load(path)
        if isinstance(obj, xgb.Booster):
            return ("booster", obj)
        return ("sklearn", obj)

    booster = xgb.Booster()
    booster.load_model(path)
    return ("booster", booster)

@st.cache_resource
def load_llm(base_model_name: str, adapter_dir: str):
    if not os.path.exists(adapter_dir):
        raise FileNotFoundError(f"Adapter klasörü bulunamadı: {adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return tokenizer, model

# =========================
# Core: Food / RAG / XGB / LLM
# =========================
def food_topk(img: Image.Image, food_model, class_names, tfm, k: int) -> List[Tuple[str, float]]:
    x = tfm(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = food_model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    idxs = np.argsort(probs)[-k:][::-1]
    return [(class_names[i], float(probs[i])) for i in idxs]

def rag_topk_lookup(query: str, docs, doc_embeds, embedder, k: int) -> List[Dict[str, Any]]:
    q = embedder.encode([query], normalize_embeddings=True)
    sims = cosine_similarity(q, doc_embeds)[0]
    idxs = np.argsort(sims)[-k:][::-1]
    out = []
    for i in idxs:
        d = docs[int(i)].copy()
        d["similarity"] = float(sims[int(i)])
        out.append(d)
    return out

def compute_cgm_trend(g_m120: float, g_m60: float, g_0: float) -> str:
    total = g_0 - g_m120
    if abs(total) < 8:
        return "stabil"
    if total > 0:
        return "yükselen"
    return "düşen"

def xgb_predict_t120(model_kind: str, model_obj, feats: Dict[str, float]) -> float:
    order = ["Carbs", "Protein", "Fat", "g_m120", "g_m60", "g_0"]
    x_arr = np.array([[float(feats.get(c, 0.0)) for c in order]], dtype=np.float32)

    if model_kind == "booster":
        dmat = xgb.DMatrix(x_arr, feature_names=order)
        pred = model_obj.predict(dmat)
        return float(pred[0])

    pred = model_obj.predict(x_arr)
    return float(pred[0])

# --- LLM prompt
PLACEHOLDER_RE = re.compile(r"(<[^>]+>|\(.*?\)|template|örnek|1-2\s*cümle|etiket\+|placeholder)", re.IGNORECASE)

# ✅ Regex ile 4 satırı sağlam yakala
RE_4LINES = re.compile(
    r"(t\+120_glukoz_mgdl:\s*[^\n]+)\s*\n"
    r"(durum:\s*[^\n]+)\s*\n"
    r"(mesaj:\s*[^\n]+)\s*\n"
    r"(referans:\s*[^\n]+)",
    flags=re.IGNORECASE
)

def extract_4lines_regex(text: str) -> Optional[str]:
    if not text:
        return None
    matches = RE_4LINES.findall(text)
    if not matches:
        return None
    a, b, c, d = matches[-1]
    return f"{a.strip()}\n{b.strip()}\n{c.strip()}\n{d.strip()}\n"

def build_llm_texts(
    food_label_summary: str,
    carbs_g: float, protein_g: float, fat_g: float,
    g_m120: float, g_m60: float, g_0: float,
    xgb_pred_t120: float,
    hyper_thr: float,
    hypo_thr: float
) -> Tuple[str, str, str]:
    """
    return: (system_text, user_text, expected_ref_line)
    """
    expected_ref = f"{food_label_summary} | carbs={carbs_g:.1f}g protein={protein_g:.1f}g fat={fat_g:.1f}g"

    # ✅ DURUMU KODLA NET HESAPLA (model tahmini bozamaz)
    val = float(xgb_pred_t120)
    if val >= float(hyper_thr) or val < float(hypo_thr):
        expected_status = "Uyarı"
    else:
        expected_status = "NORMAL"

    trend = compute_cgm_trend(g_m120, g_m60, g_0)

    system = (
        "Sen Türkçe yazan bir diyabet asistanısın. Tıbbi tavsiye vermezsin; ilaç/insülin dozu önermezsin.\n"
        "KURALLAR (ÇOK ÖNEMLİ):\n"
        "1) Sana verilen t+120_glukoz_mgdl sayısını ASLA değiştirme.\n"
        "2) ÇIKTI SADECE 4 satır olacak; başka hiçbir satır/metin YAZMA.\n"
        "3) Placeholder/şablon/parantez içi açıklama YAZMA.\n"
        "4) 'mesaj' 1-2 GERÇEK ve kısa cümle olacak.\n"
        "5) 'referans' satırı gerçek değerlerle dolu olacak.\n"       
        "6) Yalnızca Türkçe yaz.\n"
        "7) 'durum' satırına, 'Veriler' kısmında sana verilen 'HESAPLANMIŞ_DURUM' bilgisini AYNEN yaz. Kendin yeniden değerlendirme YAPMA.\n"        "8) 'referans' satırında gerçek değerler dışında başka bir şey yazmayacak. \n"
        "9) 'mesaj' satırında gerçek şeyler yazacak. \n"
    )

    user = (
        "Veriler:\n"
        f"Öğün: {food_label_summary}\n"
        f"Makrolar: carbs={carbs_g:.1f}g, protein={protein_g:.1f}g, fat={fat_g:.1f}g\n"
        f"CGM: t-120={g_m120:.1f}, t-60={g_m60:.1f}, t0={g_0:.1f} (mg/dL) | trend: {trend}\n"
        f"Sayı (XGBoost, DEĞİŞTİRME): t+120_glukoz_mgdl={val:.1f}\n"
        f"NORMAL aralık tanımı: {hypo_thr:.0f} <= değer < {hyper_thr:.0f}\n"
        f"UYARI koşulu: değer < {hypo_thr:.0f} VEYA değer >= {hyper_thr:.0f}\n\n"
        f"HESAPLANMIŞ_DURUM: {expected_status}\n"
        "ÇIKTI (SADECE 4 satır, başka hiçbir şey yok):\n"
        f"t+120_glukoz_mgdl: {val:.1f}\n"
        f"durum: {expected_status}\n"
        "mesaj: (1-2 kısa cümle; makro + CGM trendi + risk)\n"
        f"referans: {expected_ref}\n"
    )

    return system, user, expected_ref

def make_messages(system_text: str, user_text: str, tokenizer) -> List[Dict[str, str]]:
    msgs = [{"role": "system", "content": system_text}, {"role": "user", "content": user_text}]
    if not hasattr(tokenizer, "apply_chat_template"):
        return msgs
    try:
        _ = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return msgs
    except Exception:
        merged = system_text.strip() + "\n\n" + user_text.strip()
        return [{"role": "user", "content": merged}]

# ✅ Prompt echo KES: sadece yeni tokenları decode et
def llm_generate(tokenizer, model, messages: List[Dict[str, str]], max_new_tokens: int) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        if len(messages) == 2 and messages[0]["role"] == "system":
            prompt_text = f"{messages[0]['content']}\n\n{messages[1]['content']}\n\nYANIT:\n"
        else:
            prompt_text = f"{messages[0]['content']}\n\nYANIT:\n"

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.05,
        )

    gen_tokens = out[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return decoded.strip()

def sanitize_output_keep_4lines(text: str) -> Optional[str]:
    """
    - regex ile 4 satırı yakala
    - satır içi fazla boşlukları düzelt
    - junk alt satırları asla gösterme
    """
    out4 = extract_4lines_regex(text or "")
    if not out4:
        return None

    lines = [ln.strip() for ln in out4.splitlines() if ln.strip()]
    if len(lines) != 4:
        return None

    # Placeholder kontrolü (çok katı yapmıyoruz, sadece bariz şablonları engelliyoruz)
    if PLACEHOLDER_RE.search(out4):
        return None

    # Mesaj satırında çok saçma tekrar varsa (örn: ngilizngiliz...) yakala
    msg = lines[2].split(":", 1)[-1].strip()
    if re.search(r"(ngiliz){5,}", msg, flags=re.IGNORECASE):
        return None

    return "\n".join(lines) + "\n"

# =========================
# Preload
# =========================
if preload:
    try:
        _ = load_food_model(FOOD_PTH)
        _ = load_rag_index(NUTR_JSON)
        _ = load_xgb_any(XGB_PATH)
        _ = load_llm(BASE_MODEL, ADAPTER_DIR)
        st.success("✅ Food101 + RAG + XGBoost + LLM cache’lendi.")
    except Exception as e:
        st.error(f"Ön yükleme başarısız: {e}")
        st.stop()

# =========================
# MAIN UI
# =========================
st.subheader("1) Yemek fotoğraf(lar)ını yükle")
uploaded_files = st.file_uploader(
    "Birden fazla JPG/PNG seçebilirsin",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

st.subheader("2) CGM geçmişi")
with st.form("cgm_form"):
    st.write("Son 2 saatlik glukoz değerlerini gir (mg/dL):")
    g_m120 = st.number_input("2 saat önce (t-120)", value=120.0, step=1.0)
    g_m60  = st.number_input("1 saat önce (t-60)", value=115.0, step=1.0)
    g_0    = st.number_input("Şu an / öğün anı (t0)", value=100.0, step=1.0)
    submitted = st.form_submit_button("Tahmin üret")

if uploaded_files and submitted:
    # ---- Load everything ----
    try:
        food_model, class_names, tfm = load_food_model(FOOD_PTH)
        docs, doc_embeds, embedder = load_rag_index(NUTR_JSON)
        xgb_kind, xgb_model = load_xgb_any(XGB_PATH)
        tokenizer, llm = load_llm(BASE_MODEL, ADAPTER_DIR)
    except Exception as e:
        st.error(f"Yükleme hatası: {e}")
        st.stop()

    if DEBUG:
        st.info("DEBUG: Modeller yüklendi. Food101 → RAG → XGB → LLM akışı başlıyor.")

    # ---- Per-image Food + RAG selection ----
    st.write("## 🍽️ Fotoğraflar ve yemek seçimi")
    chosen_food_labels: List[str] = []
    food101_topk_all: List[List[Tuple[str, float]]] = []
    rag_retrieved_all: List[List[Dict[str, Any]]] = []

    for idx, uf in enumerate(uploaded_files):
        img = Image.open(uf).convert("RGB")
        st.image(img, caption=f"Fotoğraf #{idx+1}", use_container_width=True)

        preds = food_topk(img, food_model, class_names, tfm, k=topk_food)
        food101_topk_all.append(preds)

        best_label, best_conf = preds[0]
        st.write(f"Food101 en iyi: {best_label} (conf={best_conf:.2f})")

        options = [f"{lab} (conf={conf:.2f})" for lab, conf in preds]
        if best_conf < conf_threshold:
            st.warning(f"Güven {conf_threshold:.2f} altında → listeden seç.")
        pick = st.selectbox(f"Fotoğraf #{idx+1} için yemek seç", options, index=0, key=f"pick_{idx}")
        chosen_label = pick.split(" (conf=")[0].strip()
        chosen_food_labels.append(chosen_label)

        retrieved = rag_topk_lookup(chosen_label, docs, doc_embeds, embedder, k=rag_topk)
        rag_retrieved_all.append(retrieved)

        st.write(f"RAG en iyi: {retrieved[0]['name']} (sim={retrieved[0]['similarity']:.2f})")
        with st.expander(f"RAG Top-{rag_topk} (Fotoğraf #{idx+1})"):
            for r in retrieved:
                st.write(f"- {r['name']} | sim={r['similarity']:.3f} | carbs={r['carbs_g']} protein={r['protein_g']} fat={r['fat_g']}")

        st.divider()

    # ---- Aggregate macros ----
    total_carbs = 0.0
    total_protein = 0.0
    total_fat = 0.0

    rag_best_names = []
    rag_best_sims = []
    rag_found_flags = []

    for retrieved in rag_retrieved_all:
        best = retrieved[0]
        sim = safe_float(best.get("similarity"), 0.0)
        found = bool(sim >= float(min_sim))
        rag_found_flags.append(found)

        rag_best_names.append(str(best.get("name", "unknown")))
        rag_best_sims.append(sim)

        total_carbs += safe_float(best.get("carbs_g"), 0.0) * float(portion_mult)
        total_protein += safe_float(best.get("protein_g"), 0.0) * float(portion_mult)
        total_fat += safe_float(best.get("fat_g"), 0.0) * float(portion_mult)

    st.write("## 📦 Toplam makrolar (RAG)")
    st.write(f"- Karbonhidrat: {total_carbs:.1f} g")
    st.write(f"- Protein: {total_protein:.1f} g")
    st.write(f"- Yağ: {total_fat:.1f} g")

    if DEBUG:
        st.success("DEBUG: RAG tamamlandı → XGBoost'a geçiyorum...")

    # ---- XGB predict ----
    feats = {
        "Carbs": float(total_carbs),
        "Protein": float(total_protein),
        "Fat": float(total_fat),
        "g_m120": float(g_m120),
        "g_m60": float(g_m60),
        "g_0": float(g_0),
    }

    xgb_pred = xgb_predict_t120(xgb_kind, xgb_model, feats)

    st.write("## 📈 XGBoost t+120 tahmini")
    st.write(f"t+120_glukoz_mgdl: {xgb_pred:.1f} mg/dL")

    if DEBUG:
        st.success("DEBUG: XGBoost tamamlandı → LLM yoruma geçiyorum...")

    # ---- LLM generate (no red-stop, clean output) ----
    st.write("## 🤖 LLM kısa yorum (4 satır)")

    food_summary = " + ".join(chosen_food_labels) if chosen_food_labels else "unknown_meal"
    system_text, user_text, expected_ref = build_llm_texts(
        food_label_summary=food_summary,
        carbs_g=total_carbs, protein_g=total_protein, fat_g=total_fat,
        g_m120=float(g_m120), g_m60=float(g_m60), g_0=float(g_0),
        xgb_pred_t120=float(xgb_pred),
        hyper_thr=float(hyper_thr),
        hypo_thr=float(hypo_thr),
    )
    messages = make_messages(system_text, user_text, tokenizer)

    if DEBUG:
        with st.expander("DEBUG: LLM Prompt"):
            if len(messages) == 2 and messages[0]["role"] == "system":
                st.code(system_text)
                st.code(user_text)
            else:
                st.code(messages[0]["content"])

    final_4 = None
    decoded_last = ""

    for attempt in range(2):
        with st.spinner(f"LLM yanıt üretiyor... (deneme {attempt+1}/2)"):
            decoded = llm_generate(tokenizer, llm, messages, max_new_tokens=int(max_new_tokens))
            decoded_last = decoded

        cleaned = sanitize_output_keep_4lines(decoded)
        if cleaned:
            final_4 = cleaned
            break

        if attempt == 0:
            user_text2 = user_text + "\n\nUYARI: SADECE 4 satır yaz. Başka hiçbir metin yazma. Yalnızca Türkçe."
            messages = make_messages(system_text, user_text2, tokenizer)

    if final_4 is None:
        # ✅ KIRMIZI STOP YOK
        st.warning("LLM çıktısı tam temiz formatta gelmedi; en yakın temiz bölüm gösteriliyor.")
        rough = extract_4lines_regex(decoded_last or "")
        if rough:
            st.code(rough)
        else:
            # en kötü ihtimal ilk 600 karakter göster (debug amaçlı)
            st.code((decoded_last[:600] if decoded_last else ""))
    else:
        st.code(final_4)

    # ---- Logging ----
    if ENABLE_LOG:
        ensure_dir(LOG_DIR)
        pred_path = os.path.join(LOG_DIR, f"pred_{RUN_TAG}.jsonl")
        rag_path = os.path.join(LOG_DIR, f"rag_log_{RUN_TAG}.csv")

        pred_record = {
            "ts": now_iso(),
            "run_tag": RUN_TAG,
            "n_images": len(uploaded_files),
            "chosen_food_labels": chosen_food_labels,
            "rag_best_names": rag_best_names,
            "rag_best_sims": rag_best_sims,
            "rag_found_flags": rag_found_flags,
            "macros": {"carbs_g": total_carbs, "protein_g": total_protein, "fat_g": total_fat, "portion_mult": portion_mult},
            "cgm": {"g_m120": float(g_m120), "g_m60": float(g_m60), "g_0": float(g_0)},
            "xgb_pred_t120": float(xgb_pred),
            "thresholds": {"hyper": float(hyper_thr), "hypo": float(hypo_thr)},
            "llm_output_4lines": final_4,
            "llm_raw": decoded_last if DEBUG else "",
            "llm_messages": messages,
            "xgb_model_path": XGB_PATH,
            "base_model": BASE_MODEL,
            "adapter_dir": ADAPTER_DIR,
        }
        append_jsonl(pred_path, pred_record)

        rag_header = [
            "ts", "run_tag", "image_idx",
            "food101_best", "food101_best_conf",
            "chosen_label",
            "rag_best_name", "rag_best_sim", "rag_found",
            "retrieved_topk_names", "retrieved_topk_sims",
        ]

        for i in range(len(uploaded_files)):
            preds = food101_topk_all[i]
            retrieved = rag_retrieved_all[i]
            sim0 = safe_float(retrieved[0].get("similarity"), 0.0)
            row = {
                "ts": now_iso(),
                "run_tag": RUN_TAG,
                "image_idx": i + 1,
                "food101_best": preds[0][0],
                "food101_best_conf": safe_float(preds[0][1], 0.0),
                "chosen_label": chosen_food_labels[i],
                "rag_best_name": retrieved[0]["name"],
                "rag_best_sim": sim0,
                "rag_found": int(bool(sim0 >= float(min_sim))),
                "retrieved_topk_names": json.dumps([r["name"] for r in retrieved], ensure_ascii=False),
                "retrieved_topk_sims": json.dumps([safe_float(r.get("similarity"), 0.0) for r in retrieved], ensure_ascii=False),
            }
            append_csv_row(rag_path, rag_header, row)

        st.success(f"✅ Log yazıldı:\n- {pred_path}\n- {rag_path}")

else:
    st.info("Fotoğraf(lar) yükleyip CGM formunu doldurduktan sonra Tahmin üret ile devam et.")