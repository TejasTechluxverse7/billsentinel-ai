"""
BILLSENTINEL AI — Streamlit Frontend
Run: streamlit run app.py
"""

# ── stdlib / compat ───────────────────────────────────────────────────────────
import sys, io, os, re, json, random, warnings, textwrap
from datetime import datetime, timedelta
from collections import Counter

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import streamlit as st
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# optional OCR / PDF deps — graceful fallback
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    from pdf2image import convert_from_bytes
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BillSentinel AI",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── hero banner ── */
.hero {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-radius: 20px;
    padding: 2.8rem 2.5rem 2.2rem;
    margin-bottom: 1.8rem;
    text-align: center;
    box-shadow: 0 8px 40px rgba(0,0,0,.45);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute; top: -60px; right: -60px;
    width: 260px; height: 260px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(130,80,255,.25) 0%, transparent 70%);
}
.hero-title {
    font-size: 2.8rem; font-weight: 800; color: #fff; margin: 0;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 1.05rem; color: rgba(255,255,255,.65); margin-top: .5rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(130,80,255,.25);
    border: 1px solid rgba(130,80,255,.5);
    border-radius: 30px;
    padding: .25rem .9rem;
    font-size: .78rem; font-weight: 600;
    color: #c4a7ff;
    margin-bottom: 1rem;
    letter-spacing: .5px;
    text-transform: uppercase;
}

/* ── cards ── */
.card {
    background: #1a1a2e;
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,.3);
}
.card-title {
    font-size: 1rem; font-weight: 700;
    color: #a78bfa; margin-bottom: .8rem;
    display: flex; align-items: center; gap: .5rem;
}

/* ── error pills ── */
.err-pill {
    display: inline-block;
    border-radius: 20px;
    padding: .22rem .8rem;
    font-size: .78rem; font-weight: 600;
    margin: .18rem .3rem .18rem 0;
}
.pill-rule  { background: rgba(239,68,68,.18);  color: #fca5a5; border: 1px solid rgba(239,68,68,.4); }
.pill-ml    { background: rgba(251,146,60,.18); color: #fdba74; border: 1px solid rgba(251,146,60,.4); }
.pill-ok    { background: rgba(52,211,153,.18); color: #6ee7b7; border: 1px solid rgba(52,211,153,.4); }

/* ── score ring ── */
.score-ring {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    width: 130px; height: 130px;
    border-radius: 50%;
    font-size: 2.4rem; font-weight: 800; color: #fff;
    margin: 0 auto .6rem;
    box-shadow: 0 0 28px rgba(130,80,255,.4);
}
.score-label { font-size: .8rem; color: rgba(255,255,255,.55); text-align: center; }

/* ── info banner ── */
.info-banner {
    background: rgba(130,80,255,.12);
    border-left: 3px solid #8250ff;
    border-radius: 8px;
    padding: .7rem 1rem;
    font-size: .88rem; color: rgba(255,255,255,.75);
    margin-bottom: .8rem;
}
.warn-banner {
    background: rgba(239,68,68,.12);
    border-left: 3px solid #ef4444;
    border-radius: 8px;
    padding: .7rem 1rem;
    font-size: .88rem; color: rgba(255,255,255,.75);
    margin-bottom: .8rem;
}
.good-banner {
    background: rgba(52,211,153,.12);
    border-left: 3px solid #34d399;
    border-radius: 8px;
    padding: .7rem 1rem;
    font-size: .88rem; color: rgba(255,255,255,.75);
    margin-bottom: .8rem;
}

/* ── line item table ── */
.li-table { width: 100%; border-collapse: collapse; font-size: .85rem; }
.li-table th {
    background: rgba(130,80,255,.18); color: #c4b5fd;
    padding: .5rem .8rem; text-align: left; font-weight: 600;
}
.li-table td { padding: .45rem .8rem; border-bottom: 1px solid rgba(255,255,255,.06); color: rgba(255,255,255,.8); }
.li-table tr:hover td { background: rgba(255,255,255,.03); }

/* ── explanation box ── */
.exp-box {
    background: rgba(255,255,255,.04);
    border-radius: 10px;
    padding: .8rem 1rem;
    margin-bottom: .5rem;
    border-left: 3px solid #8250ff;
}
.exp-type { font-size: .75rem; font-weight: 700; color: #a78bfa; text-transform: uppercase; letter-spacing: .5px; }
.exp-text { font-size: .88rem; color: rgba(255,255,255,.78); margin-top: .3rem; }

/* ── sidebar ── */
section[data-testid="stSidebar"] {
    background: #0f0c29 !important;
    border-right: 1px solid rgba(255,255,255,.06);
}
.sidebar-logo { font-size: 1.3rem; font-weight: 700; color: #a78bfa; padding: .5rem 0 1rem; }

/* ── metric override ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,.04);
    border-radius: 12px; padding: .8rem 1rem;
    border: 1px solid rgba(255,255,255,.07);
}
[data-testid="stMetricLabel"]  { color: rgba(255,255,255,.55) !important; font-size: .8rem !important; }
[data-testid="stMetricValue"]  { color: #c4b5fd !important; font-size: 1.6rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"]  { font-size: .78rem !important; }

div[data-testid="stButton"] button {
    background: linear-gradient(135deg,#7c3aed,#4f46e5);
    color: #fff; border: none; border-radius: 10px;
    padding: .55rem 1.8rem; font-weight: 600; font-size: .97rem;
    transition: opacity .2s;
    width: 100%;
}
div[data-testid="stButton"] button:hover { opacity: .85; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ████  BILLSENTINEL AI CORE ENGINE  (embedded from billsentinel_ai.py)  ████
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

SERVICE_CATALOGUE = {
    "ICU": [
        {"desc": "ICU Daily Charges",           "code": "ICU001", "cost_range": (5000,  20000)},
        {"desc": "Ventilator Support",           "code": "ICU002", "cost_range": (8000,  20000)},
        {"desc": "Cardiac Monitoring",           "code": "ICU003", "cost_range": (5000,  15000)},
    ],
    "Surgery": [
        {"desc": "Knee Replacement",             "code": "SUR001", "cost_range": (40000, 100000)},
        {"desc": "Appendectomy",                 "code": "SUR002", "cost_range": (20000,  60000)},
        {"desc": "Laparoscopic Cholecystectomy", "code": "SUR003", "cost_range": (25000,  70000)},
        {"desc": "Hernia Repair",                "code": "SUR004", "cost_range": (20000,  55000)},
    ],
    "X-ray": [
        {"desc": "Chest X-Ray",                 "code": "XRY001", "cost_range": (500,   3000)},
        {"desc": "Abdominal X-Ray",             "code": "XRY002", "cost_range": (600,   2500)},
        {"desc": "Spine X-Ray",                 "code": "XRY003", "cost_range": (700,   3000)},
        {"desc": "MRI Brain",                   "code": "XRY004", "cost_range": (3000, 12000)},
    ],
    "Consultation": [
        {"desc": "General Consultation",        "code": "CON001", "cost_range": (200,   1000)},
        {"desc": "Specialist Consultation",     "code": "CON002", "cost_range": (500,   1500)},
        {"desc": "Follow-up Visit",             "code": "CON003", "cost_range": (200,    800)},
    ],
    "Medicine": [
        {"desc": "Antibiotic Course",           "code": "MED001", "cost_range": (100,   2000)},
        {"desc": "Pain Management Drugs",       "code": "MED002", "cost_range": (200,   3000)},
        {"desc": "IV Fluids",                  "code": "MED003", "cost_range": (100,   1500)},
        {"desc": "Anaesthesia Drugs",          "code": "MED004", "cost_range": (500,   5000)},
    ],
}

CATEGORY_AVG = {
    cat: float(np.mean([((s["cost_range"][0] + s["cost_range"][1]) / 2)
                        for s in svcs]))
    for cat, svcs in SERVICE_CATALOGUE.items()
}

ALL_SERVICES = [
    {**svc, "category": cat}
    for cat, svcs in SERVICE_CATALOGUE.items()
    for svc in svcs
]

DIAGNOSIS_CODES = ["E11.9","I10","J18.9","M54.5","K21.0","N39.0","F41.1",
                   "Z87.891","I25.10","J44.1","N18.3","M16.9"]
PROCEDURE_CODES = ["99213","99214","99232","71046","93000","80053","85025",
                   "27447","70553","99244","99246","93010"]
ERROR_TYPES     = ["duplicate_charge","post_discharge_charge","mismatch_treatment",
                   "hidden_fee","overpricing"]
ML_ANOMALY_THRESHOLD = 0.55

# ── dataset helpers ───────────────────────────────────────────────────────────
def _jitter(cost):
    return round(cost * (1.0 + random.uniform(-0.20, 0.20)), 2)

def _rand_date(start, end):
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, max(delta, 0)))

def _make_item(adm, dis, svc=None, date_override=None, category=None):
    if svc is None:
        svc = random.choice(ALL_SERVICES)
    cat  = svc.get("category", category or "General")
    cost = _jitter(random.uniform(*svc["cost_range"]))
    date = date_override if date_override else _rand_date(adm, dis)
    return {
        "desc":     svc["desc"], "code": svc["code"], "category": cat,
        "date":     date.strftime("%Y-%m-%d"),
        "cost":     max(round(cost, 2), 1.0),
    }

def _build_clean_bill(adm, dis):
    items, n_days = [], max((dis - adm).days, 1)
    for _ in range(max(1, n_days // 3)):
        items.append(_make_item(adm, dis, random.choice(SERVICE_CATALOGUE["Consultation"]), category="Consultation"))
    for _ in range(random.randint(2, 4)):
        items.append(_make_item(adm, dis, random.choice(SERVICE_CATALOGUE["Medicine"]), category="Medicine"))
    for _ in range(random.randint(1, 2)):
        items.append(_make_item(adm, dis, random.choice(SERVICE_CATALOGUE["X-ray"]), category="X-ray"))
    if random.random() < 0.30:
        items.append(_make_item(adm, dis, random.choice(SERVICE_CATALOGUE["Surgery"]), category="Surgery"))
        if random.random() < 0.5:
            for _ in range(random.randint(1, 3)):
                items.append(_make_item(adm, dis, random.choice(SERVICE_CATALOGUE["ICU"]), category="ICU"))
    random.shuffle(items)
    return items[:8] if len(items) > 8 else (
        items if len(items) >= 3 else items + [_make_item(adm, dis) for _ in range(3 - len(items))]
    )

def _inject_errors(items, adm, dis, n_errors):
    errors = []
    pool = random.sample(
        (ERROR_TYPES * ((n_errors // len(ERROR_TYPES)) + 2))[:n_errors],
        min(n_errors, n_errors)
    )
    for etype in pool:
        if etype == "duplicate_charge":
            src = random.choice(items); dup = dict(src)
            items.append(dup)
            errors.append({"type": "duplicate_charge", "line_index": len(items)-1})
        elif etype == "post_discharge_charge":
            post = dis + timedelta(days=random.randint(1, 7))
            items.append(_make_item(adm, dis, random.choice(ALL_SERVICES), date_override=post))
            errors.append({"type": "post_discharge_charge", "line_index": len(items)-1})
        elif etype == "mismatch_treatment":
            idx = random.randint(0, len(items)-1)
            ws  = random.choice([s for s in ALL_SERVICES if s["code"] != items[idx]["code"]])
            items[idx] = dict(items[idx]); items[idx]["code"] = ws["code"]
            errors.append({"type": "mismatch_treatment", "line_index": idx})
        elif etype == "hidden_fee":
            idx  = random.randint(0, len(items)-1)
            orig = items[idx]["cost"]
            for sp in sorted([round(orig*random.uniform(0.05,0.30),2) for _ in range(random.randint(1,2))]):
                items[idx]["cost"] = max(round(items[idx]["cost"]-sp, 2), 1.0)
                items.append({"desc":"Administrative Processing Fee","code":"ADM001",
                              "category":"Admin","date":items[idx]["date"],"cost":sp})
                errors.append({"type": "hidden_fee", "line_index": len(items)-1})
        elif etype == "overpricing":
            idx = random.randint(0, len(items)-1)
            items[idx] = dict(items[idx])
            items[idx]["cost"] = round(items[idx]["cost"]*random.uniform(3.0,8.0), 2)
            errors.append({"type": "overpricing", "line_index": idx})
    return errors

def generate_dataset(n=2000):
    dataset = []
    labels  = ["clean"]*int(n*.4) + ["faulty"]*(n-int(n*.4))
    random.shuffle(labels)
    for i, label in enumerate(labels):
        adm = datetime(2022,1,1) + timedelta(days=random.randint(0,700))
        dis = adm + timedelta(days=random.randint(1,14))
        items  = _build_clean_bill(adm, dis)
        errors = _inject_errors(items, adm, dis, random.randint(1,3)) if label=="faulty" else []
        dataset.append({
            "bill_id":        f"BILL-{i+1:05d}",
            "admission_date": adm.strftime("%Y-%m-%d"),
            "discharge_date": dis.strftime("%Y-%m-%d"),
            "diagnosis_codes": random.sample(DIAGNOSIS_CODES, k=random.randint(1,3)),
            "procedure_codes": random.sample(PROCEDURE_CODES, k=random.randint(1,4)),
            "line_items":      items,
            "total_cost":      round(sum(it["cost"] for it in items), 2),
            "errors":          errors,
        })
    return dataset

# ── feature engineering ───────────────────────────────────────────────────────
def _count_duplicates(items):
    seen, count = [], 0
    for it in items:
        k = (it["desc"], it.get("date"), it["cost"])
        if k in seen: count += 1
        else: seen.append(k)
    return count

def _post_discharge_count(items, discharge_date):
    dis = datetime.strptime(discharge_date, "%Y-%m-%d")
    return sum(1 for it in items if datetime.strptime(it["date"], "%Y-%m-%d") > dis)

def _same_day_repeat(items):
    dates = [it["date"] for it in items]
    return sum(1 for c in Counter(dates).values() if c > 1)

def extract_features(dataset):
    X, y = [], []
    for bill in dataset:
        items  = bill["line_items"]
        costs  = np.array([it["cost"] for it in items], dtype=np.float64)
        n      = len(items)
        avg    = float(np.mean(costs))
        mx     = float(np.max(costs))
        nd     = _count_duplicates(items)
        X.append([
            n, avg, mx,
            float(np.std(costs)) if n > 1 else 0.0,
            nd/n if n else 0.0,
            _post_discharge_count(items, bill["discharge_date"]),
            mx/avg if avg > 0 else 1.0,
            _same_day_repeat(items),
            len(set(it["desc"] for it in items)),
        ])
        y.append(1 if len(bill["errors"]) > 0 else 0)
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int32)

# ── model training ────────────────────────────────────────────────────────────
_iso_forest = None
_rf_clf     = None

def train_models(dataset):
    global _iso_forest, _rf_clf
    X, y = extract_features(dataset)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    _iso_forest = IsolationForest(n_estimators=300, contamination=0.5, random_state=SEED)
    _iso_forest.fit(Xtr)
    _rf_clf = RandomForestClassifier(n_estimators=300, random_state=SEED,
                                     class_weight="balanced", min_samples_leaf=2)
    _rf_clf.fit(Xtr, ytr)
    iso_p = (_iso_forest.predict(Xte)==-1).astype(int)
    rf_p  = _rf_clf.predict(Xte)
    return {
        "IF":  {"acc": accuracy_score(yte,iso_p), "prec": precision_score(yte,iso_p,zero_division=0), "rec": recall_score(yte,iso_p,zero_division=0)},
        "RF":  {"acc": accuracy_score(yte,rf_p),  "prec": precision_score(yte,rf_p, zero_division=0), "rec": recall_score(yte,rf_p, zero_division=0)},
    }

# ── rule engine ───────────────────────────────────────────────────────────────
def _cat_avg(cat): return CATEGORY_AVG.get(cat, 5000.0)

def rule_engine(bill):
    issues, items = [], bill["line_items"]
    dis     = datetime.strptime(bill["discharge_date"], "%Y-%m-%d")
    ddmap   = {}; dcmap  = {}
    for idx, it in enumerate(items):
        ddmap.setdefault((it["desc"],it["date"]),[]).append(idx)
        dcmap.setdefault(it["desc"],[]).append(it["cost"])
    reported = set()

    for idx, it in enumerate(items):
        key = (it["desc"],it["date"])
        if len(ddmap[key])>1 and idx!=ddmap[key][0]:
            tag=("duplicate_charge",idx)
            if tag not in reported:
                reported.add(tag)
                issues.append({"type":"duplicate_charge","line_index":idx,
                               "detail":f"'{it['desc']}' billed multiple times on {it['date']}.","cost":it["cost"]})
        try:
            if datetime.strptime(it["date"],"%Y-%m-%d")>dis:
                tag=("post_discharge_charge",idx)
                if tag not in reported:
                    reported.add(tag)
                    issues.append({"type":"post_discharge_charge","line_index":idx,
                                   "detail":f"Charge on {it['date']} after discharge {bill['discharge_date']}.","cost":it["cost"]})
        except: pass
        ca = _cat_avg(it.get("category",""))
        if it["cost"]>2.0*ca:
            tag=("overpricing",idx)
            if tag not in reported:
                reported.add(tag)
                issues.append({"type":"overpricing","line_index":idx,
                               "detail":f"'{it['desc']}' costs ₹{it['cost']:,.2f} vs avg ₹{ca:,.2f}.","cost":it["cost"]-ca})

    for desc,costs in dcmap.items():
        if len(costs)>1:
            tot,mx = sum(costs),max(costs)
            if tot>1.5*mx:
                for idx2,it2 in enumerate(items):
                    if it2["desc"]==desc and ("hidden_fee",idx2) not in reported:
                        reported.add(("hidden_fee",idx2))
                        issues.append({"type":"hidden_fee","line_index":idx2,
                                       "detail":f"'{desc}' split across {len(costs)} entries; total ₹{tot:,.2f}.","cost":round(tot-mx,2)})

    date_items={}
    for idx,it in enumerate(items): date_items.setdefault(it["date"],[]).append((idx,it))
    for ds,grp in date_items.items():
        for desc in set(g[1]["desc"] for g in grp):
            occur=[(i2,i2t) for i2,i2t in grp if i2t["desc"]==desc]
            if len(occur)>1:
                for idx2,it2 in occur[1:]:
                    tag=("same_day_repeat",idx2)
                    if tag not in reported:
                        reported.add(tag)
                        issues.append({"type":"duplicate_charge","line_index":idx2,
                                       "detail":f"Same-day repeat: '{desc}' on {ds}.","cost":it2["cost"]})
    return issues

# ── explanation generator ─────────────────────────────────────────────────────
_TEMPLATES = {
    "duplicate_charge":      "The same service was billed more than once, inflating the total. Request removal of the duplicate entry.",
    "post_discharge_charge": "A charge was added after the patient's discharge date — this is a likely billing error or fraud. Dispute with the hospital.",
    "mismatch_treatment":    "The procedure code does not match the described service, suggesting upcoding or a data entry error. Request itemized clarification.",
    "hidden_fee":            "A service cost appears split into smaller unlabeled entries, obscuring the true amount. Request a fully itemized receipt.",
    "overpricing":           "This item is priced significantly above the typical rate for its category. Compare with standard schedules and negotiate.",
    "anomaly_detected":      "The AI model flagged this bill as statistically unusual. Manual review by a certified medical auditor is recommended.",
}

def generate_explanation(error_type, bill_context=None):
    return _TEMPLATES.get(error_type, "An unclassified billing irregularity was detected. Seek expert review.")

# ── savings & score ───────────────────────────────────────────────────────────
def estimate_savings(errors, bill):
    total = sum(e["cost"] for e in errors if e.get("source")=="rule" and "cost" in e)
    ml_e  = [e for e in errors if e.get("source")=="ml"]
    if ml_e: total += len(ml_e)*bill["total_cost"]*0.05
    return round(total, 2)

def calculate_score(errors): return max(0, 100-len(errors)*10)

# ── analyze_bill ──────────────────────────────────────────────────────────────
def analyze_bill(bill):
    if _iso_forest is None or _rf_clf is None:
        raise RuntimeError("Models not trained.")

    rule_errors = [
        {"type":e["type"],"line_index":e["line_index"],"detail":e.get("detail",""),
         "cost":e.get("cost",0.0),"confidence":1.0,"source":"rule"}
        for e in rule_engine(bill)
    ]

    feats,_ = extract_features([bill])
    iso_flag = (_iso_forest.predict(feats)[0]==-1)
    rf_proba = float(_rf_clf.predict_proba(feats)[0][1])
    ml_errors = []
    if rf_proba >= ML_ANOMALY_THRESHOLD:
        ml_errors.append({"type":"anomaly_detected","line_index":-1,
                          "detail":f"IF={'anomaly' if iso_flag else 'normal'}, RF_p={rf_proba:.3f}",
                          "cost":0.0,"confidence":round(rf_proba,4),"source":"ml"})

    seen, merged = set(), []
    for e in rule_errors+ml_errors:
        k=(e["type"],e["line_index"])
        if k not in seen: seen.add(k); merged.append(e)

    explanations = [{"type":e["type"],"explanation":generate_explanation(e["type"],bill)} for e in merged]
    conf = round(float(np.mean([e["confidence"] for e in merged])),4) if merged else 0.0

    return {
        "errors":            merged,
        "explanations":      explanations,
        "total_issues":      len(merged),
        "confidence_score":  conf,
        "estimated_savings": estimate_savings(merged, bill),
        "bill_score":        calculate_score(merged),
    }

# ─────────────────────────────────────────────────────────────────────────────
# ████  OCR / DOCUMENT PARSER  ████
# ─────────────────────────────────────────────────────────────────────────────
_SERVICE_KEYWORDS = {
    "consultation":  ("General Consultation",  "CON001", "Consultation"),
    "visit":         ("General Consultation",  "CON001", "Consultation"),
    "specialist":    ("Specialist Consultation","CON002", "Consultation"),
    "follow":        ("Follow-up Visit",       "CON003", "Consultation"),
    "x-ray":         ("Chest X-Ray",           "XRY001", "X-ray"),
    "xray":          ("Chest X-Ray",           "XRY001", "X-ray"),
    "mri":           ("MRI Brain",             "XRY004", "X-ray"),
    "ct":            ("Abdominal X-Ray",       "XRY002", "X-ray"),
    "ultrasound":    ("Abdominal X-Ray",       "XRY002", "X-ray"),
    "icu":           ("ICU Daily Charges",     "ICU001", "ICU"),
    "ventilator":    ("Ventilator Support",    "ICU002", "ICU"),
    "cardiac":       ("Cardiac Monitoring",    "ICU003", "ICU"),
    "surgery":       ("Appendectomy",          "SUR002", "Surgery"),
    "operation":     ("Appendectomy",          "SUR002", "Surgery"),
    "knee":          ("Knee Replacement",      "SUR001", "Surgery"),
    "hernia":        ("Hernia Repair",         "SUR004", "Surgery"),
    "antibiotic":    ("Antibiotic Course",     "MED001", "Medicine"),
    "medicine":      ("Pain Management Drugs", "MED002", "Medicine"),
    "drug":          ("Pain Management Drugs", "MED002", "Medicine"),
    "iv":            ("IV Fluids",            "MED003", "Medicine"),
    "fluid":         ("IV Fluids",            "MED003", "Medicine"),
    "anaesthesia":   ("Anaesthesia Drugs",    "MED004", "Medicine"),
    "anesthesia":    ("Anaesthesia Drugs",    "MED004", "Medicine"),
}

_DATE_RX   = re.compile(r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{4}[\/\-]\d{2}[\/\-]\d{2})\b")
_COST_RX   = re.compile(r"(?:rs\.?|inr|₹|\$)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)", re.IGNORECASE)
_LABEL_RX  = re.compile(r"([a-zA-Z][a-zA-Z\s\-/]{2,40})\s*[:\-]?\s*(?:rs\.?|inr|₹|\$)?\s*(\d[\d,\.]+)", re.IGNORECASE)


def _parse_date(s):
    for fmt in ("%d/%m/%Y","%d-%m-%Y","%d.%m.%Y","%Y/%m/%d","%Y-%m-%d",
                "%d/%m/%y","%d-%m-%y","%m/%d/%Y","%m-%d-%Y"):
        try: return datetime.strptime(s, fmt)
        except: pass
    return None


def _extract_text_from_image(img_bytes):
    if not OCR_AVAILABLE:
        return ""
    try:
        img  = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        text = pytesseract.image_to_string(img, config="--psm 6")
        return text
    except Exception as e:
        return f"[OCR error: {e}]"


def _extract_text_from_pdf(pdf_bytes):
    if not PDF_AVAILABLE:
        return ""
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=200)
        texts = []
        for pg in pages:
            if OCR_AVAILABLE:
                texts.append(pytesseract.image_to_string(pg, config="--psm 6"))
            else:
                texts.append("")
        return "\n".join(texts)
    except Exception as e:
        return f"[PDF error: {e}]"


def _build_bill_from_text(raw_text: str, bill_id: str = "USER_BILL") -> dict:
    """Convert raw OCR/text into structured bill dict."""
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

    # ── date extraction ──
    all_dates = []
    for line in lines:
        for m in _DATE_RX.finditer(line):
            d = _parse_date(m.group(0))
            if d:
                all_dates.append(d)

    if len(all_dates) >= 2:
        adm = min(all_dates); dis = max(all_dates)
    elif len(all_dates) == 1:
        adm = all_dates[0]; dis = adm + timedelta(days=3)
    else:
        adm = datetime.today() - timedelta(days=5)
        dis = datetime.today()

    if (dis - adm).days < 0:
        adm, dis = dis, adm
    if adm == dis:
        dis = adm + timedelta(days=1)

    # ── line item extraction ──
    items = []
    for line in lines:
        m = _LABEL_RX.search(line)
        if not m:
            continue
        raw_label = m.group(1).strip()
        raw_cost  = m.group(2).replace(",", "")
        try:
            cost = float(raw_cost)
        except:
            continue
        if cost <= 0:
            continue

        # map keyword → canonical service
        label_lower = raw_label.lower()
        desc, code, cat = raw_label, "GEN001", "General"
        for kw, (d, c, ca) in _SERVICE_KEYWORDS.items():
            if kw in label_lower:
                desc, code, cat = d, c, ca
                break

        # pick a date for this item (nearest date before in text, else use adm)
        item_date = adm
        for line2 in lines:
            dm = _DATE_RX.search(line2)
            if dm:
                pd = _parse_date(dm.group(0))
                if pd and adm <= pd <= dis + timedelta(days=7):
                    item_date = pd

        items.append({
            "desc":     desc,
            "code":     code,
            "category": cat,
            "date":     item_date.strftime("%Y-%m-%d"),
            "cost":     round(cost, 2),
        })

    # ── fallback: at least 3 items ──
    if len(items) < 3:
        base  = [_make_item(adm, dis) for _ in range(3 - len(items))]
        items = base + items

    return {
        "bill_id":        bill_id,
        "admission_date": adm.strftime("%Y-%m-%d"),
        "discharge_date": dis.strftime("%Y-%m-%d"),
        "diagnosis_codes": random.sample(DIAGNOSIS_CODES, 1),
        "procedure_codes": random.sample(PROCEDURE_CODES, 1),
        "line_items":      items,
        "total_cost":      round(sum(it["cost"] for it in items), 2),
        "errors":          [],
    }


def parse_bill(uploaded_file) -> tuple:
    """Returns (bill_dict, raw_text)."""
    fname = uploaded_file.name.lower()
    raw   = uploaded_file.read()

    if fname.endswith(".txt"):
        raw_text = raw.decode("utf-8", errors="replace")
    elif fname.endswith(".pdf"):
        raw_text = _extract_text_from_pdf(raw)
    else:  # jpg / png / jpeg / webp
        raw_text = _extract_text_from_image(raw)

    bill = _build_bill_from_text(raw_text, bill_id="USER_BILL")
    return bill, raw_text


# ─────────────────────────────────────────────────────────────────────────────
# ████  MODEL BOOTSTRAP (cached across reruns)  ████
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _bootstrap_models():
    ds     = generate_dataset(n=2000)
    metrics = train_models(ds)
    return metrics

# ─────────────────────────────────────────────────────────────────────────────
# ████  UI HELPERS  ████
# ─────────────────────────────────────────────────────────────────────────────
ERROR_ICONS = {
    "duplicate_charge":      "🔁",
    "post_discharge_charge": "📅",
    "mismatch_treatment":    "⚠️",
    "hidden_fee":            "🕵️",
    "overpricing":           "💰",
    "anomaly_detected":      "🤖",
}

def _score_color(score):
    if score >= 80: return "#34d399"
    if score >= 50: return "#fbbf24"
    return "#ef4444"

def _score_label(score):
    if score >= 80: return "✅ Low Risk"
    if score >= 50: return "⚠️ Moderate Risk"
    return "🔴 High Risk"

def _render_score_ring(score):
    color = _score_color(score)
    st.markdown(f"""
    <div class="score-ring" style="background:conic-gradient({color} {score*3.6}deg, rgba(255,255,255,.08) 0deg);">
        <span>{score}</span>
    </div>
    <div class="score-label">{_score_label(score)}</div>
    """, unsafe_allow_html=True)

def _render_line_items(items):
    rows = "".join(
        f"<tr><td>{i+1}</td><td>{it['desc']}</td>"
        f"<td><span style='background:rgba(130,80,255,.15);color:#c4b5fd;border-radius:6px;padding:.1rem .5rem;font-size:.75rem'>{it.get('category','—')}</span></td>"
        f"<td>{it['date']}</td>"
        f"<td style='font-weight:600;color:#a7f3d0'>₹{it['cost']:,.2f}</td></tr>"
        for i, it in enumerate(items)
    )
    st.markdown(f"""
    <table class="li-table">
      <thead><tr><th>#</th><th>Service</th><th>Category</th><th>Date</th><th>Cost</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>""", unsafe_allow_html=True)

def _render_errors(errors):
    if not errors:
        st.markdown('<div class="good-banner">✅ No errors detected in this bill.</div>', unsafe_allow_html=True)
        return
    for e in errors:
        icon   = ERROR_ICONS.get(e["type"], "⚠️")
        src    = e.get("source","rule")
        pill   = "pill-ml" if src=="ml" else "pill-rule"
        src_lbl = "ML" if src=="ml" else "Rule"
        conf   = e["confidence"]
        conf_c = "#34d399" if conf>=0.9 else "#fbbf24" if conf>=0.6 else "#f87171"

        st.markdown(f"""
        <div class="card" style="border-left:3px solid {'#a78bfa' if src=='rule' else '#fb923c'}">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:.5rem">
            <span style="font-weight:700;color:#e2e8f0">{icon} {e['type'].replace('_',' ').title()}</span>
            <span>
              <span class="err-pill {pill}">{src_lbl}</span>
              <span style="color:{conf_c};font-size:.82rem;font-weight:600">conf: {conf:.0%}</span>
            </span>
          </div>
          <div style="font-size:.84rem;color:rgba(255,255,255,.6)">
            Line #{e['line_index']} &nbsp;·&nbsp; {e.get('detail','—')}
          </div>
        </div>
        """, unsafe_allow_html=True)

def _render_explanations(explanations):
    seen = set()
    for exp in explanations:
        if exp["type"] in seen: continue
        seen.add(exp["type"])
        icon = ERROR_ICONS.get(exp["type"], "💡")
        st.markdown(f"""
        <div class="exp-box">
          <div class="exp-type">{icon} {exp['type'].replace('_',' ').title()}</div>
          <div class="exp-text">{exp['explanation']}</div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ████  SIDEBAR  ████
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🧾 BillSentinel AI</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("#### ℹ️ About")
    st.markdown("""
    **BillSentinel AI** detects errors in medical bills using:
    - 🤖 ML models (IsolationForest + RandomForest)
    - ⚖️ Rule-based auditor
    - 🧾 OCR document parser
    """)
    st.markdown("---")
    st.markdown("#### 📂 Supported Formats")
    st.markdown("PDF · JPG · PNG · TXT")
    st.markdown("---")
    st.markdown("#### 🎲 Don't have a bill?")
    if st.button("Use Sample Bill", key="sample_btn"):
        st.session_state["use_sample"] = True
        st.session_state["uploaded_bill"] = None
    st.markdown("---")
    st.caption("v2.0 · Powered by sklearn + Streamlit")

# ─────────────────────────────────────────────────────────────────────────────
# ████  MAIN PAGE  ████
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">AI-Powered Billing Audit</div>
  <div class="hero-title">🧾 BillSentinel AI</div>
  <div class="hero-sub">Upload your hospital bill · Detect errors instantly · Estimate savings</div>
</div>
""", unsafe_allow_html=True)

# ── Model bootstrap status ────────────────────────────────────────────────────
with st.spinner("🔧 Initializing AI models (first load ~30s)…"):
    model_metrics = _bootstrap_models()

st.markdown("""<div class="info-banner">
  ✅ AI models ready &nbsp;|&nbsp;
  RandomForest accuracy: <strong>{:.1%}</strong> &nbsp;|&nbsp;
  IsolationForest accuracy: <strong>{:.1%}</strong>
</div>""".format(model_metrics["RF"]["acc"], model_metrics["IF"]["acc"]),
unsafe_allow_html=True)

# ── Upload area ───────────────────────────────────────────────────────────────
st.markdown("### 📤 Upload Medical Bill")
uploaded = st.file_uploader(
    "Drag and drop or click to browse",
    type=["pdf", "jpg", "jpeg", "png", "txt"],
    label_visibility="collapsed",
)

# Sync uploaded file into session state
if uploaded is not None:
    st.session_state["use_sample"] = False
    st.session_state["uploaded_bill"] = uploaded

use_sample  = st.session_state.get("use_sample", False)
bill_ready  = use_sample or (uploaded is not None)

analyze_clicked = st.button("🔍 Analyze Bill", key="analyze_btn", disabled=not bill_ready)

# ─────────────────────────────────────────────────────────────────────────────
# ████  ANALYSIS PIPELINE  ████
# ─────────────────────────────────────────────────────────────────────────────
if analyze_clicked or use_sample:
    # ── get bill ──────────────────────────────────────────────────────────────
    raw_text_display = None
    if use_sample and (uploaded is None):
        # Generate a deterministic faulty sample from internal dataset
        ds_sample  = generate_dataset(n=50)
        faulty     = [b for b in ds_sample if len(b["errors"]) > 0]
        bill       = faulty[0] if faulty else ds_sample[0]
        raw_text_display = "[ Sample bill generated by BILLSENTINEL AI dataset generator ]"
        st.session_state["use_sample"] = False   # prevent loop on next rerun
    else:
        with st.spinner("📄 Reading document…"):
            try:
                bill, raw_text_display = parse_bill(uploaded)
            except Exception as ex:
                st.error(f"❌ Failed to parse document: {ex}")
                st.stop()

    # ── run analysis ──────────────────────────────────────────────────────────
    with st.spinner("🧠 Analyzing bill for errors…"):
        try:
            result = analyze_bill(bill)
        except Exception as ex:
            st.error(f"❌ Analysis failed: {ex}")
            st.stop()

    # ─────────────────────────────────────────────────────────────────────────
    # RESULTS LAYOUT
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")

    # ── top-level metrics ─────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("🧾 Total Issues",  result["total_issues"],
                  delta="errors found" if result["total_issues"]>0 else "clean")
    with c2:
        st.metric("🎯 Confidence",    f"{result['confidence_score']:.1%}")
    with c3:
        st.metric("💰 Est. Savings",  f"₹{result['estimated_savings']:,.0f}")
    with c4:
        st.metric("📋 Bill Score",    f"{result['bill_score']}/100")
    with c5:
        st.metric("🧮 Line Items",    len(bill["line_items"]))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── two-column layout ─────────────────────────────────────────────────────
    left, right = st.columns([3, 2], gap="large")

    with left:
        # ── bill info ──
        st.markdown('<div class="card-title">🏥 Extracted Bill Data</div>', unsafe_allow_html=True)
        meta_l, meta_r = st.columns(2)
        with meta_l:
            st.markdown(f"""
            <div class="card">
              <div style='font-size:.78rem;color:rgba(255,255,255,.45);margin-bottom:.2rem'>BILL ID</div>
              <div style='font-weight:700;color:#e2e8f0'>{bill['bill_id']}</div>
              <br>
              <div style='font-size:.78rem;color:rgba(255,255,255,.45);margin-bottom:.2rem'>ADMISSION</div>
              <div style='color:#a7f3d0'>{bill['admission_date']}</div>
            </div>""", unsafe_allow_html=True)
        with meta_r:
            st.markdown(f"""
            <div class="card">
              <div style='font-size:.78rem;color:rgba(255,255,255,.45);margin-bottom:.2rem'>TOTAL AMOUNT</div>
              <div style='font-weight:700;font-size:1.3rem;color:#fbbf24'>₹{bill['total_cost']:,.2f}</div>
              <br>
              <div style='font-size:.78rem;color:rgba(255,255,255,.45);margin-bottom:.2rem'>DISCHARGE</div>
              <div style='color:#fca5a5'>{bill['discharge_date']}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="card-title" style="margin-top:.8rem">📋 Line Items</div>', unsafe_allow_html=True)
        _render_line_items(bill["line_items"])

        # ── OCR text expander ──
        if raw_text_display:
            with st.expander("📄 Raw Extracted Text", expanded=False):
                st.code(raw_text_display[:3000] + ("…" if len(raw_text_display)>3000 else ""), language="text")

    with right:
        # ── bill score ring ──
        st.markdown('<div class="card-title">🎯 Bill Health Score</div>', unsafe_allow_html=True)
        _render_score_ring(result["bill_score"])

        st.markdown("<br>", unsafe_allow_html=True)

        # ── savings breakdown ──
        rule_err = [e for e in result["errors"] if e.get("source")=="rule"]
        ml_err   = [e for e in result["errors"] if e.get("source")=="ml"]
        st.markdown(f"""
        <div class="card">
          <div class="card-title">💰 Savings Breakdown</div>
          <div style='display:flex;justify-content:space-between;margin-bottom:.4rem'>
            <span style='color:rgba(255,255,255,.6)'>Rule-detected excess</span>
            <span style='color:#a7f3d0;font-weight:600'>
              ₹{sum(e.get('cost',0) for e in rule_err):,.2f}
            </span>
          </div>
          <div style='display:flex;justify-content:space-between;margin-bottom:.4rem'>
            <span style='color:rgba(255,255,255,.6)'>ML anomaly estimate</span>
            <span style='color:#fbbf24;font-weight:600'>
              ₹{sum(e.get('cost',0) for e in ml_err):,.2f}
            </span>
          </div>
          <hr style='border-color:rgba(255,255,255,.08);margin:.5rem 0'>
          <div style='display:flex;justify-content:space-between'>
            <span style='color:#fff;font-weight:700'>Total Estimated Savings</span>
            <span style='color:#34d399;font-weight:800;font-size:1.05rem'>
              ₹{result['estimated_savings']:,.2f}
            </span>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── detected errors ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🚨 Detected Errors")

    if result["total_issues"] > 0:
        n_rule = len([e for e in result["errors"] if e.get("source")=="rule"])
        n_ml   = len([e for e in result["errors"] if e.get("source")=="ml"])
        st.markdown(f"""
        <div class="warn-banner">
          Found <strong>{result['total_issues']} issue(s)</strong> —
          {n_rule} rule-based &nbsp;·&nbsp; {n_ml} ML-flagged
        </div>""", unsafe_allow_html=True)
    _render_errors(result["errors"])

    # ── explanations ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💬 Explanations & Recommended Actions")
    _render_explanations(result["explanations"])

    # ── raw JSON ──────────────────────────────────────────────────────────────
    with st.expander("🔬 Full Analysis JSON", expanded=False):
        st.json(result)

    # ── download report ───────────────────────────────────────────────────────
    report = {
        "bill_id":           bill["bill_id"],
        "analyzed_at":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "admission_date":    bill["admission_date"],
        "discharge_date":    bill["discharge_date"],
        "total_billed":      bill["total_cost"],
        "line_items":        bill["line_items"],
        "analysis":          result,
    }
    st.download_button(
        label="📥 Download Report (JSON)",
        data=json.dumps(report, indent=2),
        file_name=f"billsentinel_report_{bill['bill_id']}.json",
        mime="application/json",
    )

else:
    # ── empty state ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center;padding:3rem 1rem;color:rgba(255,255,255,.35)'>
      <div style='font-size:4rem;margin-bottom:1rem'>🧾</div>
      <div style='font-size:1.1rem;font-weight:600'>Upload a medical bill to get started</div>
      <div style='font-size:.88rem;margin-top:.5rem'>
        Supports PDF, JPG, PNG and TXT formats<br>
        Or click <strong>Use Sample Bill</strong> in the sidebar
      </div>
    </div>""", unsafe_allow_html=True)
