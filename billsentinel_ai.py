"""
BILLSENTINEL AI v2 — Enhanced Medical Billing Error Detection Pipeline
"""
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import json, random, warnings
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# SEED
# ─────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
# SERVICE CATALOGUE  (category → list of services)
# ─────────────────────────────────────────────
SERVICE_CATALOGUE = {
    "ICU": [
        {"desc": "ICU Daily Charges",        "code": "ICU001", "cost_range": (5000,  20000)},
        {"desc": "Ventilator Support",        "code": "ICU002", "cost_range": (8000,  20000)},
        {"desc": "Cardiac Monitoring",        "code": "ICU003", "cost_range": (5000,  15000)},
    ],
    "Surgery": [
        {"desc": "Knee Replacement",          "code": "SUR001", "cost_range": (40000, 100000)},
        {"desc": "Appendectomy",              "code": "SUR002", "cost_range": (20000,  60000)},
        {"desc": "Laparoscopic Cholecystectomy","code":"SUR003", "cost_range": (25000,  70000)},
        {"desc": "Hernia Repair",             "code": "SUR004", "cost_range": (20000,  55000)},
    ],
    "X-ray": [
        {"desc": "Chest X-Ray",              "code": "XRY001", "cost_range": (500,   3000)},
        {"desc": "Abdominal X-Ray",          "code": "XRY002", "cost_range": (600,   2500)},
        {"desc": "Spine X-Ray",              "code": "XRY003", "cost_range": (700,   3000)},
        {"desc": "MRI Brain",               "code": "XRY004", "cost_range": (3000, 12000)},
    ],
    "Consultation": [
        {"desc": "General Consultation",     "code": "CON001", "cost_range": (200,   1000)},
        {"desc": "Specialist Consultation",  "code": "CON002", "cost_range": (500,   1500)},
        {"desc": "Follow-up Visit",          "code": "CON003", "cost_range": (200,    800)},
    ],
    "Medicine": [
        {"desc": "Antibiotic Course",        "code": "MED001", "cost_range": (100,   2000)},
        {"desc": "Pain Management Drugs",    "code": "MED002", "cost_range": (200,   3000)},
        {"desc": "IV Fluids",               "code": "MED003", "cost_range": (100,   1500)},
        {"desc": "Anaesthesia Drugs",       "code": "MED004", "cost_range": (500,   5000)},
    ],
}

# Category average cost (midpoint) for overpricing detection
CATEGORY_AVG = {
    cat: np.mean([((s["cost_range"][0] + s["cost_range"][1]) / 2) for s in svcs])
    for cat, svcs in SERVICE_CATALOGUE.items()
}

# Flat list with category tag
ALL_SERVICES = [
    {**svc, "category": cat}
    for cat, svcs in SERVICE_CATALOGUE.items()
    for svc in svcs
]

DIAGNOSIS_CODES  = ["E11.9","I10","J18.9","M54.5","K21.0","N39.0","F41.1","Z87.891",
                     "I25.10","J44.1","N18.3","M16.9"]
PROCEDURE_CODES  = ["99213","99214","99232","71046","93000","80053","85025","27447",
                     "70553","99244","99246","93010"]
ERROR_TYPES      = ["duplicate_charge","post_discharge_charge","mismatch_treatment",
                    "hidden_fee","overpricing"]

ML_ANOMALY_THRESHOLD = 0.55    # RF proba threshold to include ML anomaly


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _jitter(cost: float) -> float:
    """±10–20% noise."""
    factor = 1.0 + random.uniform(-0.20, 0.20)
    return round(cost * factor, 2)


def _rand_date(start: datetime, end: datetime) -> datetime:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, max(delta, 0)))


def _make_item(adm: datetime, dis: datetime,
               svc: dict = None, date_override: datetime = None,
               category: str = None) -> dict:
    if svc is None:
        svc = random.choice(ALL_SERVICES)
    # category may come from ALL_SERVICES (has key) or be passed explicitly
    cat = svc.get("category", category or "General")
    cost = _jitter(random.uniform(*svc["cost_range"]))
    date = date_override if date_override else _rand_date(adm, dis)
    return {
        "desc":     svc["desc"],
        "code":     svc["code"],
        "category": cat,
        "date":     date.strftime("%Y-%m-%d"),
        "cost":     max(round(cost, 2), 1.0),
    }


# ─────────────────────────────────────────────
# STEP 1 — DATASET GENERATOR
# ─────────────────────────────────────────────
def _build_clean_bill(adm: datetime, dis: datetime) -> list:
    """Realistic correlated base items."""
    items = []
    n_days = max((dis - adm).days, 1)

    # Consultations — 1 per 3 days minimum
    for _ in range(max(1, n_days // 3)):
        items.append(_make_item(adm, dis,
            random.choice(SERVICE_CATALOGUE["Consultation"]), category="Consultation"))

    # Medicines — multiple
    for _ in range(random.randint(2, 4)):
        items.append(_make_item(adm, dis,
            random.choice(SERVICE_CATALOGUE["Medicine"]), category="Medicine"))

    # Diagnostic (X-ray/imaging) — 1–2
    for _ in range(random.randint(1, 2)):
        items.append(_make_item(adm, dis,
            random.choice(SERVICE_CATALOGUE["X-ray"]), category="X-ray"))

    # Chance of Surgery (30%)
    if random.random() < 0.30:
        items.append(_make_item(adm, dis,
            random.choice(SERVICE_CATALOGUE["Surgery"]), category="Surgery"))
        # Surgery implies ICU stay
        if random.random() < 0.5:
            for _ in range(random.randint(1, 3)):
                items.append(_make_item(adm, dis,
                    random.choice(SERVICE_CATALOGUE["ICU"]), category="ICU"))

    # Trim/pad to 3–8
    random.shuffle(items)
    return items[:8] if len(items) > 8 else (items if len(items) >= 3
                                             else items + [_make_item(adm, dis) for _ in range(3 - len(items))])


def _inject_errors(items: list, adm: datetime, dis: datetime,
                   n_errors: int) -> list:
    """Inject balanced errors; return error list."""
    errors = []
    pool = (ERROR_TYPES * ((n_errors // len(ERROR_TYPES)) + 2))[:n_errors]
    pool = random.sample(pool, min(n_errors, len(pool)))

    for etype in pool:
        if etype == "duplicate_charge":
            src = random.choice(items)
            dup = dict(src)
            items.append(dup)
            errors.append({"type": "duplicate_charge", "line_index": len(items) - 1})

        elif etype == "post_discharge_charge":
            post = dis + timedelta(days=random.randint(1, 7))
            svc  = random.choice(ALL_SERVICES)
            items.append(_make_item(adm, dis, svc, date_override=post))
            errors.append({"type": "post_discharge_charge", "line_index": len(items) - 1})

        elif etype == "mismatch_treatment":
            idx = random.randint(0, len(items) - 1)
            wrong_svc = random.choice([s for s in ALL_SERVICES if s["code"] != items[idx]["code"]])
            items[idx] = dict(items[idx])
            items[idx]["code"] = wrong_svc["code"]   # code doesn't match desc
            errors.append({"type": "mismatch_treatment", "line_index": idx})

        elif etype == "hidden_fee":
            idx  = random.randint(0, len(items) - 1)
            orig = items[idx]["cost"]
            n_splits = random.randint(2, 3)
            splits = sorted([round(orig * random.uniform(0.05, 0.30), 2)
                             for _ in range(n_splits - 1)])
            for sp in splits:
                items[idx]["cost"] = max(round(items[idx]["cost"] - sp, 2), 1.0)
                items.append({
                    "desc":     "Administrative Processing Fee",
                    "code":     "ADM001",
                    "category": "Admin",
                    "date":     items[idx]["date"],
                    "cost":     sp,
                })
                errors.append({"type": "hidden_fee", "line_index": len(items) - 1})

        elif etype == "overpricing":
            idx = random.randint(0, len(items) - 1)
            items[idx] = dict(items[idx])
            items[idx]["cost"] = round(items[idx]["cost"] * random.uniform(3.0, 8.0), 2)
            errors.append({"type": "overpricing", "line_index": idx})

    return errors


def generate_dataset(n: int = 2000) -> list:
    dataset = []
    n_clean  = int(n * 0.40)
    n_faulty = n - n_clean
    labels   = ["clean"] * n_clean + ["faulty"] * n_faulty
    random.shuffle(labels)

    # Error type counter for balance tracking
    error_counter = {e: 0 for e in ERROR_TYPES}

    for i, label in enumerate(labels):
        bill_id = f"BILL-{i+1:05d}"
        adm = datetime(2022, 1, 1) + timedelta(days=random.randint(0, 700))
        dis = adm + timedelta(days=random.randint(1, 14))

        items  = _build_clean_bill(adm, dis)
        errors = []

        if label == "faulty":
            n_err = random.randint(1, 3)
            errors = _inject_errors(items, adm, dis, n_err)
            for e in errors:
                error_counter[e["type"]] = error_counter.get(e["type"], 0) + 1

        total_cost = round(sum(it["cost"] for it in items), 2)

        dataset.append({
            "bill_id":        bill_id,
            "admission_date": adm.strftime("%Y-%m-%d"),
            "discharge_date": dis.strftime("%Y-%m-%d"),
            "diagnosis_codes": random.sample(DIAGNOSIS_CODES, k=random.randint(1, 3)),
            "procedure_codes": random.sample(PROCEDURE_CODES, k=random.randint(1, 4)),
            "line_items":      items,
            "total_cost":      total_cost,
            "errors":          errors,
        })

    with open("dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"[OK] Dataset saved: {n} records  ({n_clean} clean / {n_faulty} faulty)")
    print(f"     Error distribution: {error_counter}")
    return dataset


# ─────────────────────────────────────────────
# STEP 2 — ADVANCED FEATURE ENGINEERING
# ─────────────────────────────────────────────
def _count_duplicates(items: list) -> int:
    seen, count = [], 0
    for it in items:
        key = (it["desc"], it.get("date"), it["cost"])
        if key in seen:
            count += 1
        else:
            seen.append(key)
    return count


def _post_discharge_count(items: list, discharge_date: str) -> int:
    dis = datetime.strptime(discharge_date, "%Y-%m-%d")
    return sum(1 for it in items
               if datetime.strptime(it["date"], "%Y-%m-%d") > dis)


def _same_day_repeat(items: list) -> int:
    from collections import Counter
    dates = [it["date"] for it in items]
    return sum(1 for c in Counter(dates).values() if c > 1)


def extract_features(dataset: list):
    X, y = [], []
    for bill in dataset:
        items = bill["line_items"]
        costs = np.array([it["cost"] for it in items], dtype=np.float64)
        adm   = datetime.strptime(bill["admission_date"], "%Y-%m-%d")
        dis   = datetime.strptime(bill["discharge_date"], "%Y-%m-%d")

        num_items       = len(items)
        avg_cost        = float(np.mean(costs))
        max_cost        = float(np.max(costs))
        cost_std        = float(np.std(costs)) if num_items > 1 else 0.0
        n_dups          = _count_duplicates(items)
        duplicate_ratio = n_dups / num_items if num_items else 0.0
        pdc             = _post_discharge_count(items, bill["discharge_date"])
        cost_spike      = max_cost / avg_cost if avg_cost > 0 else 1.0
        same_day_rep    = _same_day_repeat(items)
        unique_svcs     = len(set(it["desc"] for it in items))

        X.append([
            num_items, avg_cost, max_cost, cost_std,
            duplicate_ratio, pdc, cost_spike,
            same_day_rep, unique_svcs,
        ])
        y.append(1 if len(bill["errors"]) > 0 else 0)

    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int32)


FEATURE_NAMES = [
    "num_items", "avg_cost", "max_cost", "cost_std",
    "duplicate_ratio", "post_discharge_count", "cost_spike_ratio",
    "same_day_repeat", "unique_service_count",
]


# ─────────────────────────────────────────────
# STEP 3 — MODEL TRAINING
# ─────────────────────────────────────────────
_iso_forest = None
_rf_clf     = None


def train_models(dataset: list):
    global _iso_forest, _rf_clf

    X, y = extract_features(dataset)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # IsolationForest
    _iso_forest = IsolationForest(
        n_estimators=300, contamination=0.5, random_state=SEED
    )
    _iso_forest.fit(X_tr)
    iso_pred = (_iso_forest.predict(X_te) == -1).astype(int)

    print("\n[IsolationForest]")
    print(f"  Accuracy : {accuracy_score(y_te, iso_pred):.4f}")
    print(f"  Precision: {precision_score(y_te, iso_pred, zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(y_te, iso_pred, zero_division=0):.4f}")

    # RandomForest with predict_proba
    _rf_clf = RandomForestClassifier(
        n_estimators=300, random_state=SEED, class_weight="balanced",
        max_depth=None, min_samples_leaf=2
    )
    _rf_clf.fit(X_tr, y_tr)
    rf_pred = _rf_clf.predict(X_te)

    print("\n[RandomForestClassifier]")
    print(f"  Accuracy : {accuracy_score(y_te, rf_pred):.4f}")
    print(f"  Precision: {precision_score(y_te, rf_pred, zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(y_te, rf_pred, zero_division=0):.4f}")

    return _iso_forest, _rf_clf


# ─────────────────────────────────────────────
# STEP 4 — ENHANCED RULE ENGINE
# ─────────────────────────────────────────────
def _get_category_avg(category: str) -> float:
    return CATEGORY_AVG.get(category, 5000.0)


def rule_engine(bill: dict) -> list:
    issues = []
    items  = bill["line_items"]
    dis    = datetime.strptime(bill["discharge_date"], "%Y-%m-%d")

    # Build lookup structures
    desc_date_map  = {}   # (desc, date) → list of indices
    desc_cost_map  = {}   # desc → list of costs

    for idx, it in enumerate(items):
        key = (it["desc"], it["date"])
        desc_date_map.setdefault(key, []).append(idx)
        desc_cost_map.setdefault(it["desc"], []).append(it["cost"])

    reported = set()

    for idx, it in enumerate(items):
        key = (it["desc"], it["date"])

        # Rule 1 — Same-day duplicate (same desc + same date)
        if len(desc_date_map[key]) > 1 and idx != desc_date_map[key][0]:
            tag = ("duplicate_charge", idx)
            if tag not in reported:
                reported.add(tag)
                issues.append({
                    "type":       "duplicate_charge",
                    "line_index": idx,
                    "detail":     f"'{it['desc']}' billed multiple times on {it['date']}.",
                    "cost":       it["cost"],
                })

        # Rule 2 — Post-discharge charge
        try:
            item_dt = datetime.strptime(it["date"], "%Y-%m-%d")
            if item_dt > dis:
                tag = ("post_discharge_charge", idx)
                if tag not in reported:
                    reported.add(tag)
                    issues.append({
                        "type":       "post_discharge_charge",
                        "line_index": idx,
                        "detail":     f"Charge on {it['date']} after discharge {bill['discharge_date']}.",
                        "cost":       it["cost"],
                    })
        except Exception:
            pass

        # Rule 3 — Overpricing: cost > 2× category average
        cat_avg = _get_category_avg(it.get("category", ""))
        if it["cost"] > 2.0 * cat_avg:
            tag = ("overpricing", idx)
            if tag not in reported:
                reported.add(tag)
                issues.append({
                    "type":       "overpricing",
                    "line_index": idx,
                    "detail":     (
                        f"'{it['desc']}' costs INR {it['cost']:,.2f} "
                        f"vs category avg INR {cat_avg:,.2f}."
                    ),
                    "cost":       it["cost"] - cat_avg,   # excess
                })

    # Rule 4 — Hidden fee: multiple entries of same desc where total > 1.5× single max
    for desc, costs in desc_cost_map.items():
        if len(costs) > 1:
            total = sum(costs)
            mx    = max(costs)
            if total > 1.5 * mx:
                for idx2, it2 in enumerate(items):
                    if it2["desc"] == desc and ("hidden_fee", idx2) not in reported:
                        reported.add(("hidden_fee", idx2))
                        issues.append({
                            "type":       "hidden_fee",
                            "line_index": idx2,
                            "detail":     (
                                f"'{desc}' split across {len(costs)} entries; "
                                f"total INR {total:,.2f} vs largest INR {mx:,.2f}."
                            ),
                            "cost":       round(total - mx, 2),
                        })

    # Rule 5 — Same-day repeat service (across all descs)
    date_items = {}
    for idx, it in enumerate(items):
        date_items.setdefault(it["date"], []).append((idx, it))
    for date_str, grp in date_items.items():
        descs = [g[1]["desc"] for g in grp]
        for desc in set(descs):
            occurrences = [(idx2, it2) for idx2, it2 in grp if it2["desc"] == desc]
            if len(occurrences) > 1:
                for idx2, it2 in occurrences[1:]:
                    tag = ("same_day_repeat", idx2)
                    if tag not in reported:
                        reported.add(tag)
                        issues.append({
                            "type":       "duplicate_charge",
                            "line_index": idx2,
                            "detail":     f"Same-day repeat: '{desc}' on {date_str}.",
                            "cost":       it2["cost"],
                        })

    return issues


# ─────────────────────────────────────────────
# STEP 5 — HYBRID FUSION LOGIC
# ─────────────────────────────────────────────
def _ml_confidence(bill: dict) -> tuple:
    """Returns (is_anomaly_bool, confidence_float)."""
    feats, _ = extract_features([bill])
    iso_flag  = (_iso_forest.predict(feats)[0] == -1)
    rf_proba  = _rf_clf.predict_proba(feats)[0][1]   # P(faulty)
    return iso_flag, float(rf_proba)


# ─────────────────────────────────────────────
# STEP 6 — DYNAMIC EXPLANATION GENERATOR
# ─────────────────────────────────────────────
_TEMPLATES = {
    "duplicate_charge": (
        "The same service has been billed more than once, "
        "which inflates the bill unfairly. "
        "Request removal of the duplicate entry."
    ),
    "post_discharge_charge": (
        "A charge has been added after the patient was discharged, "
        "suggesting a billing error or fraudulent entry. "
        "Verify with the hospital and dispute if unconfirmed."
    ),
    "mismatch_treatment": (
        "The procedure code does not match the service description, "
        "which can indicate upcoding or data entry errors. "
        "Ask for itemized clarification from the billing department."
    ),
    "hidden_fee": (
        "A service cost appears split into smaller unlabeled entries, "
        "making the total higher than it should be. "
        "Request a fully itemized receipt to verify every charge."
    ),
    "overpricing": (
        "This item is priced significantly above the typical category rate, "
        "indicating possible inflated billing. "
        "Compare with standard rate schedules and negotiate or dispute."
    ),
    "anomaly_detected": (
        "The AI model has flagged this bill as statistically unusual "
        "compared to normal billing patterns. "
        "A manual review by a medical auditor is recommended."
    ),
}


def generate_explanation(error_type: str, bill_context: dict = None) -> str:
    return _TEMPLATES.get(error_type,
                          "An unclassified billing irregularity was detected. Seek expert review.")


# ─────────────────────────────────────────────
# STEP 7 — COST SAVINGS ESTIMATION
# ─────────────────────────────────────────────
def estimate_savings(errors: list, bill: dict) -> float:
    """Sum suspicious costs from rule-flagged errors."""
    total = 0.0
    for err in errors:
        if err.get("source") == "rule" and "cost" in err:
            total += err["cost"]
    # Fallback: if no cost attached, use 5% of total bill per ML error
    ml_errors = [e for e in errors if e.get("source") == "ml"]
    if ml_errors:
        total += len(ml_errors) * bill["total_cost"] * 0.05
    return round(total, 2)


# ─────────────────────────────────────────────
# STEP 8 — BILL SCORE
# ─────────────────────────────────────────────
def calculate_score(errors: list) -> int:
    return max(0, 100 - len(errors) * 10)


# ─────────────────────────────────────────────
# STEP 9 — FINAL PIPELINE: analyze_bill
# ─────────────────────────────────────────────
def analyze_bill(bill: dict) -> dict:
    if _iso_forest is None or _rf_clf is None:
        raise RuntimeError("Models not trained. Call train_models() first.")

    # Step A — Rule engine (high confidence)
    rule_issues = rule_engine(bill)
    rule_errors = [
        {
            "type":       iss["type"],
            "line_index": iss["line_index"],
            "detail":     iss.get("detail", ""),
            "cost":       iss.get("cost", 0.0),
            "confidence": 1.0,
            "source":     "rule",
        }
        for iss in rule_issues
    ]

    # Step B — ML anomaly detection
    iso_flag, rf_proba = _ml_confidence(bill)
    ml_errors = []
    if rf_proba >= ML_ANOMALY_THRESHOLD:
        ml_errors.append({
            "type":       "anomaly_detected",
            "line_index": -1,
            "detail":     (
                f"IsolationForest={'anomaly' if iso_flag else 'normal'}, "
                f"RF_proba={rf_proba:.3f}"
            ),
            "cost":       0.0,
            "confidence": round(rf_proba, 4),
            "source":     "ml",
        })

    # Step C — Merge (dedup by type+line_index)
    seen, merged = set(), []
    for err in rule_errors + ml_errors:
        key = (err["type"], err["line_index"])
        if key not in seen:
            seen.add(key)
            merged.append(err)

    # Step D — Explanations
    explanations = [
        {
            "type":        e["type"],
            "explanation": generate_explanation(e["type"], bill),
        }
        for e in merged
    ]

    # Step E — Aggregate confidence
    confidences = [e["confidence"] for e in merged]
    overall_conf = round(float(np.mean(confidences)), 4) if confidences else 0.0

    # Step F — Savings & Score
    savings    = estimate_savings(merged, bill)
    bill_score = calculate_score(merged)

    return {
        "errors":            merged,
        "explanations":      explanations,
        "total_issues":      len(merged),
        "confidence_score":  overall_conf,
        "estimated_savings": savings,
        "bill_score":        bill_score,
    }


# ─────────────────────────────────────────────
# STEP 10 — MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    SEP = "=" * 65

    print(SEP)
    print("  BILLSENTINEL AI v2 — Medical Billing Error Detection")
    print(SEP)

    # 1. Dataset
    print("\n[STEP 1] Generating dataset (n=2000)...")
    dataset = generate_dataset(n=2000)

    # 2. Train
    print("\n[STEP 3] Training models...")
    train_models(dataset)

    # 3. Pick a faulty sample
    faulty = [b for b in dataset if len(b["errors"]) > 0]
    sample = faulty[0]

    print(f"\n{SEP}")
    print("[STEP 9] Analyzing sample bill...")
    print(f"  Bill ID      : {sample['bill_id']}")
    print(f"  Admitted     : {sample['admission_date']}")
    print(f"  Discharged   : {sample['discharge_date']}")
    print(f"  Total        : INR {sample['total_cost']:,.2f}")
    print(f"  Known errors : {[e['type'] for e in sample['errors']]}")
    print(SEP)

    result = analyze_bill(sample)

    print(f"\n  Total Issues     : {result['total_issues']}")
    print(f"  Confidence Score : {result['confidence_score']:.4f}")
    print(f"  Estimated Savings: INR {result['estimated_savings']:,.2f}")
    print(f"  Bill Score       : {result['bill_score']} / 100")

    print("\n  DETECTED ERRORS:")
    for err in result["errors"]:
        print(f"    [{err['source'].upper()}] {err['type']} "
              f"| line={err['line_index']} "
              f"| conf={err['confidence']:.2f}")
        if err["detail"]:
            print(f"           {err['detail']}")

    print("\n  EXPLANATIONS:")
    for exp in result["explanations"]:
        print(f"    [{exp['type']}]")
        print(f"      {exp['explanation']}")

    print(f"\n{SEP}")
    print("  Pipeline completed successfully.")
    print(SEP)
