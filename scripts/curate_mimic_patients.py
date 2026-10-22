#!/usr/bin/env python3
"""
Curate MIMIC-IV demo patients for PIMAP demonstration.

Rewritten per scripts/jc_rewrite_curation_plan.txt (Option C):
  - Each curated patient keeps exactly ONE "most interesting" ICU stay,
    eliminating multi-stay timeline conflation.
  - Pressure-ulcer evidence reuse: corrected methodology from the
    ulcer_analysis rework (ICD-9 707.0x/707.2x, ICD-10 L89.x; charted
    itemids 228649=1, 228549-228558 'Pressure Ulcer', stage items).
    ICD-9 707.1x (foot ulcers), Braden/skin-integrity assessments, and
    'Pressure Ulcer Present = 0' are NOT evidence.

Selection rules (one stay per patient):
  - Ulcer-positive patients: the ICU stay containing charted pressure-
    ulcer evidence wins (prefer POST-INTAKE timing >12h after hospital
    admission, then most evidence rows, then longest stay).
  - Low-risk class: longest stay with stable-high Braden (min sum >= 16).
  - Med/high-risk class: stay >= 48h with risk evolution -- either
    charted ulcer evidence arising POST-INTAKE or a Braden sum change
    >= 3 points at an assessment > 12h after ICU intime; fallback =
    longest stay.
  - Single-stay patients keep their only stay regardless.

Outputs (same filenames as before):
  data/mimic_demo/patients_curated.json
  data/mimic_demo/vitals_curated.json.gz   (one stay_id per patient)

Requires the MIMIC-IV demo CSVs (see MIMIC_PATH below).
"""

import csv
import gzip
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

MIMIC_PATH = Path("/mnt/c/Users/jcasp/Downloads/mimic-iv-clinical-database-demo-2.2")

# --- roster -----------------------------------------------------------------
# Low risk: stable high Braden (control cases for the dashboard demo)
LOW_RISK = [
    '10022880',  # 19-19
    '10002930',  # 18-22
    '10029484',  # 18-18
    '10013049',  # 18-20
    '10009049',  # 18-21
]

# Med/high risk: low/declining Braden or mixed signals
HIGH_RISK = [
    '10015931',  # Braden 6-18, declined 15->11
    '10038081',  # Braden 6-12, declined 12->6
    '10007818',  # Braden 8-14, low BP (62); ulcer POST-INTAKE
    '10005817',  # Braden 8-20, low BP (65); ulcer POST-INTAKE
    '10039708',  # Braden 9-22, declined 19->14
    '10021487',  # Braden 9-18
    '10035631',  # 20->9 severe decline
    '10031757',  # 18->13; ulcer POST-INTAKE
    '10020187',  # Braden 17, SBP 95 (mixed signals)
    '10012853',  # Braden 16, SBP 75
    '10026354',  # Braden 16, SBP 85
    '10006053',  # 8-12, no BP data (for variety)
]

# Pressure-ulcer cohort (corrected analysis, tier 1 + tier 2);
# 10005817 / 10007818 / 10031757 already in HIGH_RISK
ULCER_ADDITIONS = [
    '10002428',  # ulcer stays incl. POST-INTAKE (38875437)
    '10003400',  # ulcer POST-INTAKE (Stage IV sacrum)
    '10004235',  # ulcer POST-INTAKE
    '10010867',  # ulcer POST-INTAKE (Stage II chin)
    '10019003',  # ulcer POST-INTAKE (DTI gluteal)
    '10020640',  # ulcer AT-INTAKE (coccyx)
    '10020786',  # ulcer AT-INTAKE (Stage I coccyx)
    '10032725',  # ICD 707.03/707.22 + charted Stage II
    '10040025',  # ICD L89.899 + charted Stage II coccyx
]

# Hand-picked demo ulcer events: patient_id -> (hadm_id, event timestamp).
# Event timestamps are the first charted pressure-ulcer evidence times from
# pressure_ulcer_patients_corrected.txt (MIMIC-IV demo 2.2). For each ulcer
# patient, curation keeps the ICU stay whose [intime, outtime] window
# contains the event, so the demo data always spans the ulcer onset.
# Multi-stay ulcer patients are hand-picked:
#   10002428 -> hadm 28662225 (POST-INTAKE shortly after ICU admission)
#   10032725 -> hadm 20611640 (only stay with charted ICU data)
ULCER_EVENT_OVERRIDES = {
    '10002428': ('28662225', '2156-04-19 18:55:00'),
    '10003400': ('23559586', '2137-08-10 20:00:00'),
    '10004235': ('24181354', '2196-02-25 08:04:00'),
    '10005817': ('28661809', '2135-01-17 16:42:00'),
    '10007818': ('22987108', '2146-06-26 00:00:00'),
    '10010867': ('22429197', '2148-01-08 12:09:00'),
    '10019003': ('27525946', '2153-04-14 11:15:00'),
    '10020640': ('27984218', '2153-02-13 01:55:00'),
    '10020786': ('23488445', '2189-06-09 13:05:00'),
    '10031757': ('28477280', '2137-10-13 20:00:00'),
    '10032725': ('20611640', '2143-03-22 08:36:00'),
    '10040025': ('27996267', '2148-01-25 10:43:00'),
}

SELECTED_PATIENTS = LOW_RISK + HIGH_RISK + ULCER_ADDITIONS

# --- pressure-ulcer evidence definitions (corrected methodology) ------------
# ICD-9 707.0x (site) / 707.2x (stage); ICD-10 L89.*. 707.1x foot ulcers excluded.
ULCER_ICD9 = re.compile(r"707(0\d|2[0-5])")
ULCER_ICD10_PREFIX = "L89"
PU_PRESENT_ITEM = 228649          # 'Pressure Ulcer Present' == '1'
PU_TYPE_ITEMS = set(range(228549, 228559))    # PU #N- Type == 'Pressure Ulcer'
PU_STAGE_ITEMS = ({224631}                        # Pressure Ulcer Stage #1
                  | set(range(224965, 224972))   # Pressure Ulcer Stage #2-#8
                  | set(range(228813, 228823)))  # Pressure ulcer #N- Stage
ULCER_INTAKE_THRESHOLD_H = 12.0

BRADEN_FIELDS_RAW = [
    'braden_sensory', 'braden_moisture', 'braden_activity',
    'braden_mobility', 'braden_nutrition', 'braden_friction_shear',
]

SMOKING_ICD9 = {'3051', 'V1582'}
SMOKING_ICD10_PREFIXES = ('F17', 'Z87891')

# Selection thresholds (per curation plan)
HIGH_RISK_MIN_DURATION_H = 48.0
BRADEN_CHANGE_THRESHOLD = 3
BRADEN_CHANGE_AFTER_H = 12.0     # assessment must occur > 12h after ICU intime
LOW_RISK_MIN_BRADEN = 16.0       # 'stable-high' = stay min Braden sum >= 16


# --- loading helpers ---------------------------------------------------------
def read_mimic_csv(rel_path):
    with gzip.open(MIMIC_PATH / rel_path, "rt", newline="") as f:
        return list(csv.DictReader(f))


def parse_dt(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S") if s else None


def load_admissions():
    return {r['hadm_id']: parse_dt(r['admittime'])
            for r in read_mimic_csv("hosp/admissions.csv.gz")}


def load_icustays():
    """subject_id -> list of stay dicts (sorted by intime)."""
    stays = defaultdict(list)
    for r in read_mimic_csv("icu/icustays.csv.gz"):
        stays[r['subject_id']].append({
            'stay_id': int(r['stay_id']),
            'hadm_id': r['hadm_id'],
            'intime': parse_dt(r['intime']),
            'outtime': parse_dt(r['outtime']),
            'unit': r['first_careunit'],
        })
    for v in stays.values():
        v.sort(key=lambda s: s['intime'])
    return stays


def load_smoking_status():
    """subject_id (str) -> True if smoking-related diagnosis present."""
    smokers = {}
    for r in read_mimic_csv("hosp/diagnoses_icd.csv.gz"):
        code = r['icd_code'].strip()
        version = r.get('icd_version', '9')
        if version == '9':
            if code in SMOKING_ICD9:
                smokers[r['subject_id']] = True
        elif version == '10':
            if any(code.startswith(p) for p in SMOKING_ICD10_PREFIXES):
                smokers[r['subject_id']] = True
    return smokers


def is_ulcer_icd(code, version):
    if version == '9':
        return bool(ULCER_ICD9.fullmatch(code))
    return version == '10' and code.startswith(ULCER_ICD10_PREFIX)


# --- pressure-ulcer evidence --------------------------------------------------
def load_ulcer_evidence():
    """Gather corrected-methodology pressure-ulcer evidence.

    Returns (icd_subjects, chart_evidence) where chart_evidence maps
    subject_id -> list of dicts:
      {hadm_id, charttime, kind ('present'|'type'|'stage'), value}
    Evidence is attached to an ICU stay only when its charttime falls
    within the stay's [intime, outtime] window (see attach_evidence).
    """
    icd_subjects = set()
    chart_evidence = defaultdict(list)

    for r in read_mimic_csv("hosp/diagnoses_icd.csv.gz"):
        if is_ulcer_icd(r['icd_code'].strip(), r.get('icd_version', '9')):
            icd_subjects.add(r['subject_id'])

    for r in read_mimic_csv("icu/chartevents.csv.gz"):
        item = int(r['itemid'])
        val = r['value']
        kind = None
        if item == PU_PRESENT_ITEM and val == '1':
            kind = 'present'
        elif item in PU_TYPE_ITEMS and val == 'Pressure Ulcer':
            kind = 'type'
        elif item in PU_STAGE_ITEMS and val and val != 'Not applicable':
            kind = 'stage'
        if kind:
            chart_evidence[r['subject_id']].append({
                'hadm_id': r['hadm_id'],
                'charttime': parse_dt(r['charttime']),
                'kind': kind,
                'value': val,
            })
    return icd_subjects, chart_evidence


def attach_evidence_to_stays(evidence_rows, stays, admittime):
    """Map evidence rows onto ICU stays by charttime containment.

    Returns dict stay_id -> {'rows': [...], 'first_vs_admit_h': float|None,
    'timing': 'AT-INTAKE'|'POST-INTAKE'|None}
    """
    by_stay = {}
    for ev in evidence_rows:
        for s in stays:
            if ev['hadm_id'] == s['hadm_id'] and s['intime'] and ev['charttime'] \
                    and s['intime'] <= ev['charttime'] <= (s['outtime'] or datetime.max):
                by_stay.setdefault(s['stay_id'], []).append(ev)
                break
    result = {}
    for stay_id, rows in by_stay.items():
        first_ev = min(r['charttime'] for r in rows)
        admit = admittime.get(rows[0]['hadm_id'])
        hours = ((first_ev - admit).total_seconds() / 3600.0
                 if (first_ev and admit) else None)
        timing = None
        if hours is not None:
            timing = 'POST-INTAKE' if hours > ULCER_INTAKE_THRESHOLD_H else 'AT-INTAKE'
        result[stay_id] = {'rows': rows, 'first_vs_admit_h': hours, 'timing': timing}
    return result


# --- per-stay metrics ----------------------------------------------------------
def compute_stay_metrics(records, intime):
    """Braden trajectory metrics for one ICU stay.

    records: time-sorted vitals dicts for this stay.
    Returns dict with assessment count, first/min/max sum, max |change|
    from first, and criterion_b (Braden sum change >= 3 points at an
    assessment > 12h after ICU intime).
    """
    assessments = []
    for r in records:
        vals = [r.get(f) for f in BRADEN_FIELDS_RAW]
        if all(v is not None for v in vals):
            assessments.append((r['timestamp'], sum(vals)))
    if not assessments:
        return {'n_assess': 0, 'first_sum': None, 'min_sum': None,
                'max_sum': None, 'max_change': None, 'criterion_b': False}

    first = assessments[0][1]
    min_sum = min(a[1] for a in assessments)
    max_sum = max(a[1] for a in assessments)
    max_change = max(abs(a[1] - first) for a in assessments)
    criterion_b = False
    for ts_str, val in assessments:
        ts = datetime.strptime(ts_str[:19], "%Y-%m-%dT%H:%M:%S")
        if intime and (ts - intime).total_seconds() / 3600.0 > BRADEN_CHANGE_AFTER_H \
                and abs(val - first) >= BRADEN_CHANGE_THRESHOLD:
            criterion_b = True
            break
    return {'n_assess': len(assessments), 'first_sum': first,
            'min_sum': min_sum, 'max_sum': max_sum,
            'max_change': max_change, 'criterion_b': criterion_b}


# --- stay selection ------------------------------------------------------------
def select_stay(pid, patient_class, stay_metrics, ulcer_subject):
    """Pick one stay per patient. Returns (stay_id, reason)."""

    # Ulcer-positive patients: the stay containing the hand-picked demo
    # ulcer event wins; without an override, fall back to ranking by
    # charted evidence (POST-INTAKE first, then rows, then duration).
    if ulcer_subject:
        override = ULCER_EVENT_OVERRIDES.get(pid)
        if override:
            hadm_id, event_s = override
            event = datetime.strptime(event_s, "%Y-%m-%d %H:%M:%S")
            containing = [
                m for m in stay_metrics
                if m['hadm_id'] == hadm_id
                and m['intime'] and m['intime'] <= event
                and (m['outtime'] is None or event <= m['outtime'])
            ]
            if containing:
                best = containing[0]
                return best['stay_id'], (
                    f"override: stay containing demo ulcer event "
                    f"{event_s} (hadm {hadm_id})"
                )
            print(f"  {pid}  WARNING: ulcer event override {event_s} "
                  f"(hadm {hadm_id}) falls outside every ICU stay window; "
                  f"falling back to evidence ranking")

        evidence_stays = [m for m in stay_metrics if m['ulcer_rows']]
        if evidence_stays:
            best = sorted(
                evidence_stays,
                key=lambda m: (
                    0 if m['ulcer_timing'] == 'POST-INTAKE' else 1,  # POST first
                    -m['ulcer_rows'],                                # then rows
                    -m['duration_h'],                                # then length
                ),
            )[0]
            return best['stay_id'], (
                f"ulcer stay ({best['ulcer_rows']} evidence rows, "
                f"first evidence {best['ulcer_first_h']:+.1f}h vs admittime "
                f"= {best['ulcer_timing']})"
            )
        # ICD-only ulcer patient without charted evidence in any stay:
        # fall through to class rules.

    if len(stay_metrics) == 1:
        m = stay_metrics[0]
        return m['stay_id'], "single stay"

    if patient_class == 'low-risk':
        stable = [m for m in stay_metrics
                  if m['min_sum'] is not None and m['min_sum'] >= LOW_RISK_MIN_BRADEN]
        pool = stable or stay_metrics
        best = max(pool, key=lambda m: m['duration_h'])
        return best['stay_id'], (
            f"longest stable-high Braden stay "
            f"(Braden {best['first_sum']:g}-{best['max_sum']:g}, "
            f"{best['duration_h']:.0f}h)"
        )

    # med/high-risk: >=48h with risk evolution, else longest
    qualifying = [m for m in stay_metrics
                  if m['duration_h'] >= HIGH_RISK_MIN_DURATION_H
                  and (m['criterion_b'] or (m['ulcer_timing'] == 'POST-INTAKE'))]
    if qualifying:
        best = max(qualifying, key=lambda m: (m['duration_h'], m['n_assess']))
        why = []
        if best['ulcer_timing'] == 'POST-INTAKE':
            why.append("POST-INTAKE ulcer evidence")
        if best['criterion_b']:
            # direction of the largest notable change (up or down from first)
            lo, hi = best['min_sum'], best['max_sum']
            worst = lo if abs(lo - best['first_sum']) >= abs(hi - best['first_sum']) else hi
            why.append(f"Braden change >= {BRADEN_CHANGE_THRESHOLD} after "
                       f"{BRADEN_CHANGE_AFTER_H:g}h "
                       f"({best['first_sum']:g}->{worst:g}, min {lo:g})")
        return best['stay_id'], (
            f"qualifying stay {best['duration_h']:.0f}h: " + "; ".join(why)
        )
    best = max(stay_metrics, key=lambda m: (m['duration_h'], m['n_assess']))
    return best['stay_id'], (
        f"fallback: longest/most-dense stay "
        f"({best['duration_h']:.0f}h, {best['n_assess']} assessments)"
    )


# --- main -----------------------------------------------------------------------
def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "mimic_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading extracted MIMIC demo data...")
    with open(output_dir / "patients_mimic_demo.json") as f:
        all_patients = json.load(f)
    with open(output_dir / "vitals_mimic_demo.json") as f:
        all_vitals = json.load(f)

    print("Loading ICU stays, admissions, smoking status, ulcer evidence...")
    icustays = load_icustays()
    admittime = load_admissions()
    smoking_status = load_smoking_status()
    icd_ulcer_subjects, chart_evidence = load_ulcer_evidence()
    ulcer_subjects = icd_ulcer_subjects | set(chart_evidence)
    print(f"  smoking flags: {len(smoking_status)} patients")
    print(f"  ulcer-positive patients in demo: {len(ulcer_subjects)} "
          f"(ICD: {len(icd_ulcer_subjects)}, charted: {len(chart_evidence)})")

    patient_class = {}
    for pid in LOW_RISK:
        patient_class[pid] = 'low-risk'
    for pid in HIGH_RISK + ULCER_ADDITIONS:
        patient_class.setdefault(pid, 'high-risk')

    print(f"\nCurating {len(SELECTED_PATIENTS)} patients "
          f"({len(LOW_RISK)} low-risk, {len(HIGH_RISK)} med/high-risk, "
          f"{len(ULCER_ADDITIONS)} ulcer additions)...")
    print("Selection report:")
    print("  pid       class      chosen stay  duration  criteria")

    chosen_stay = {}
    roster = sorted(set(SELECTED_PATIENTS), key=int)
    for pid in roster:
        stays = icustays.get(pid, [])
        vitals_by_stay = defaultdict(list)
        for v in all_vitals:
            if v['patient_id'] == pid:
                vitals_by_stay[v.get('stay_id')].append(v)

        stay_metrics = []
        ev_by_stay = attach_evidence_to_stays(
            chart_evidence.get(pid, []), stays, admittime)
        for s in stays:
            recs = sorted(vitals_by_stay.get(s['stay_id'], []),
                          key=lambda r: r['timestamp'])
            dur = ((s['outtime'] - s['intime']).total_seconds() / 3600.0
                   if (s['outtime'] and s['intime']) else 0.0)
            braden = compute_stay_metrics(recs, s['intime'])
            ev = ev_by_stay.get(s['stay_id'],
                                {'rows': [], 'first_vs_admit_h': None,
                                 'timing': None})
            stay_metrics.append({
                'stay_id': s['stay_id'], 'hadm_id': s['hadm_id'],
                'intime': s['intime'], 'outtime': s['outtime'],
                'duration_h': dur, 'n_records': len(recs),
                'ulcer_rows': len(ev['rows']),
                'ulcer_first_h': ev['first_vs_admit_h'],
                'ulcer_timing': ev['timing'],
                **braden,
            })

        if not stay_metrics:
            print(f"  {pid}  WARNING: no ICU stays found; keeping all records")
            chosen_stay[pid] = None
            continue

        stay_id, reason = select_stay(
            pid, patient_class.get(pid, 'high-risk'), stay_metrics,
            ulcer_subject=pid in ulcer_subjects)
        chosen_stay[pid] = stay_id
        is_ulcer = 'ULCER' if pid in ulcer_subjects else '      '
        print(f"  {pid}  {patient_class.get(pid, '?'):9s}  {is_ulcer} "
              f"{stay_id}  {next(m['duration_h'] for m in stay_metrics if m['stay_id'] == stay_id):7.0f}h  {reason}")

    # ---- build outputs: one stay per patient ----
    curated_patients = [p for p in all_patients
                        if p['patient_id'] in SELECTED_PATIENTS]
    for p in curated_patients:
        p['smoker_any'] = smoking_status.get(p['patient_id'], False)

    stay_to_intime = {}
    for subj_stays in icustays.values():
        for s in subj_stays:
            stay_to_intime[s['stay_id']] = s['intime']

    curated_vitals = []
    for v in all_vitals:
        if v['patient_id'] not in SELECTED_PATIENTS:
            continue
        keep = chosen_stay.get(v['patient_id'])
        if keep is None or v.get('stay_id') == keep:
            stay_id = v.get('stay_id')
            if stay_id and stay_id in stay_to_intime:
                vt = datetime.fromisoformat(v['timestamp'].replace('Z', '+00:00'))
                v['icu_stay_duration'] = max(
                    0, (vt.replace(tzinfo=None) - stay_to_intime[stay_id]).days)
            else:
                v['icu_stay_duration'] = 0
            curated_vitals.append(v)

    print(f"\nSelected {len(set(SELECTED_PATIENTS))} patients "
          f"({len(curated_patients)} patient-floor records), "
          f"{len(curated_vitals)} vital records")
    smokers_found = sum(1 for p in curated_patients if p.get('smoker_any'))
    print(f"  Smokers identified: {smokers_found}")

    multi = [pid for pid, s in chosen_stay.items() if s is not None]
    stays_per_patient = defaultdict(set)
    for v in curated_vitals:
        stays_per_patient[v['patient_id']].add(v.get('stay_id'))
    bad = {pid: ss for pid, ss in stays_per_patient.items() if len(ss) > 1}
    if bad:
        print(f"  WARNING: multi-stay patients remain: {bad}")
    else:
        print(f"  Verified: exactly one stay_id per patient "
              f"({len(stays_per_patient)} patients with vitals)")

    # Verify ulcer-event coverage: each ulcer patient's kept stay must
    # contain the demo event, and its kept vitals must start at or
    # before the event so the dashboard plot spans the ulcer onset.
    print("\nUlcer event coverage check:")
    by_patient_ts = defaultdict(list)
    for v in curated_vitals:
        by_patient_ts[v['patient_id']].append(v['timestamp'])
    all_ok = True
    for pid, (hadm_id, event_s) in sorted(ULCER_EVENT_OVERRIDES.items()):
        event = datetime.strptime(event_s, "%Y-%m-%d %H:%M:%S")
        stay = chosen_stay.get(pid)
        stays = icustays.get(pid, [])
        match = next((s for s in stays if s['stay_id'] == stay), None)
        ts = by_patient_ts.get(pid, [])
        first = datetime.fromisoformat(min(ts)) if ts else None
        ok = bool(match and match['hadm_id'] == hadm_id
                  and match['intime'] and match['intime'] <= event
                  and (match['outtime'] is None or event <= match['outtime'])
                  and first is not None and first <= event)
        all_ok = all_ok and ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {pid}: event {event_s} "
              f"hadm={hadm_id} stay={stay} "
              f"first_vitals={first.isoformat() if first else 'NONE' if not ts else first}")
    print(f"  overall: {'ALL CHECKS PASSED' if all_ok else 'CHECKS FAILED'}")

    patients_output = output_dir / "patients_curated.json"
    with open(patients_output, 'w') as f:
        json.dump(curated_patients, f, indent=2)
    print(f"Wrote {patients_output}")

    vitals_output = output_dir / "vitals_curated.json.gz"
    with gzip.open(vitals_output, 'wt') as f:
        json.dump(curated_vitals, f)
    print(f"Wrote {vitals_output} ({vitals_output.stat().st_size / 1024:.1f} KB)")

    print("\nPatient summary:")
    for p in curated_patients:
        pid = p['patient_id']
        p_vitals = [v for v in curated_vitals if v['patient_id'] == pid]
        smoker = "SMOKER" if p.get('smoker_any') else ""
        ulcer = "ULCER" if pid in ulcer_subjects else ""
        print(f"  {pid}: {len(p_vitals)} records, stay={chosen_stay.get(pid)}, "
              f"floor={p.get('floor', '?')[:30]} {smoker} {ulcer}")


if __name__ == '__main__':
    main()
