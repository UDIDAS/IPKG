import json, os, csv, datetime, glob
from collections import Counter, defaultdict

SCR = os.path.dirname(os.path.abspath(__file__))
M = json.load(open(r'<app-workspace>\round2\results\split_manifests_2026-10-01.json'))
T = json.load(open(r'<app-workspace>\round2\results\promptfree_twin_exposure_2026-10-01.json'))
P = json.load(open(r'<app-workspace>\round2\results\promptfree_round2_2026-10-01.json'))
C = json.load(open(r'<app-workspace>\paper\flare23_twin_scan_2026-09-20.json'))


def norm(x):
    return 'volume-' + x.split('_')[1] if x.startswith('liver_') else x


# ---- the co-author voxel verdicts -------------------------------------------------
vox = {}
for fn in (['twin_batch1.json', 'twin_batch2.json'] +
           sorted(glob.glob(os.path.join(SCR, 'twin_chunk*.json'))) +
           sorted(glob.glob(os.path.join(SCR, 'missed_chunk*.json')))):
    p = fn if os.path.isabs(fn) else os.path.join(SCR, fn)
    if os.path.exists(p):
        for r in json.load(open(p)):
            vox[frozenset((norm(r['a']), norm(r['b'])))] = r['verdict']

# ---- our independent census -------------------------------------------
ours = {f: norm(v['twin']['case']) for f, v in C['results'].items() if v.get('twin')}
our_scanned = set(C['results'])
inv = defaultdict(set)
for f, t in ours.items():
    inv[t].add(f)

TK = {'lits': 'lits', 'msd': 'pancreas', 'kits': 'kits', 'flare23': 'flare'}
OK = {'lits': 'lits', 'msd': 'msd', 'kits': 'kits', 'flare23': 'flare23'}
COLL = {'lits': 'LiTS', 'msd': 'MSD Pancreas', 'kits': 'KiTS23', 'flare23': 'FLARE23'}
SHOW = {'volume-76', 'case_00067', 'pancreas_125', 'FLARE23_0405'}


def role(case, block, key):
    for r in ('train', 'val', 'test'):
        if case in block[key][r]:
            return r
    return 'absent'


def grole(c):
    for ds in ('pancreas', 'lits', 'flare', 'kits'):
        for r in ('train', 'val', 'test'):
            if c in M['tumor_pools']['global'][ds][r]:
                return ds + '/' + r
    return 'not in tumor pools'


def coll_of(cid):
    for pre, name in (('FLARE23_', 'FLARE23'), ('volume-', 'LiTS'),
                      ('pancreas_', 'MSD Pancreas'), ('case_', 'KiTS23')):
        if cid.startswith(pre):
            return name
    return ''


testlists = {dn: [c['case'] for c in dv['cases']]
             for dn, dv in P['models']['sam3_tumor_generic']['datasets'].items()}
alltest = set().union(*[set(v) for v in testlists.values()])

rows = []
for dn in ('lits', 'msd', 'kits', 'flare23'):
    tw = {c['case']: c for c in T['datasets'][dn]['cases']}
    for pid in testlists[dn]:
        e = tw.get(pid, {})
        pipeline_twins = [t['twin'] for t in e.get('twins', [])]
        pipeline_roles = {t['twin']: t['twin_role'] for t in e.get('twins', [])}
        # our census partners for this patient (both directions)
        if dn == 'flare23':
            app_twins = {ours[pid]} if pid in ours else set()
            scanned = pid in our_scanned
        else:
            app_twins = set(inv.get(pid, set()))
            scanned = bool(app_twins) or any(t in our_scanned for t in pipeline_twins)
        # every distinct partner, from either census
        allt = sorted(set(pipeline_twins) | app_twins)
        verdicts = {t: vox.get(frozenset((pid, t)), 'not voxel-checked') for t in allt}
        refuted = {t for t in allt if verdicts[t] in
                   ('SHAPE_MISMATCH', 'VALUE_HISTOGRAM_DIFFERS')}
        confirmed = [t for t in allt if t not in refuted]
        exposing = [t for t in confirmed if grole(t).endswith('/train') or grole(t).endswith('/val')]
        dup = [t for t in confirmed if t in alltest]
        rows.append({
            'collection': COLL[dn],
            'patient_id': pid,
            'pipeline_role_global_split': role(pid, M['tumor_pools']['global'], TK[dn]),
            'pipeline_role_filtered_split': role(pid, M['tumor_pools']['filtered'], TK[dn]),
            'pipeline_sam3_organ_model_role_round1': role(pid, M['organ_pool_lkp']['filtered'], OK[dn]),
            'pipeline_twins': ';'.join(pipeline_twins),
            'app_census_twins': ';'.join(sorted(app_twins)),
            'twins_pipeline_missed': ';'.join(sorted(app_twins - set(pipeline_twins))),
            'twins_pipeline_claimed_that_app_refuted': ';'.join(sorted(refuted & set(pipeline_twins))),
            'twin_roles_global_split': ';'.join('%s=%s' % (t, grole(t)) for t in allt),
            'app_voxel_verdicts': ';'.join('%s=%s' % (t, verdicts[t]) for t in allt),
            'twin_collection': ';'.join(sorted(set(coll_of(t) for t in allt))),
            'pipeline_exposed_via_twin': 'yes' if e.get('exposed') else 'no',
            'app_exposed_via_twin': 'yes' if exposing else 'no',
            'app_exposing_twin': ';'.join(exposing),
            'duplicate_of_another_test_patient': ';'.join(dup),
            'pipeline_twin_clean': 'no' if e.get('exposed') else 'yes',
            'app_twin_clean': 'no' if exposing else 'yes',
            'app_primary_cohort': ('no' if (exposing or any(d < pid for d in dup)) else 'yes'),
            'showcase_case': 'yes' if pid in SHOW else 'no',
        })

summary = {}
for dn in ('lits', 'msd', 'kits', 'flare23'):
    rs = [r for r in rows if r['collection'] == COLL[dn]]
    summary[COLL[dn]] = {
        'n_in_pipeline_test_list': len(rs),
        'pipeline_exposed_via_twin': sum(1 for r in rs if r['pipeline_exposed_via_twin'] == 'yes'),
        'app_exposed_via_twin': sum(1 for r in rs if r['app_exposed_via_twin'] == 'yes'),
        'pipeline_twin_clean_n': sum(1 for r in rs if r['pipeline_twin_clean'] == 'yes'),
        'app_twin_clean_n': sum(1 for r in rs if r['app_twin_clean'] == 'yes'),
        'app_primary_cohort_n': sum(1 for r in rs if r['app_primary_cohort'] == 'yes'),
        'n_twins_pipeline_missed': sum(1 for r in rs if r['twins_pipeline_missed']),
        'n_twins_pipeline_claimed_refuted': sum(1 for r in rs if r['twins_pipeline_claimed_that_app_refuted']),
        'n_duplicate_rows': sum(1 for r in rs if r['duplicate_of_another_test_patient']),
        'sam3_organ_model_roles_round1': dict(Counter(r['pipeline_sam3_organ_model_role_round1'] for r in rs)),
        'pipeline_filtered_split_roles': dict(Counter(r['pipeline_role_filtered_split'] for r in rs)),
        'app_voxel_verdict_tally': dict(Counter(
            v.split('=')[1] for r in rs for v in r['app_voxel_verdicts'].split(';') if v)),
    }
summary['TOTAL'] = {
    'n_in_pipeline_test_list': len(rows),
    'pipeline_exposed_via_twin': sum(1 for r in rows if r['pipeline_exposed_via_twin'] == 'yes'),
    'app_exposed_via_twin': sum(1 for r in rows if r['app_exposed_via_twin'] == 'yes'),
    'pipeline_twin_clean_n': sum(1 for r in rows if r['pipeline_twin_clean'] == 'yes'),
    'app_twin_clean_n': sum(1 for r in rows if r['app_twin_clean'] == 'yes'),
    'app_primary_cohort_n': sum(1 for r in rows if r['app_primary_cohort'] == 'yes'),
    'distinct_physical_patients': len(rows) - sum(
        1 for r in rows if r['duplicate_of_another_test_patient'] and
        any(d < r['patient_id'] for d in r['duplicate_of_another_test_patient'].split(';'))),
}

pack = {
    'title': 'IPKG aligned evaluation split pack (application to pipeline)',
    'generated': datetime.date.today().isoformat(),
    'purpose': 'Fix ONE patient list per collection so the pipeline team tumour Dice and the co-author organ Dice are computed on identical patients.',
    'basis': {
        'pipeline_file': 'round2/results/split_manifests_2026-10-01.json',
        'pipeline_test_lists': 'tumor_pools.global.<collection>.test, the split sam3_tumor_generic was trained under',
        'app_split_verification': 'recipe re-run independently (numpy RandomState(42) consumed once per dataset in dataset_order, sorted then shuffle, 20pct test / 10pct val / rest train); all 8 dataset x mode splits reproduced member-for-member',
        'app_twin_verification': 'voxel comparison under axis permutation and flips, exact equality; 57 pairs checked',
        'app_census': 'paper/flare23_twin_scan_2026-09-20.json, 576 FLARE23 studies scanned, 310 twins',
    },
    'conventions': {
        'pipeline_role_global_split': 'role under the GLOBAL split, the one sam3_tumor_generic was trained under, i.e. the round-2 evaluation split',
        'pipeline_role_filtered_split': 'role under the per-dataset FILTERED split, the one the round-1 per-dataset AUSAM models were trained under',
        'warning': 'the same patient has DIFFERENT roles in the two splits; a role is meaningless without naming the model',
        'app_primary_cohort': 'twin-clean AND not a duplicate of an earlier-sorted test patient; this is the cohort we propose as primary',
    },
    'summary': summary,
    'patients': rows,
}

json.dump(pack, open(os.path.join(SCR, 'ipkg_split_pack.json'), 'w'), indent=1)
with open(os.path.join(SCR, 'ipkg_split_pack.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(json.dumps(summary, indent=1))
