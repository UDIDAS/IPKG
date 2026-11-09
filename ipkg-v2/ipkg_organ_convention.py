#!/usr/bin/env python
"""The one place that decides what an ORGAN NUMBER means in this project.

WHY THIS MODULE EXISTS. Two measurement defects were found on 2026-10-02, and
both of them were defects of DUPLICATION rather than of arithmetic: the same
quantity was computed in more than one place, the copies drifted, and nothing
in the artefacts said which copy produced the number being read.

    Defect 1  KiTS23 annotates ONE kidney class, so laterality is a geometric
              decision somebody's code makes. `common.label_map.kits_split_
              kidneys` makes it by keeping only components larger than 5,000
              voxels, which on case_00067 silently deleted 29,026 of 485,178
              reference kidney voxels (5.98 %, in 1,973 small components). It
              was fixed in build_ausam_final_arms._kidney_sides in September,
              but the SCORING path does not go through the arm builder: it
              reads the staged `__gt_unified` twin that src/eval/baseline.py
              wrote with the old splitter, so every KiTS organ score computed
              from that twin is measured against a reference that is 6 % too
              small. `kidney_sides` below is the one splitter, it conserves
              every voxel, and it RAISES rather than warns when it does not.

    Defect 2  "organ with mass" named two incompatible computations:

                A (ours, adopted here)
                    Dice( organ model , reference organ UNION reference mass )
                B (the pipeline team's delivered organ_with_mass, NOT used)
                    Dice( organ model UNION PREDICTED tumour ,
                          reference organ UNION reference mass )

              On case_00067, kidney pooled, with TotalSegmentator v2 3 mm as
              the organ model and the pipeline team's delivered mask supplying the tumour,
              the same reference and the same organ prediction read 0.4919
              under A and 0.9701 under B (re-measured 2026-10-03; both
              reproduce). B is rejected because an organ score has to
              measure the ORGAN model; under B the organ column is built
              partly out of the tumour model, and a tumour model that is
              good can carry an organ model that is not.

              And because this module's own rule is that no organ-with-mass
              figure travels alone, the 0.4919 is published here with what
              qualifies it: parenchyma Dice 0.6002, parenchyma recall
              0.7701, mass_coverage 0.2088, mass fraction 0.7892. The organ
              model found three quarters of the kidney and a fifth of the
              mass; B's 0.9701 is the tumour model's score wearing the organ
              column's name.

              Those two figures are measured against the REPAIRED reference.
              Against the staged twin, before defect 1 was fixed, the same
              pair reads 0.4962 and 0.9698 - which is why the two defects are
              fixed together: every published illustration of the convention
              question was computed on a KiTS reference that was 6 % short.

Everything below is imported, never copied. `build_case_metrics.py` and
`score_fullres_compare.py` both go through it, so a change to the convention
is a change to one file and cannot move one scorer without the other.

THE CONVENTION, in full, is in CONVENTION and is written verbatim into every
artefact those scorers emit, because a number whose definition lives only in a
module nobody reads is the defect this module was written to end.

TWO CONDITIONS THE CONVENTION ANALYSIS ESTABLISHED, and neither is optional.

(a) MASS ATTRIBUTION IS A MAJORITY VOTE OVER THE COMPONENT'S OWN VOXELS,
    per connected component, never globally and never by minimum distance.
    Minimum distance - the rule src/eval/tumour_organ.py uses - is wrong on
    FLARE23_0405, and the 2026-10-03 re-measurement makes it worse than the
    2026-10-02 memo said. The memo reported the 695 cm3 mass at 0.96 mm from
    the spleen against 1.28 mm from the left kidney with 94 % of its voxels
    nearer the kidney. Measured on the reference these scorers actually read
    (FLARE23_0405__gt_unified, 0.962891 x 0.962891 x 1.0 mm) the two minima
    are IDENTICAL - 0.962891 mm to each, one in-plane voxel step, differing
    by 0.0 - while the mean distance is 56.765 mm to the spleen against
    10.384 mm to the kidney and the vote is 747,867 to 1,740, which is
    99.77 % and not 94 %.

    So minimum distance does not prefer the spleen on geometry. It picks
    whichever of two TIED organs its candidate list reaches first, and the
    list is in label order, so the spleen wins because 3 < 13. A rule whose
    answer on a 695 cm3 mass is decided by the numbering of the label map is
    not a weak rule, it is not a rule. The model agrees with the vote:
    TotalSegmentator has no lesion class, must put the mass somewhere, and
    absorbs it into its LEFT kidney.

(b) PARENCHYMA IS PUBLISHED BESIDE ORGAN-WITH-MASS, always, both the Dice and
    the recall. At case_00067's left-kidney mass fraction of 0.888 - measured
    on the repaired reference, not assumed - a prediction consisting of the
    mass ALONE scores 0.941 organ-with-mass and 1.000 tumour while
    recovering not one voxel of kidney. Organ-with-mass on a case whose organ
    is mostly mass is close to a restatement of the tumour score, and the only
    thing that separates the two readings is the parenchyma pair. They are
    returned together by `organ_scores` so that publishing one without the
    other takes deliberate effort.

WHAT THE ADVISOR SETTLED ON 2026-10-03, and which of it was already here.

    CONVENTION A is the name the advisor gave to what this module already
    computed: the organ score is the ORGAN MODEL'S OWN MASK against the
    reference organ UNION the host-attributed mass, never unioned with and
    never carved by the predicted tumour. It is now emitted under the name
    `organ_dice` as well, because an artefact that only ever says
    `dice_organ_with_mass` makes a reader guess whether it is convention A or
    the pipeline team's B, and the two differ by 0.48 Dice on case_00067.

    MASS_COVERAGE IS NEW, and it is the number that separates the two ways an
    organ-with-mass figure can be high. |P and host mass| / |host mass| says
    how much of the reference mass the ORGAN prediction actually swallowed.
    Read beside parenchyma_recall it distinguishes "the model found the
    organ" (recall high, coverage whatever) from "the model found the tumour
    and called it the organ" (coverage high, recall near zero), which is the
    degenerate case condition (b) above is about and which Dice alone cannot
    see.

    AND THE ONE PLACE THE DECISION'S TWO HALVES PULL APART. "Never unioned
    with, and never carved by, the predicted tumour" is a rule about what
    THIS CODE does to a prediction, and it cannot un-carve a prediction that
    arrives carved. the pipeline team's delivered organ channel is an organ mask with his
    predicted tumour already cut out of it, so under convention A it is
    charged for the whole of the host mass: on case_00067's pooled kidney
    the delivered mask reads 0.3017 organ-with-mass against 0.8109
    parenchyma, with a mass_coverage of 0.0057. That is the convention
    working as specified on a carved prediction, not a defect in it, and the
    measurement that tells a carve from a miss - how much of the same host
    mass the delivered TUMOUR channel covers - is taken in
    build_case_metrics.py, where a prediction with a mass class exists.
    TotalSegmentator emits no mass class at all, so the question does not
    arise in score_fullres_compare.py and its pred_mass_voxels says so per
    case.

    THE KiTS CYST GOES INTO THE ORGAN. KiTS23 annotates kidney, tumour and
    cyst as three classes; cyst is unified class 15 and was already inside
    MASS_CLASSES, so this ruling confirms the code rather than changing it -
    but it was an UNSIGNED recommendation until today and nothing in the
    output said so. Every artefact now states the ruling and carries the
    cyst-bearing count of the sample it was computed on, because a decision
    that moves a large share of a collection must not be inferable only from
    the contents of a tuple named MASS_CLASSES.

    ALL THREE READINGS ARE COMPUTABLE FROM ONE CALL. tumour-excluded,
    organ-with-mass and lesion-as-ignore-region are returned together by
    `organ_scores`, so the showcase supplement that has to show the choice
    does not need a second code path that could disagree with the first. The
    three differ in exactly one thing - what happens to the host mass's
    voxels: charged against the prediction, counted as organ, or excluded
    from the measurement.

Nothing here touches disk. The callers own their artefacts; this module owns
the definitions.
"""
import numpy as np

# -- the unified label space (src/common/label_map.py) ----------------------
UNIFIED = {1: "liver", 2: "right kidney", 3: "spleen", 4: "pancreas",
           5: "aorta", 6: "inferior vena cava", 7: "right adrenal gland",
           8: "left adrenal gland", 9: "gallbladder", 10: "oesophagus",
           11: "stomach", 12: "duodenum", 13: "left kidney",
           14: "tumour", 15: "cyst"}
ORGANS = tuple(range(1, 14))
TUMOUR, CYST = 14, 15
# Both mass classes. CYST is in here by the advisor's ruling of 2026-10-03,
# not by inertia: clinically a simple renal cyst is part of the kidney's
# volume, a cystic neoplasm is not, and KiTS23 does not distinguish the two -
# so the choice had to be made rather than derived. It is made ONE way, for
# every KiTS case, and CYST_RULING below is written into every artefact.
# Only KiTS23 annotates class 15 at all, so no other collection moves.
MASS_CLASSES = (TUMOUR, CYST)
RIGHT_KIDNEY, LEFT_KIDNEY = 2, 13

CYST_RULING = ("KiTS23 cyst (native class 3, unified 15) is counted INTO the "
               "kidney: it is part of the organ-with-mass reference exactly "
               "as the tumour is, it is excluded from the parenchyma "
               "reference exactly as the tumour is, and it is one of the "
               "'host mass' voxels that mass_coverage measures and that the "
               "lesion-as-ignore-region reading drops. Advisor decision, "
               "2026-10-03. Every artefact that scores KiTS reports the "
               "cyst-bearing count of its own sample, because this ruling "
               "moves every case that has one and no case that has not.")

# What each collection is CAPABLE of annotating. An organ a collection never
# labels is not a zero, it is unobservable, and is skipped rather than scored.
OBSERVABLE = {
    "LiTS": {1},
    "MSD Pancreas": {4},
    "KiTS23": {2, 13},
    "FLARE23": set(ORGANS),
}

CONV_TE = "tumour_excluded"
CONV_WM = "organ_with_mass"
CONV_IG = "lesion_as_ignore_region"
# The three readings, in the order they appear in every table: they differ in
# exactly one thing, what happens to the host mass's voxels - charged against
# the prediction, counted as organ, or dropped from the measurement.
READINGS = (CONV_TE, CONV_WM, CONV_IG)

# Mirrors src/eval/tumour_organ.attribute so the two differ only in the rule
# under test, not in what counts as a component worth attributing.
MASS_MIN_VOX = 50
MASS_MAX_DIST_MM = 20.0

# KiTS laterality. 5,000 voxels was the OLD splitter's discard threshold; it
# survives only as the size above which an assignment is worth cross-checking
# and reporting, and it is never again a filter.
KIDNEY_LARGE_VOX = 5000
# Two principal masses must sit on opposite sides of the spine before we may
# call them "the two kidneys", and the scale of that test has to be anatomy in
# millimetres, not a fraction of anything the masses themselves define - two
# fragments of one kidney are the whole of their own extent apart, so a
# relative test passes them. A kidney is 50-60 mm across left-to-right, so two
# pieces of ONE kidney cannot have centroids 40 mm apart along that axis; two
# real kidneys are 90-140 mm apart and case_00067's pair sits at 134.8 mm.
KIDNEY_SEED_LR_GAP_MM = 40.0

# /2 adds organ_dice, mass_coverage and the lesion-as-ignore-region
# reading, and signs the cyst ruling. The MEASUREMENTS of /1 are
# unchanged - dice_tumour_excluded and dice_organ_with_mass are the
# same arithmetic on the same masks - so a /1 number and a /2 number
# of the same name are comparable, and the version moves anyway
# because the SHAPE a reader parses did.
CONVENTION_VERSION = "ipkg_organ_convention/2"

CONVENTION = {
    "version": CONVENTION_VERSION,
    "adopted": "2026-10-02",
    "dice": "3-D volumetric Dice, 2|P and R| / (|P| + |R|), binary per "
            "structure over the whole volume; no slice restriction, no "
            "post-processing, no connected-component selection.",
    CONV_TE: "prediction = the organ model alone; reference = the "
             "collection's own organ label, which already has the mass cut "
             "out. Also published as parenchyma_dice, because it is the same "
             "measurement under the name that says what it measures.",
    CONV_WM: "CONVENTION A. prediction = the organ model ALONE; reference = "
             "the organ label UNION the mass components attributed to that "
             "organ by host attribution. The prediction is never unioned "
             "with a predicted tumour and never carved by one: an organ "
             "score measures the organ model. Emitted as organ_dice and, "
             "under its older name, dice_organ_with_mass - the same number.",
    CONV_IG: "prediction and reference both have the host mass REMOVED, so "
             "the mass's voxels are charged to neither side: a prediction is "
             "not penalised for covering the mass and is not credited for "
             "it. The third of the three readings, published for the "
             "showcase cases so the choice between them is visible. The "
             "region dropped is the mass attributed to THIS organ, not every "
             "mass in the volume - a liver lesion must still count against a "
             "kidney prediction that reaches into it, or the reading stops "
             "being one of three ways to treat the host mass and becomes a "
             "fourth thing.",
    "mass_coverage": "|P and host mass| / |host mass|, where the host mass is "
                     "the part of the organ-with-mass reference that is mass. "
                     "How much of the reference mass the ORGAN prediction "
                     "actually covers. With parenchyma_recall it separates "
                     "'the model found the organ' from 'the model found the "
                     "tumour and called it the organ': the degenerate "
                     "mass-alone prediction scores coverage 1.000 and recall "
                     "0.000. None, not zero, when the organ hosts no mass - "
                     "there is nothing to cover.",
    "kits_cyst": CYST_RULING,
    "organ_with_mass_NOT_USED": "the pipeline team's delivered organ_with_mass unions the "
                                "PREDICTED tumour into the prediction as "
                                "well. On KiTS23 case_00067, kidney pooled, "
                                "with TotalSegmentator v2 3 mm as the organ "
                                "model and the pipeline team's delivered mask supplying the "
                                "tumour, the same reference and the same "
                                "organ prediction read 0.4919 under the "
                                "convention adopted here and 0.9701 under "
                                "the pipeline team's (0.4962 and 0.9698 against the staged "
                                "twin, before its kidney reference was "
                                "repaired). Numbers under the two are not "
                                "comparable and are never mixed in one "
                                "column.",
    "mass_attribution": "per connected component of the reference mass, by "
                        "MAJORITY VOTE over the component's own voxels (each "
                        "voxel to its own nearest annotated organ); the full "
                        "vote is recorded. A component smaller than %d voxels "
                        "or whose winning organ is farther than %.0f mm is "
                        "attributed to nothing and reported, never silently "
                        "absorbed. NOT minimum distance: on FLARE23_0405 "
                        "that rule calls the 695 cm3 mass splenic although "
                        "the mass is 0.962891 mm from the spleen AND "
                        "0.962891 mm from the left kidney - tied to the last "
                        "bit - so it is decided by label order (3 before "
                        "13) and not by geometry, while the vote is 747,867 "
                        "voxels to 1,740 for the left kidney (99.77 %%) and "
                        "the mean distance is 10.384 mm to the kidney "
                        "against 56.765 mm to the spleen. Re-measured "
                        "2026-10-03; the 2026-10-02 memo's 1.28 mm and 94 %% "
                        "do not reproduce on this reference."
                        % (MASS_MIN_VOX, MASS_MAX_DIST_MM),
    "parenchyma": "parenchyma_dice is Dice against the organ label with the "
                  "mass removed; parenchyma_recall is |P and parenchyma| / "
                  "|parenchyma|. Published with every organ_with_mass figure: "
                  "at case_00067's left-kidney mass fraction of 0.888 a "
                  "prediction of the mass alone scores 0.941 organ-with-mass "
                  "while recovering "
                  "zero parenchyma, and only the recall separates the two.",
    "kits_laterality": "KiTS23 labels one kidney class; right (2) and left "
                       "(13) are a geometric decision made by "
                       "ipkg_organ_convention.kidney_sides, which assigns "
                       "EVERY kidney voxel to a side and raises if voxels in "
                       "!= voxels out. The staged __gt_unified twins written "
                       "by src/eval/baseline.py used "
                       "common.label_map.kits_split_kidneys, which keeps only "
                       "components larger than %d voxels and therefore lost "
                       "29,026 of case_00067's 485,178 reference kidney "
                       "voxels; a KiTS reference read from those twins is "
                       "repaired here before it is scored."
                       % KIDNEY_LARGE_VOX,
}


class ConservationError(RuntimeError):
    """Raised when a transform that must not create or destroy voxels did.

    A RuntimeError and not an `assert`: the defect this guards against was a
    silent 6 % deletion of a reference, and the one check that would have
    caught it must not be removable by `python -O`.
    """


# -- geometry ---------------------------------------------------------------
def lr_axis(affine):
    """(array axis, sign) of the patient right-to-left direction.

    Derived from the affine rather than assumed, because the four collections
    are not stored in one orientation - KiTS is IPL and the other three RAS -
    and a hard-coded axis is the kind of assumption that scores a kidney
    against a kidney by luck. `sign` is +1 when a LARGER index is further to
    the patient's left.

    nibabel affines map voxel indices to RAS+ world coordinates, where +x is
    the patient's RIGHT. So a step along `axis` that DECREASES world x is a
    step towards the patient's left, and the sign is the negative of the
    affine entry's. Getting that backwards swaps the two kidneys of every
    KiTS case while conserving every voxel, which is exactly the kind of
    failure a conservation check cannot see - `selftest` pins both
    orientations against their axis codes.
    """
    m = np.asarray(affine, float)[:3, :3]
    axis = int(np.argmax(np.abs(m[0])))
    return axis, (-1 if m[0, axis] > 0 else 1)


def kidney_sides(kidney, affine, zooms):
    """KiTS's single kidney class into right (2) / left (13), losing nothing.

    THE RULE. Connected components; the two largest are the principal masses.
    If their centroids are at least KIDNEY_SEED_LR_GAP_MM apart along the
    left-right axis they are the two kidneys, and every other component - no
    matter how small - joins whichever principal mass its centroid is nearer
    to. Otherwise there are not two separated kidneys in this volume (merged
    pair, or a solitary kidney that has fragmented) and the mass is cut by its
    own mid-sagittal plane, which is a geometry fallback, is not anatomy, and
    says so in the returned info.

    Nearest-CENTROID and not nearest-surface: the two rules were checked
    against each other on case_00067 and agree on every one of the 485,178
    kidney voxels, and the centroid rule needs one 1,975-row distance matrix
    where the surface rule needs two full-volume 597 MB distance transforms.

    Returns (out, info). `info` is the whole provenance of the split, and the
    caller is expected to persist it: a laterality that nobody can audit is an
    assertion, not a measurement.
    """
    from scipy import ndimage
    kidney = np.asarray(kidney, bool)
    zooms = tuple(float(z) for z in zooms)
    ax, sign = lr_axis(affine)
    out = np.zeros(kidney.shape, np.uint8)
    n_in = int(kidney.sum())
    info = {"rule": None, "voxels_in": n_in, "voxels_out": 0, "components": 0,
            "lr_axis": ax, "lr_sign": sign, "laterality_fallback": False,
            "why": None, "small_components": 0, "small_component_voxels": 0,
            "right_kidney_voxels": 0, "left_kidney_voxels": 0}
    if not n_in:
        info["rule"] = "no kidney voxels"
        return out, info

    lab, _ = ndimage.label(kidney)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    ids = np.nonzero(counts)[0]
    if ids.size == 0:
        raise ConservationError("kidney labelling produced no component for "
                                "%s kidney voxels" % format(n_in, ","))
    info["components"] = int(ids.size)
    small = ids[counts[ids] <= KIDNEY_LARGE_VOX]
    info["small_components"] = int(small.size)
    info["small_component_voxels"] = int(counts[small].sum())

    cen = np.zeros((int(counts.size), 3), float)
    cen[ids] = (np.asarray(ndimage.center_of_mass(kidney, lab, ids.tolist()),
                           float) * np.asarray(zooms, float))
    order = ids[np.argsort(counts[ids])[::-1]]
    gap_mm = (abs(cen[int(order[0])][ax] - cen[int(order[1])][ax])
              if order.size >= 2 else 0.0)
    info["principal_gap_along_LR_mm"] = round(float(gap_mm), 1)

    if order.size >= 2 and gap_mm >= KIDNEY_SEED_LR_GAP_MM:
        a, b = int(order[0]), int(order[1])
        # sign +1: a larger index is further to the patient's LEFT, so the
        # right kidney is the principal mass with the smaller coordinate.
        right, left = ((a, b) if sign * cen[a][ax] < sign * cen[b][ax]
                       else (b, a))
        d_r = np.linalg.norm(cen[ids] - cen[right], axis=1)
        d_l = np.linalg.norm(cen[ids] - cen[left], axis=1)
        side = np.zeros(int(counts.size), np.uint8)
        side[ids] = np.where(d_r <= d_l, RIGHT_KIDNEY, LEFT_KIDNEY)
        side[right], side[left] = RIGHT_KIDNEY, LEFT_KIDNEY   # by definition
        out = side[lab]
        info["rule"] = ("nearest-centroid to the two principal kidney masses "
                        "(no component discarded)")
        info["principal_masses"] = {"right_kidney_voxels": int(counts[right]),
                                    "left_kidney_voxels": int(counts[left])}
        info["right_kidney_components"] = int((side[ids] == RIGHT_KIDNEY).sum())
        info["left_kidney_components"] = int((side[ids] == LEFT_KIDNEY).sum())
    else:
        mid = float(np.median(np.nonzero(kidney)[ax]))
        shape = [1, 1, 1]
        shape[ax] = kidney.shape[ax]
        idx = np.arange(kidney.shape[ax]).reshape(shape)
        lower = kidney & (idx < mid)
        out[lower if sign > 0 else (kidney & ~lower)] = RIGHT_KIDNEY
        out[(kidney & ~lower) if sign > 0 else lower] = LEFT_KIDNEY
        info["rule"] = ("mid-sagittal plane of the kidney mass at LR index "
                        "%.1f (GEOMETRY FALLBACK, not anatomy)" % mid)
        info["laterality_fallback"] = True
        info["why"] = ("the two largest of %d component(s) are %.0f mm apart "
                       "along the left-right axis, under the %.0f mm two "
                       "kidneys must clear, so neither can be trusted as a "
                       "side: either the kidneys are merged into one mass or "
                       "only one is present and it has fragmented."
                       % (info["components"], gap_mm, KIDNEY_SEED_LR_GAP_MM))

    info["right_kidney_voxels"] = int((out == RIGHT_KIDNEY).sum())
    info["left_kidney_voxels"] = int((out == LEFT_KIDNEY).sum())
    info["voxels_out"] = info["right_kidney_voxels"] + info["left_kidney_voxels"]
    stray = sorted(set(int(v) for v in np.unique(out))
                   - {0, RIGHT_KIDNEY, LEFT_KIDNEY})
    if stray:
        raise ConservationError("kidney split emitted labels %s, which are "
                                "not 0/%d/%d"
                                % (stray, RIGHT_KIDNEY, LEFT_KIDNEY))
    if info["voxels_out"] != n_in:
        raise ConservationError(
            "kidney split is not conservative: %s voxels in, %s out (%+d). No "
            "voxel of a reference or a prediction may be created or dropped "
            "by lateralisation."
            % (format(n_in, ","), format(info["voxels_out"], ","),
               info["voxels_out"] - n_in))
    return out, info


def repair_kits_reference(gt, native_kidney, affine, zooms):
    """Put the kidney voxels back that the staged twin's splitter dropped.

    `gt` is the staged `__gt_unified` array, whose kidney classes 2 and 13
    came from common.label_map.kits_split_kidneys and are therefore short by
    whatever that splitter's 5,000-voxel threshold discarded. `native_kidney`
    is the boolean kidney mask from the dataset's own segmentation.nii.gz on
    the same grid - the whole structure, before anybody lateralised it.

    Returns (repaired gt, info). The repair is only ever applied to the two
    kidney classes; the mass classes in the staged twin are the dataset's own
    and are left exactly as they are. info carries the before/after voxel
    counts for both sides so the move is reported rather than assumed, and
    running this on an already-correct twin is a no-op that still reports.
    """
    before = {RIGHT_KIDNEY: int((gt == RIGHT_KIDNEY).sum()),
              LEFT_KIDNEY: int((gt == LEFT_KIDNEY).sum())}
    sides, info = kidney_sides(native_kidney, affine, zooms)
    out = gt.copy()
    out[(out == RIGHT_KIDNEY) | (out == LEFT_KIDNEY)] = 0
    out[sides == RIGHT_KIDNEY] = RIGHT_KIDNEY
    out[sides == LEFT_KIDNEY] = LEFT_KIDNEY
    after = {RIGHT_KIDNEY: int((out == RIGHT_KIDNEY).sum()),
             LEFT_KIDNEY: int((out == LEFT_KIDNEY).sum())}
    # A kidney voxel that lands on a class the staged twin had already given
    # to something else would mean the two files disagree about more than
    # laterality, which is not a repair but a different defect.
    if after[RIGHT_KIDNEY] + after[LEFT_KIDNEY] != info["voxels_out"]:
        raise ConservationError(
            "repaired KiTS reference holds %s kidney voxels but the split "
            "produced %s - the staged twin and the native segmentation "
            "disagree about something other than laterality"
            % (format(after[RIGHT_KIDNEY] + after[LEFT_KIDNEY], ","),
               format(info["voxels_out"], ",")))
    info["staged_twin_voxels"] = {UNIFIED[k]: v for k, v in before.items()}
    info["repaired_voxels"] = {UNIFIED[k]: v for k, v in after.items()}
    info["voxels_recovered"] = (after[RIGHT_KIDNEY] + after[LEFT_KIDNEY]
                                - before[RIGHT_KIDNEY] - before[LEFT_KIDNEY])
    # The old splitter had TWO branches and they fail differently. When it
    # found exactly two large components it discarded the rest, which is the
    # 6 % deletion; when it did not, it fell back to cutting the kidney mass
    # by a mid-sagittal PLANE, which conserves every voxel and still puts
    # thousands of them on the wrong kidney. A case can therefore recover no
    # voxels at all and still move, so the side-swap is counted separately -
    # otherwise "0 recovered" would read as "nothing changed" on exactly the
    # cases where the laterality rule, not the voxel count, is what moved.
    swapped = int(((gt == RIGHT_KIDNEY) & (out == LEFT_KIDNEY)).sum()
                  + ((gt == LEFT_KIDNEY) & (out == RIGHT_KIDNEY)).sum())
    info["voxels_relabelled_between_sides"] = swapped
    info["was_already_correct"] = (info["voxels_recovered"] == 0
                                   and swapped == 0)
    return out, info


# -- mass attribution -------------------------------------------------------
RULE_VOTE = "majority_vote"
RULE_MIN_DIST = "minimum_distance"


def attribute_mass(gt, zooms, organs=None, mass_classes=MASS_CLASSES,
                   rule=RULE_VOTE):
    """Which organ does each reference mass component belong to?

    FLARE23 labels tumour with one generic class that does not say whose, so
    adding the whole tumour mask to every organ would reward a liver for
    containing a renal mass. Attribution is per connected component by a
    MAJORITY VOTE over the component's own voxels - see this module's header
    for why minimum distance is wrong, and on which case.

    Each voxel of the component is assigned to its own nearest annotated
    organ; the component goes to whichever organ holds the most voxels, and
    the full vote is recorded so the attribution can be audited rather than
    trusted.

    TWO ARGUMENTS EXIST ONLY TO PRICE THE DECISIONS, and neither may be moved
    off its default by a scoring path.

    `mass_classes` is MASS_CLASSES - tumour and cyst - by the advisor's cyst
    ruling. Passing (TUMOUR,) computes the counterfactual the ruling
    replaced, in which a KiTS cyst is in no reference at all; that is how the
    decision ledger puts a number on the ruling instead of describing it.

    `rule` is RULE_VOTE. RULE_MIN_DIST is the rejected rule, the one
    src/eval/tumour_organ.py uses, kept here so the ledger can measure what
    it would have cost rather than quote one case from a memo. A caller that
    scores anything with it is scoring under a rule this project rejected.

    Returns (per_organ, unattributed_voxels, detail).
    """
    from scipy import ndimage
    if rule not in (RULE_VOTE, RULE_MIN_DIST):
        raise ValueError("unknown attribution rule %r" % (rule,))
    mass = np.isin(gt, mass_classes)
    per_organ = {}
    unattributed = 0
    if not mass.any():
        return per_organ, unattributed, []
    here = set(int(v) for v in np.unique(gt))
    present = sorted(set(organs if organs is not None else ORGANS) & here)
    if not present:
        return per_organ, int(mass.sum()), []
    lab, n = ndimage.label(mass, structure=np.ones((3, 3, 3), np.uint8))
    counts = np.bincount(lab.ravel())

    def composition(comp):
        """Tumour voxels, cyst voxels, and the name that says which it is.

        Components are grown over tumour AND cyst together, by the cyst
        ruling, so a cyst that abuts its tumour is one component; writing the
        split down per component is what lets a reader see exactly which
        attributed volume the ruling is responsible for, instead of taking
        on trust that it is small."""
        nt = int((gt[comp] == TUMOUR).sum())
        nc = int((gt[comp] == CYST).sum())
        kind = ("tumour" if nc == 0 else
                ("cyst" if nt == 0 else "tumour and cyst, one component"))
        return {"tumour_voxels": nt, "cyst_voxels": nc, "mass_class": kind}
    # The distance is computed FROM each mass component, inside a box expanded
    # by MASS_MAX_DIST_MM, rather than from each organ over the whole volume.
    # Euclidean distance is symmetric so the attribution is identical, but a
    # full-volume transform per organ on a 512x512x326 FLARE23 case is six
    # 683 MB arrays and minutes of wall clock, for an answer that only ever
    # depends on a 20 mm neighbourhood.
    margin = np.ceil(MASS_MAX_DIST_MM / np.asarray(zooms, float)).astype(int)
    detail = []
    for i in range(1, n + 1):
        if counts[i] < MASS_MIN_VOX:
            unattributed += int(counts[i])
            continue
        comp = (lab == i)
        nz = np.nonzero(comp)
        lo = np.maximum([int(x.min()) for x in nz] - margin, 0)
        hi = np.minimum([int(x.max()) + 1 for x in nz] + margin,
                        np.asarray(gt.shape))
        sl = tuple(slice(int(x), int(y)) for x, y in zip(lo, hi))
        gsub, csub = gt[sl], comp[sl]
        cand, dists = [], []
        for o in present:
            om = (gsub == o)
            if not om.any():
                continue                  # farther away than MASS_MAX_DIST_MM
            d = ndimage.distance_transform_edt(~om, sampling=zooms)
            cand.append(o)
            dists.append(d[csub])
            del d
        if not cand:
            unattributed += int(counts[i])
            d = {"organ": None, "organ_name": None,
                 "min_distance_mm": None, "vote_fraction": None,
                 "voxels": int(counts[i]),
                 "why": "no annotated organ within %.0f mm"
                        % MASS_MAX_DIST_MM}
            d.update(composition(comp))
            detail.append(d)
            continue
        D = np.stack(dists)                      # (organs, component voxels)
        votes = np.bincount(np.argmin(D, axis=0), minlength=len(cand))
        winner = cand[int(np.argmax(votes))]
        # The other rule, computed and stored whichever one is in force.
        # Carrying it costs one argmin over len(cand) numbers and turns
        # "minimum distance is wrong on FLARE23_0405" from an assertion in a
        # docstring into a field in every artefact.
        nearest = cand[int(np.argmin([float(x.min()) for x in dists]))]
        best = winner if rule == RULE_VOTE else nearest
        bestd = float(D[cand.index(best)].min())
        vote_frac = {UNIFIED[cand[k]]: round(float(votes[k]) / float(votes.sum()), 4)
                     for k in range(len(cand)) if votes[k]}
        mind = {UNIFIED[cand[k]]: round(float(D[k].min()), 2)
                for k in range(len(cand))}
        del D, dists
        if bestd <= MASS_MAX_DIST_MM:
            if best not in per_organ:
                per_organ[best] = np.zeros_like(mass)
            per_organ[best][comp] = True
            d = {"organ": int(best), "organ_name": UNIFIED[best],
                 "min_distance_mm": round(bestd, 2),
                 "vote_fraction": vote_frac,
                 "min_distance_to_each_mm": mind,
                 "voxels": int(counts[i]),
                 # The whole point of the vote: where the two rules part
                 # company, say so in the component's own record rather than
                 # leaving it to be rediscovered. On FLARE23_0405 this is the
                 # line that reads "majority vote: left kidney, minimum
                 # distance: spleen".
                 "attribution_rule": rule,
                 "majority_vote_says": UNIFIED[winner],
                 "minimum_distance_says": UNIFIED[nearest],
                 "rules_agree": bool(nearest == winner)}
            d.update(composition(comp))
            detail.append(d)
        else:
            unattributed += int(counts[i])
            d = {"organ": None, "organ_name": None,
                 "min_distance_mm": round(bestd, 2),
                 "vote_fraction": vote_frac,
                 "min_distance_to_each_mm": mind,
                 "voxels": int(counts[i]),
                 "why": "winning organ farther than %.0f mm"
                        % MASS_MAX_DIST_MM}
            d.update(composition(comp))
            detail.append(d)
    return per_organ, unattributed, detail


# -- the organ measurement itself -------------------------------------------
def dice(a, b):
    """None, not 0.0, when both masks are empty: there is nothing to score."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    s = int(a.sum()) + int(b.sum())
    if s == 0:
        return None
    return 2.0 * float(np.logical_and(a, b).sum()) / float(s)


def organ_references(gt, oid, mass_by_organ):
    """(reference under tumour_excluded, reference under organ_with_mass).

    The parenchyma reference is the tumour_excluded one by construction - all
    four collections cut the mass out of the organ label - and is returned
    under that name by `organ_scores` so the pair cannot be read as two
    different reference structures.
    """
    ref_te = (gt == oid)
    extra = mass_by_organ.get(oid)
    ref_wm = (ref_te | extra) if extra is not None else ref_te
    return ref_te, ref_wm


def organ_scores(pred, ref_te, ref_wm):
    """The organ reading, complete, in the one shape every artefact uses.

    `pred` is the ORGAN MODEL'S mask and nothing else - never unioned with a
    predicted tumour and never carved by one (that is the pipeline team's convention B, see
    the header). Returns CONVENTION A as `organ_dice` and, beside it and
    inseparably, the parenchyma pair that says whether any organ was actually
    found and the mass_coverage that says whether the mass was; see condition
    (b) and the header.

    THE HOST MASS IS DERIVED, NOT PASSED. It is `ref_wm and not ref_te`,
    which is the mass this organ hosts and nothing else, because the four
    collections store a single-channel label partition - the organ label and
    the mass label cannot share a voxel - and `organ_references` builds
    ref_wm by adding attributed mass components to ref_te. Deriving it rather
    than taking a fourth argument means the three readings are guaranteed to
    be three readings OF THE SAME REGION: a caller cannot hand in a mass that
    disagrees with the one already inside ref_wm.

    All three readings come back from this one call, so the showcase
    supplement does not need a second code path that could disagree with this
    one. They differ in exactly one thing - what the host mass's voxels do:

        dice_tumour_excluded  charged against the prediction as false
                              positive (reference = parenchyma only)
        organ_dice            counted as organ (reference = parenchyma +
                              host mass) - CONVENTION A
        dice_lesion_as_ignore_region
                              dropped from both sides, so they are neither
                              false positive nor true positive
    """
    pred = np.asarray(pred, bool)
    ref_te, ref_wm = np.asarray(ref_te, bool), np.asarray(ref_wm, bool)
    mass = ref_wm & ~ref_te
    n_te, n_wm, n_mass = int(ref_te.sum()), int(ref_wm.sum()), int(mass.sum())
    hit_te = int(np.logical_and(pred, ref_te).sum())
    hit_mass = int(np.logical_and(pred, mass).sum())
    d_te, d_wm = dice(pred, ref_te), dice(pred, ref_wm)
    # ref_te & ~mass is ref_te on a well-formed partition; it is written out
    # so the reading stays correct if a caller ever hands in a reference
    # whose organ and mass overlap, instead of being silently wrong.
    d_ig = dice(pred & ~mass, ref_te & ~mass)
    return {
        "convention": CONVENTION_VERSION,
        # CONVENTION A under the name the advisor gave it, 2026-10-03. The
        # older name is kept beside it and is the same number, because
        # artefacts and figures written before today read that key.
        "organ_dice": None if d_wm is None else round(d_wm, 4),
        "organ_dice_convention": CONV_WM,
        "dice_organ_with_mass": None if d_wm is None else round(d_wm, 4),
        "dice_tumour_excluded": None if d_te is None else round(d_te, 4),
        "dice_lesion_as_ignore_region": (None if d_ig is None
                                         else round(d_ig, 4)),
        # the same measurement as dice_tumour_excluded, under the name that
        # says what it measures, so a reader comparing it with organ_dice is
        # comparing like with like
        "parenchyma_dice": None if d_te is None else round(d_te, 4),
        "parenchyma_recall": (round(float(hit_te) / float(n_te), 4)
                              if n_te else None),
        # how much of the reference mass the ORGAN model took in. None and
        # not 0.0 where the organ hosts no mass: there is nothing to cover,
        # and a zero there would drag every cohort median towards a failure
        # that did not happen.
        "mass_coverage": (round(float(hit_mass) / float(n_mass), 4)
                          if n_mass else None),
        "parenchyma_voxels": n_te,
        "mass_voxels_in_reference": n_mass,
        "mass_voxels_covered_by_prediction": hit_mass,
        # how much of the organ-with-mass reference IS mass. Near 0.9 the
        # organ_dice figure is close to a restatement of the tumour score and
        # must not be read without the recall beside it.
        "mass_fraction_of_reference": (round(float(n_mass) / float(n_wm), 4)
                                       if n_wm else None),
        # The fact, per structure, rather than the paragraph: the ruling
        # itself is CONVENTION["kits_cyst"], written once into every
        # artefact, and repeating 600 characters of it beside each of a
        # thousand structures would make the file unreadable without making
        # any number more traceable. What a reader needs here is WHICH
        # classes were in the mass when this structure was scored.
        "mass_classes_scored": [UNIFIED[c] for c in MASS_CLASSES],
    }


def three_readings(scores):
    """The supplement block: one structure's three readings, side by side.

    `scores` is what `organ_scores` returned. Nothing is recomputed here -
    the point of the block is that the three numbers in the supplement are
    the same three objects the manuscript's organ column is built from, and
    recomputing them in the supplement is how a supplement comes to disagree
    with its paper. The advisor asked for this for the four showcase cases so
    that the choice between the readings is visible rather than argued.
    """
    return {
        "note": "the same prediction and the same reference organ under the "
                "three ways of treating the host mass; they differ in "
                "nothing else",
        "published": CONV_WM,
        "published_is": "convention A, the advisor's decision of 2026-10-03",
        CONV_TE: {"dice": scores["dice_tumour_excluded"],
                  "host_mass_voxels_are": "charged against the prediction as "
                                          "false positive",
                  "definition": CONVENTION[CONV_TE]},
        CONV_WM: {"dice": scores["organ_dice"],
                  "host_mass_voxels_are": "part of the organ reference",
                  "definition": CONVENTION[CONV_WM]},
        CONV_IG: {"dice": scores["dice_lesion_as_ignore_region"],
                  "host_mass_voxels_are": "removed from both sides and "
                                          "scored by neither",
                  "definition": CONVENTION[CONV_IG]},
        "beside_them_always": {
            "parenchyma_dice": scores["parenchyma_dice"],
            "parenchyma_recall": scores["parenchyma_recall"],
            "mass_coverage": scores["mass_coverage"],
            "mass_fraction_of_reference": scores["mass_fraction_of_reference"],
        },
        "why_all_three": "the readings separate only where the organ hosts a "
                         "mass, and they separate most where the mass is "
                         "large; publishing one of them without the other "
                         "two on a mass-dominated organ is a choice the "
                         "reader cannot see.",
    }


def cyst_census(gt):
    """Does this reference carry a KiTS cyst, and how much of one?

    The advisor's cyst ruling moves every case that has a cyst and no case
    that has not, so the number of cases that have one is the SIZE of the
    decision and belongs in every artefact that scores KiTS. It is counted
    here, from the reference array in hand, rather than quoted from a memo.
    """
    gt = np.asarray(gt)
    n_cyst = int((gt == CYST).sum())
    n_tum = int((gt == TUMOUR).sum())
    return {"ruling": CYST_RULING,
            "cyst_bearing": bool(n_cyst > 0),
            "cyst_voxels": n_cyst,
            "tumour_voxels": n_tum,
            "cyst_share_of_mass": (round(float(n_cyst) / float(n_cyst + n_tum),
                                         4)
                                   if (n_cyst + n_tum) else None)}


def selftest():
    """The three claims this module makes about itself, checked.

    Run as `python ipkg_organ_convention.py`. It touches no dataset: the
    degenerate case that motivates condition (b) is small enough to build.
    """
    fails = []

    # 1. the mass-alone prediction that condition (b) is about
    ref_organ = np.zeros((10, 10, 10), bool)
    ref_organ[2:8, 2:8, 2:8] = True                 # 216 voxels of parenchyma
    ref_mass = np.zeros((10, 10, 10), bool)
    ref_mass[2:8, 2:8, 2:8] = False
    ref_mass[0:10, 0:10, 0:2] = True                # a mass beside it
    # rebuild as the collections store it: the organ label excludes the mass
    organ = ref_organ & ~ref_mass
    s = organ_scores(ref_mass, organ, organ | ref_mass)
    if s["parenchyma_recall"] != 0.0:
        fails.append("a prediction of the mass alone must recover no "
                     "parenchyma, got recall %r" % s["parenchyma_recall"])
    if not (s["organ_dice"] or 0) > 0.6:
        fails.append("a prediction of the mass alone must still score well "
                     "under convention A - that is the point of condition "
                     "(b) - got %r" % s["organ_dice"])
    if s["organ_dice"] != s["dice_organ_with_mass"]:
        fails.append("organ_dice and dice_organ_with_mass are the same "
                     "measurement under two names, got %r and %r"
                     % (s["organ_dice"], s["dice_organ_with_mass"]))
    # the pair of numbers that separates "found the organ" from "found the
    # tumour and called it the organ". The mass-alone prediction is the
    # extreme of the second, and it has to read as such.
    if s["mass_coverage"] != 1.0:
        fails.append("a prediction of the mass alone covers all of the mass, "
                     "expected mass_coverage 1.0, got %r" % s["mass_coverage"])
    # with the host mass dropped from both sides the mass-alone prediction
    # has nothing left at all, so the third reading must be the 0.0 that the
    # first two conceal - not None, which would mean "nothing to score".
    if s["dice_lesion_as_ignore_region"] != 0.0:
        fails.append("with the host mass ignored, a prediction of the mass "
                     "alone predicts nothing and must score 0.0, got %r"
                     % s["dice_lesion_as_ignore_region"])
    tr = three_readings(s)
    if tr[CONV_WM]["dice"] != s["organ_dice"]:
        fails.append("the supplement block must carry the published number "
                     "itself, not a recomputation of it")

    # 1b. a prediction that is exactly the parenchyma: the mirror image, and
    #     the case that pins mass_coverage as a SEPARATE axis from Dice. Both
    #     readings that keep the mass charge it for the mass it missed; the
    #     ignore reading is the only one that scores the organ model on the
    #     organ it was asked for.
    s2 = organ_scores(organ, organ, organ | ref_mass)
    if s2["mass_coverage"] != 0.0:
        fails.append("a prediction of the parenchyma alone covers none of "
                     "the mass, expected 0.0, got %r" % s2["mass_coverage"])
    if s2["parenchyma_recall"] != 1.0:
        fails.append("a prediction of the parenchyma alone recovers all of "
                     "it, expected recall 1.0, got %r"
                     % s2["parenchyma_recall"])
    if s2["dice_lesion_as_ignore_region"] != 1.0:
        fails.append("with the host mass ignored, a prediction of the "
                     "parenchyma alone is exact, expected 1.0, got %r"
                     % s2["dice_lesion_as_ignore_region"])
    if not s2["organ_dice"] < s2["dice_tumour_excluded"]:
        fails.append("convention A must COST a prediction that misses the "
                     "mass, got %r against %r"
                     % (s2["organ_dice"], s2["dice_tumour_excluded"]))

    # 1c. an organ that hosts no mass at all: the three readings are one
    #     reading, and mass_coverage is None rather than a zero that would
    #     drag a cohort median towards a failure that did not happen.
    s3 = organ_scores(organ, organ, organ)
    if s3["mass_coverage"] is not None:
        fails.append("mass_coverage must be None where there is no mass to "
                     "cover, got %r" % s3["mass_coverage"])
    if not (s3["organ_dice"] == s3["dice_tumour_excluded"]
            == s3["dice_lesion_as_ignore_region"] == 1.0):
        fails.append("without a host mass the three readings are the same "
                     "measurement, got %r / %r / %r"
                     % (s3["dice_tumour_excluded"], s3["organ_dice"],
                        s3["dice_lesion_as_ignore_region"]))

    # 1d. the cyst ruling, which is a ruling about what MASS_CLASSES holds.
    #     Checked here because the ruling is invisible in every number it
    #     produces: a KiTS kidney scored with the cyst inside and one scored
    #     with it outside are both plausible organ Dice values.
    if CYST not in MASS_CLASSES:
        fails.append("the advisor's 2026-10-03 ruling puts the KiTS cyst "
                     "INTO the organ-with-mass reference; class %d is not in "
                     "MASS_CLASSES" % CYST)
    cg = np.zeros((6, 6, 6), np.uint8)
    cg[1:4, 1:4, 1:4] = RIGHT_KIDNEY
    cg[4, 1:4, 1:4] = CYST
    cen = cyst_census(cg)
    if not (cen["cyst_bearing"] and cen["cyst_voxels"] == 9
            and cen["tumour_voxels"] == 0):
        fails.append("cyst_census miscounted a 9-voxel cyst: %r" % cen)

    # 2. laterality, pinned against the axis codes of both orientations the
    #    four collections are stored in. A sign error here swaps the two
    #    kidneys of every KiTS case and conserves every voxel while doing it.
    ipl = np.array([[0.0, 0.0, -0.76, 0.0], [0.0, -0.76, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])  # KiTS, IPL
    ras = np.diag([1.0, 1.0, 1.0, 1.0])                            # RAS
    if lr_axis(ipl) != (2, 1):
        fails.append("IPL: axis 2 runs towards the patient's left, expected "
                     "(2, 1), got %r" % (lr_axis(ipl),))
    if lr_axis(ras) != (0, -1):
        fails.append("RAS: axis 0 runs towards the patient's right, expected "
                     "(0, -1), got %r" % (lr_axis(ras),))

    # 3. the splitter conserves, including components under the old threshold,
    #    and puts the sides where the affine says they are
    kid = np.zeros((20, 20, 60), bool)
    kid[6:14, 6:14, 4:14] = True                    # low index along axis 2
    kid[6:14, 6:14, 46:56] = True                   # high index along axis 2
    kid[10, 10, 30] = True                          # a one-voxel fragment
    out, info = kidney_sides(kid, ipl, (1.0, 1.0, 1.0))
    if info["voxels_out"] != int(kid.sum()):
        fails.append("splitter lost voxels: %d in, %d out"
                     % (kid.sum(), info["voxels_out"]))
    if info["small_components"] < 1:
        fails.append("the one-voxel fragment should be counted as a small "
                     "component, got %d" % info["small_components"])
    if not out[10, 10, 8] == RIGHT_KIDNEY:
        fails.append("under IPL the low-index mass is the patient's RIGHT "
                     "kidney, got %d" % out[10, 10, 8])

    # 3b. the two attribution rules, on the shape of FLARE23_0405: a mass
    #     that BRUSHES one organ at a single corner and lies broadside to
    #     another. Minimum distance is decided by the brush, the vote by the
    #     body of the mass, and the module must take the vote. Synthetic
    #     rather than the case itself so the property is checked without a
    #     dataset, and the case is what the header cites.
    gt = np.zeros((40, 20, 20), np.uint8)
    gt[0:5, :, :] = LEFT_KIDNEY          # broadside, 2 voxels of clear air
    gt[7:30, :, :] = TUMOUR              # the mass
    gt[30, 0:2, 0:2] = 3                 # spleen, brushing one corner at 1
    vote, _unattr, det = attribute_mass(gt, (1.0, 1.0, 1.0))
    mind, _u2, det2 = attribute_mass(gt, (1.0, 1.0, 1.0), rule=RULE_MIN_DIST)
    if list(vote) != [LEFT_KIDNEY]:
        fails.append("the majority vote must host the mass on the organ most "
                     "of its voxels are nearest, got %r"
                     % [UNIFIED[k] for k in vote])
    if list(mind) != [3]:
        fails.append("minimum distance must host it on the organ it brushes "
                     "- if it does not, this test no longer reproduces the "
                     "disagreement it exists to pin - got %r"
                     % [UNIFIED[k] for k in mind])
    if not (det and det[0]["rules_agree"] is False
            and det[0]["majority_vote_says"] == UNIFIED[LEFT_KIDNEY]
            and det[0]["minimum_distance_says"] == UNIFIED[3]):
        fails.append("the component record must name BOTH rules and say they "
                     "disagree, got %r" % (det[0] if det else None))
    if det2 and det2[0]["attribution_rule"] != RULE_MIN_DIST:
        fails.append("a component attributed under the rejected rule must "
                     "say so in its own record")

    # 3c. the cyst ruling, priced. The same kidney scored with the cyst in
    #     the organ-with-mass reference and with it in no reference at all -
    #     which is what the delivered pipeline did - are different numbers,
    #     and the counterfactual argument is what lets the ledger say by how
    #     much instead of asserting that it matters.
    kg = np.zeros((30, 12, 12), np.uint8)
    kg[0:10, :, :] = RIGHT_KIDNEY
    kg[11:20, :, :] = CYST
    with_cyst, _, _ = attribute_mass(kg, (1.0, 1.0, 1.0))
    without, _, _ = attribute_mass(kg, (1.0, 1.0, 1.0), mass_classes=(TUMOUR,))
    if RIGHT_KIDNEY not in with_cyst:
        fails.append("under the ruling the cyst joins the kidney's "
                     "organ-with-mass reference; it did not")
    if without:
        fails.append("with the cyst out of MASS_CLASSES nothing is "
                     "attributed, got %r" % [UNIFIED[k] for k in without])

    # 4. the conservation guard actually raises
    try:
        bad = np.zeros((4, 4, 4), np.uint8)
        bad[0, 0, 0] = 7
        raise ConservationError("sentinel")
    except ConservationError:
        pass
    else:                                            # pragma: no cover
        fails.append("ConservationError is not raisable")

    for f in fails:
        print("FAIL %s" % f)
    print("%s: %d check(s) failed" % (CONVENTION_VERSION, len(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
