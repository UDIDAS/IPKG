"""
ipkg_ui_walk.py
===============
Walk the four IPKG case pages in a real browser and assert, against the text
the browser actually renders, that the 2026-09-29 provenance audit reached the
screen.

WHY A UI HARNESS AND NOT A UNIT TEST. Every change the audit made is a change
to what a clinician READS. The sub-site correction, the semi-oracle
disclosure, the burden sentence and the rater page's freedom from accuracy
scores are all properties of one rendered page, assembled from five modules
by a Streamlit run: ipkg_provenance writes the disclosure, segx_kg resolves
the correction, segx_clinical draws the phenotype panel, segx_testcase lays
out the four panels and seg_explorer decides - via RATER_VIEW - which of them
a rater is allowed to see. A test that imports any one of those proves that
module returns the right string; it cannot prove the string is on the page, it
cannot prove a second surface has not printed the uncorrected value two panels
further down, and it cannot prove the RATER_VIEW gate is where everybody
thinks it is. Those are exactly the three things that went wrong before, so
the check has to be made where the reader is: in the DOM.

WHAT IT DOES NOT DO. It never starts, stops or restarts a server, and it never
writes anywhere but its own output directory. The v2 app is already running on
127.0.0.1:8506 and hot-reloads on edit; this attaches to it as one more
reader. Everything it produces lands under validation/ui/ - one text dump and
a screenshot series per case, plus ui_walk.txt and ui_walk.json - so a claim
in the report can be traced back to the page it was read off.

THE SCROLL CONTAINER IS NOT THE WINDOW. This application scrolls inside
section[data-testid="stMain"]; window.scrollTo() and page.mouse.wheel() are
silent no-ops against it, which is how an earlier pass at this "captured" four
screenshots of the same viewport and reported a page it had never seen below
the fold. Every scroll here goes through that element (see _scroll), and the
screenshots are a SERIES of viewport captures rather than one full_page shot,
because full_page measures the document - which is one viewport tall no matter
how long the case page is.

READINESS IS A MARKER, NOT A TIMEOUT. Streamlit renders the page in stages and
shows no spinner while it reads the phenotype record, so "the text stopped
changing" is true of a half-drawn page: on case_00067 the text sat at "reading
the phenotype record…" for eleven seconds, perfectly stable, and a
stability-only wait captured those nine lines as the whole case page. _settle
therefore waits for stability AND for a marker that only the finished page
carries, and says so when the marker never arrives instead of asserting
against a fragment.

WHAT IS ASSERTED, and why each one is phrased the way it is - the wording of
an assertion is the part that decides whether a green line means anything:

  A1  the semi-oracle disclosure is on EVERY case page. It is method rather
      than score, so policy decision 2 (no Dice for raters) does not exclude
      it; a rater asked to judge a contour is entitled to know a box drawn
      from the reference outline started it.
  A2  pancreas_125 shows the corrected sub-site WITH its provenance: the
      corrected word, the audit's labelled sentence, and the delivered value
      beside it. Both halves are required. "Head of pancreas" on its own
      would be the silent substitution policy decision 1 forbids, and the
      label on its own would be a note about a value that is not there.
  A3  no page presents "Tail of pancreas" as a bare fact. Not "the word tail
      never appears" - the delivered value is QUOTED on purpose, and a test
      that banned the word would force the app to hide what was delivered.
      The assertion is positional: every occurrence has to sit inside a
      sentence that also says it was corrected, withdrawn, voided or - for
      another patient's delivered record, which is not ours to correct -
      flagged as pending re-issue.
  A3b the same question asked of the retrieval surfaces, which print the
      sub-site as a bare table cell. Separate from A3 because it is a
      different surface with a different owner, and because - see the report
      this harness prints - it is the one that still fails.
  A4  no accuracy figure anywhere on a rater page. Two sub-checks: the metric
      vocabulary (Dice, DSC, IoU, Jaccard, Hausdorff, HD95, ASSD, NSD) and
      the audit's own measured values as literals, because a number can reach
      a page without its name. Retrieval similarity is NOT an accuracy figure
      and is not flagged: it scores this record against another record, never
      either against a reference.
  A5  the burden line does not present a self-referential cut-point as an
      independent finding. pancreas_125's tumour voxel count IS MSD's t2
      cut-point, so "high burden" here is arithmetic about one lesion and not
      a comparison with a cohort, and the page has to say so where the
      number is printed.
  A6  the neighbour panel and the index case are labelled symmetrically:
      human outline against model output, in the same words, on both halves
      of the panel. This one needs a press - the panel opens from a Compare
      CT / Open control - so the harness looks for the control, presses it,
      and reports precisely what it found when there is nothing to press.
  A7  every pool size the page states is the live one, including the one
      surface a file sweep cannot reach: the "comparable studies" tile,
      which segx_testcase draws from pool_metrics() behind an
      @st.cache_resource. That cache is why this assertion has to be made in
      a browser - a server started before a corpus swap serves the old pool
      for the life of the process, with a correct footer printed beside it,
      and on 2026-10-02 it did. The live size is never typed here; it is
      read off ipkg_final_cases, which computes it on every access.
  A8  a withheld lesion claim does not render as a negative finding. The
      page's own verdict legend promises this - "withheld is not negative,
      and N never says the organ is clear" - and A8 holds it to the promise:
      the legend has to be there, and no question resting on a withheld
      claim may answer F.
  A9  no raw diagnostic on a rater's screen: no absolute path from this
      machine, no Python errno, traceback, exception class or source
      location. Added after the drawers were opened for the first time and
      read rather than counted - the Knowledge graph drawer prints an
      [Errno 2] and an absolute Windows path on all four cases. Nothing in
      A1 to A8 could see it, because it is not a wrong number; it is a
      surface that has failed open.

A FAILURE IS NOT NECESSARILY OUR DEFECT AND THE REPORT MUST NOT PRETEND IT
IS. Four states are printed and the last two are the ones that earn this
paragraph:

  PASS     tested, held
  FAIL     tested, did not hold. The page says something it should not or
           omits something it should. Nothing here decides whether that is
           our defect or a delivered one; it prints the text it matched so
           a reader can.
  BLOCKED  the surface could not be REACHED on this deployment, which is a
           fact about the machine and is reported with its reason rather
           than counted as a pass.
  N/A      the surface WAS reached and the thing being asserted about is not
           on it. Added 2026-10-04 because five of the twenty-eight passes
           in the previous run were this and were being counted as green -
           see Result. The summary prints how many assertions were actually
           exercised, because "28 PASS, 0 FAIL" and "23 tested, 5 nothing to
           test" are different reports of the same walk.

Run:
    python ipkg_ui_walk.py                       # all four cases
    python ipkg_ui_walk.py --case pancreas_125   # one case
    python ipkg_ui_walk.py --drawers core        # skip the slow drawers
    python ipkg_ui_walk.py --url http://127.0.0.1:8506
Exit code is 1 when any assertion FAILs, 2 when one is BLOCKED and none
failed, 0 when every assertion passed. N/A does not change the exit code -
it is not a failure - but it does change the headline, which is the point.
"""
import argparse
import json
import os
import re
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "validation", "ui")
URL = "http://127.0.0.1:8506"

# The four cases, by the display id the deep link takes (/?case=<id>), with
# the collection word only so the report reads like the app rather than like
# a list of ids.
CASES = [
    ("volume-76", "Liver - LiTS"),
    ("case_00067", "Kidney - KiTS23"),
    ("pancreas_125", "Pancreas - MSD Pancreas"),
    ("FLARE23_0405", "Multi-organ - FLARE23"),
]

# THE MARKER THAT MEANS "THIS PAGE IS FINISHED". The phenotype panel's own
# heading: it is drawn from the measured record, so it cannot appear before
# the record has been read, and it is on all four cases. Chosen over the case
# id, which the header prints immediately, and over the survey block, which is
# drawn even when the case panels fail.
READY_MARKER = "WHAT THE RECORD STATES"

# The drawers on a case page are st.checkbox widgets, not expanders, so they
# are opened by clicking their label and each click costs a rerun. `core` is
# the three that carry record text - the verdict grid, the other nine
# questions and the report - which is where an uncorrected sub-site or a
# stray Dice would most plausibly be. `all` adds the four that carry no
# record text but are still rater-reachable, and is what the no-score sweep
# should be run with before a build goes out.
DRAWERS_CORE = ("Query verdicts", "Other questions", "Full report")
DRAWERS_ALL = DRAWERS_CORE + ("Knowledge graph", "Graph query", "Scan quality",
                              "Rigid spatial alignment", "Ask a question")

PASSPHRASE_ENV = ("IPKG_PASSPHRASE", "VKG_PASSPHRASE")


# ---------------------------------------------------------------------------
# text matching
# ---------------------------------------------------------------------------
# ONE CHARACTER FOR ONE CHARACTER, so an offset into the normalised text is an
# offset into the text the browser rendered. The page is written with curly
# quotes, en and em dashes, a middle dot as a separator and >= as a glyph; a
# pattern typed on a keyboard matches none of them. Collapsing whitespace
# would be the other half of the normalisation and is deliberately NOT done
# here - it would move every offset after the first line break, and then the
# snippet printed as evidence would not be the text that matched. Patterns
# join their words with \s+ instead (see _rx).
_FOLD = {
    0x2018: "'", 0x2019: "'", 0x201a: "'", 0x201b: "'",
    0x201c: '"', 0x201d: '"', 0x201e: '"',
    0x2013: "-", 0x2014: "-", 0x2015: "-", 0x2212: "-", 0x00ad: "-",
    0x00a0: " ", 0x2007: " ", 0x202f: " ", 0x2009: " ",
    0x00b7: ".", 0x2022: ".", 0x2026: ".",
    0x2265: ">", 0x2264: "<", 0x00b3: "3", 0x00b2: "2",
}


def fold(s):
    """The page's text with its typography folded to ASCII, same length."""
    return (s or "").translate(_FOLD).lower()


def _rx(phrase):
    """A plain phrase as a regex that tolerates the page's line breaks.

    Everything the app prints is one sentence in the source and may be two
    lines in the DOM - a <span> inside a flex row is enough to put a newline
    in innerText - so a literal search for a phrase with a space in it fails
    on text that reads correctly on screen.
    """
    return re.compile(r"\s+".join(re.escape(w) for w in fold(phrase).split()))


def _snip(raw, start, end, before=50, after=90):
    """The matched text with a little context, on one line, for the report."""
    a = max(0, start - before)
    b = min(len(raw), end + after)
    return re.sub(r"\s+", " ", raw[a:b]).strip()


def hits(raw, phrase, before=50, after=90):
    """Every occurrence of `phrase`, as (start, end, snippet)."""
    flat = fold(raw)
    out = []
    for m in _rx(phrase).finditer(flat):
        out.append((m.start(), m.end(), _snip(raw, m.start(), m.end(),
                                              before, after)))
    return out


def rx_hits(raw, pattern, before=50, after=90):
    """Every match of a compiled regex over the folded text."""
    out = []
    for m in pattern.finditer(fold(raw)):
        out.append((m.start(), m.end(), _snip(raw, m.start(), m.end(),
                                              before, after)))
    return out


def _collapse(lines, keep=3):
    """Ten identical rows of evidence, as a count and three examples.

    The packaged top-10 draws one comparison table per neighbour, so a defect
    in that table is found ten times on one page. Printing it ten times does
    not make it ten defects and does make the next assertion unreadable.
    """
    out, seen = [], {}
    for ln in lines:
        key = re.sub(r"\d+", "#", ln)[:120]
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= keep:
            out.append(ln)
    for key, n in seen.items():
        if n > keep:
            out.append("... and %d more row(s) of the same shape" % (n - keep))
    return out


def near(raw, start, end, phrases, window=360):
    """Which of `phrases` appear within `window` characters of a match.

    This is what makes A3 an assertion about context rather than about
    vocabulary: the delivered sub-site word is quoted on purpose, and what
    has to be true is that the quotation is inside a sentence that says so.
    """
    flat = fold(raw)
    a, b = max(0, start - window), min(len(flat), end + window)
    seg = flat[a:b]
    return [p for p in phrases if _rx(p).search(seg)]


# ---------------------------------------------------------------------------
# the browser
# ---------------------------------------------------------------------------
def _main_text(pg):
    try:
        return pg.inner_text('section[data-testid="stMain"]')
    except Exception:                                         # noqa: BLE001
        return ""


# WHAT innerText OF THE MAIN PANE DOES NOT CONTAIN. Three of the panels on a
# case page are custom components in sandboxed iframes - the three planes, the
# 3-D scene and the knowledge graph - and a fourth, the report preview, is one
# too. A document's innerText stops at an iframe boundary, so a sweep that
# reads only the main pane has not read those panels at all, and "no accuracy
# figure anywhere on a rater page" would be a claim about the part of the page
# that happens to be same-document. The frames are read separately and
# appended to the dump under a heading, so the assertions cover them and a
# reader of the dump can still tell which text came from where.
FRAME_HEAD = "\n\n===== component frames (not part of the main pane) =====\n"


def _frame_text(pg):
    """Every child frame's text, longest first, with its url as a heading."""
    out = []
    for fr in pg.frames:
        if fr == pg.main_frame:
            continue
        try:
            t = fr.locator("body").inner_text(timeout=4000)
        except Exception:                                     # noqa: BLE001
            continue
        if t and t.strip():
            out.append("--- frame %s ---\n%s" % (fr.url.split("?")[0][-70:],
                                                 t.strip()))
    return "\n".join(out)


def _js_shot_target(pg, phrase):
    """True when an element carrying `phrase` was scrolled into view.

    element.scrollIntoView() is used rather than the container arithmetic in
    _scroll because it walks EVERY scrollable ancestor, which is what a panel
    inside a fixed-height scrolling container needs; the window-level
    scrollTo that does nothing here is a different API.
    """
    return pg.evaluate(
        "(want) => {"
        " const w = want.toLowerCase();"
        " const it = document.createTreeWalker(document.body, 1);"
        " while (it.nextNode()) { const e = it.currentNode;"
        "  if (e.children.length === 0 &&"
        "      (e.textContent || '').toLowerCase().includes(w)) {"
        "    e.scrollIntoView({block: 'center'}); return true; } }"
        " return false; }", phrase)


def _settle(pg, quiet=2.0, cap=240, marker=None):
    """Wait until the main pane stops changing AND the marker has arrived.

    Returns (text, reason). `reason` is "" on a clean settle, otherwise it
    says which half timed out - which is the difference between "the page is
    slow" and "the page never finished", and the report prints it rather
    than asserting against whatever was on screen when the clock ran out.
    """
    t0, last, stable = time.time(), "", 0.0
    step = 0.5
    while time.time() - t0 < cap:
        cur = _main_text(pg)
        ready = (marker is None) or (marker in cur)
        if cur and cur == last and ready:
            stable += step
            if stable >= quiet:
                return cur, ""
        else:
            stable, last = 0.0, cur
        pg.wait_for_timeout(int(step * 1000))
    cur = _main_text(pg)
    if marker is not None and marker not in cur:
        return cur, ("the page never reached %r in %ds" % (marker, cap))
    return cur, "the page was still changing after %ds" % cap


def _passphrase():
    """The configured phrase. Never a literal in this file: the repo is
    pushed, and serve_ipkg refuses to publish without one configured."""
    for var in PASSPHRASE_ENV:
        v = (os.environ.get(var) or "").strip()
        if v:
            return v
    p = os.path.join(BASE, ".streamlit", "secrets.toml")
    if os.path.isfile(p):
        for line in open(p, encoding="utf-8"):
            if line.strip().startswith("passphrase"):
                return line.split("=", 1)[-1].strip().strip('"').strip("'")
    return ""


# The gate's own sentence. Needed because the ABSENCE of the password field
# is not evidence that the gate is behind us: Streamlit unmounts and remounts
# elements between reruns, so there is a window in which the field is gone and
# the gate is still on screen. Believing that window cost one case of the
# first full run - case_00067 was walked, scrolled, dumped and asserted
# against the doorway, and the five content assertions failed on a page that
# had never opened. "Am I through" is now a POSITIVE test, twice.
GATE_MARK = "passphrase-protected"


def _at_gate(pg):
    """True while the shared-phrase doorway is still on screen."""
    return (pg.locator('input[type="password"]').count() > 0
            or GATE_MARK in _main_text(pg))


def _gate(pg, phrase):
    """Get through the shared-phrase doorway, or say why not.

    The form's submit is a button and pressing Enter in the field submits it,
    but only while the field still has focus - and the input is re-mounted on
    Streamlit's schedule, so a keypress can land on a detached node and do
    nothing at all, silently. So the button is clicked when it is there and
    the keypress is the fallback rather than the method.
    """
    for _ in range(4):
        if not _at_gate(pg):
            return True, ""
        if not phrase:
            return False, ("no passphrase configured: set IPKG_PASSPHRASE or "
                           "put `passphrase` in .streamlit/secrets.toml")
        pw = pg.locator('input[type="password"]')
        if pw.count():
            pw.first.fill(phrase)
            pg.wait_for_timeout(300)
        btn = pg.get_by_role("button", name="Enter", exact=True)
        if btn.count():
            btn.first.click()
        else:
            pg.keyboard.press("Enter")
        for _ in range(120):
            pg.wait_for_timeout(500)
            if not _at_gate(pg):
                # confirmed a second later, because the remount window is
                # about a frame wide and this is the check that was wrong
                pg.wait_for_timeout(1200)
                if not _at_gate(pg):
                    return True, ""
            if "not right" in _main_text(pg):
                break
    return False, ("the gate did not clear after 4 attempts"
                   if "not right" not in _main_text(pg)
                   else "the app rejected the passphrase")


_MAIN_SEL = 'section[data-testid="stMain"]'


def _scroll(pg, dy):
    """Scroll the pane that actually scrolls. See the module docstring."""
    pg.evaluate(
        "(a) => { const m = document.querySelector(a.sel);"
        " if (!m) return; if (a.dy === null) { m.scrollTop = 0; }"
        " else { m.scrollBy(0, a.dy); } }",
        {"sel": _MAIN_SEL, "dy": dy})


def _scroll_top(pg):
    _scroll(pg, None)


def _walk_down(pg, shots=None, stem="", step=760, limit=40, max_shots=4):
    """Scroll the case page from top to bottom, capturing as it goes.

    The scroll is not only for the screenshots: the 3-D scenes and the
    coronal viewers are iframes that mount when they come into view, so a
    page that has never been scrolled has never drawn half of itself, and
    text inside those components is not in innerText until it has.

    `max_shots` spaces the captures over the page rather than taking one per
    step. A case page with its drawers open is thirty viewports long and
    thirty near-identical PNGs is not evidence, it is a directory nobody
    opens; the spacing is computed from the pane's own scrollHeight so the
    series covers the whole page whatever its length.
    """
    _scroll_top(pg)
    pg.wait_for_timeout(400)
    try:
        total = pg.evaluate(
            "(a) => { const m = document.querySelector(a.sel);"
            " return m ? Math.ceil(m.scrollHeight / a.step) : 1; }",
            {"sel": _MAIN_SEL, "step": step})
    except Exception:                                         # noqa: BLE001
        total = limit
    every = max(1, -(-min(int(total or 1), limit) // max(1, max_shots)))
    n = 0
    while n < limit:
        if shots is not None and stem and n % every == 0:
            shots.append(_shot(pg, "%s_scroll_%02d" % (stem, n + 1)))
        at_end = pg.evaluate(
            "(sel) => { const m = document.querySelector(sel);"
            " if (!m) return true;"
            " return m.scrollTop + m.clientHeight >= m.scrollHeight - 4; }",
            _MAIN_SEL)
        if at_end:
            break
        _scroll(pg, step)
        pg.wait_for_timeout(320)
        n += 1
    return n + 1


def _shot(pg, name):
    """One viewport capture into validation/ui/, returning its path."""
    p = os.path.join(OUT_DIR, "%s.png" % name)
    try:
        pg.screenshot(path=p)
    except Exception:                                         # noqa: BLE001
        return ""
    return p


def _shot_at(pg, phrase, name):
    """Scroll the element carrying `phrase` into view and capture it."""
    try:
        ok = _js_shot_target(pg, phrase)
    except Exception:                                         # noqa: BLE001
        ok = False
    if not ok:
        return ""
    pg.wait_for_timeout(500)
    return _shot(pg, name)


def _open_details(pg):
    """Open every <details> on the page, and say how many were shut.

    A CLOSED <details> IS IN THE DOM AND NOT IN innerText, which makes it the
    one place a rater-reachable sentence can hide from a text sweep. The
    provenance drawer under the disclosure is a <details> - deliberately, so
    that opening it costs no rerun - and so is every "6 reasoning paths"
    expander in the retrieval panel. Setting .open is a DOM property change,
    not a click: no Streamlit rerun, no widget state touched, nothing the
    server can even see. So the assertions read what a rater would read after
    opening them, rather than what a rater sees before they do.
    """
    try:
        return pg.evaluate(
            "() => { let n = 0;"
            " document.querySelectorAll('details').forEach(d => {"
            "   if (!d.open) { d.open = true; n++; } });"
            " return n; }")
    except Exception:                                         # noqa: BLE001
        return 0


def _open_drawers(pg, wanted, marker=None):
    """Tick the record drawers, so the assertions see what a reader can see.

    They are st.checkbox widgets: each click is a rerun, the ticked state
    survives it, so the text accumulates and one dump at the end covers all
    of them. Each is given its own budget and a failure to open is RECORDED
    rather than raised - a drawer that loads a sentence encoder may simply
    not finish on a busy machine, and that is a fact about the run, not an
    assertion result.
    """
    opened, missed = [], []
    for label in wanted:
        loc = pg.locator('[data-testid="stCheckbox"] label', has_text=label)
        if loc.count() == 0:
            missed.append((label, "no such drawer on this page"))
            continue
        t0 = time.time()
        try:
            loc.first.click()
        except Exception as e:                                # noqa: BLE001
            missed.append((label, "could not be clicked (%s)"
                           % type(e).__name__))
            continue
        _, why = _settle(pg, quiet=2.0, cap=150, marker=marker)
        if why:
            missed.append((label, why))
        else:
            opened.append((label, time.time() - t0))
    return opened, missed


# ---------------------------------------------------------------------------
# the assertions
# ---------------------------------------------------------------------------
class Result(object):
    """One assertion's verdict, with the text it matched.

    `state` is "PASS", "FAIL", "BLOCKED" or "N/A". Evidence is a list of
    strings and is printed whichever way the verdict went: a passing
    assertion that cannot show what it matched is indistinguishable from one
    that matched nothing, which is how a green harness comes to mean nothing
    at all.

    N/A WAS ADDED ON 2026-10-04 AND IT IS THE POINT OF THAT REVISION. The
    walk before it reported 28 PASS and 0 FAIL, and five of those passes
    were assertions about a surface that is not on the page at all: "no
    retrieval surface on this page prints a sub-site" on three cases, and
    "no pancreatic sub-site word is published on this page" plus "no
    sub-site noun phrase on this page" on FLARE23_0405. Each is a true
    sentence and none of them is a test - there was nothing there to be
    right or wrong about. Counted as PASS they inflate the green, and worse,
    they hide the day a surface disappears: a regression that dropped the
    retrieval table would turn a real assertion into a vacuous one and the
    summary would not move by a single digit.

      PASS     the assertion was tested and held
      FAIL     the assertion was tested and did not hold
      BLOCKED  the surface could not be REACHED on this deployment, which is
               a fact about the machine
      N/A      the surface WAS reached and the thing the assertion is about
               is not on it, so nothing was tested
    """

    def __init__(self, case, name, state, evidence=None, why=""):
        self.case, self.name, self.state = case, name, state
        self.evidence = list(evidence or [])
        self.why = why

    def as_dict(self):
        return {"case": self.case, "assertion": self.name, "state": self.state,
                "why": self.why, "evidence": self.evidence}


# A1 ------------------------------------------------------------------------
DISCLOSURE_HEAD = "Before you judge these contours"
DISCLOSURE_KEYS = ("How these contours were made",
                   "given a box around each structure's reference outline",
                   "the starting boxes are the reference's",
                   "not what it would produce unaided on a new scan")


def a1_disclosure(case, raw):
    """The semi-oracle disclosure, on every case page."""
    ev, missing = [], []
    h = hits(raw, DISCLOSURE_HEAD, before=0, after=120)
    if h:
        ev.append("heading: %s" % h[0][2])
    else:
        missing.append(DISCLOSURE_HEAD)
    for k in DISCLOSURE_KEYS:
        g = hits(raw, k, before=0, after=40)
        if g:
            ev.append("clause: %s" % g[0][2])
        else:
            missing.append(k)
    # The recounted slice numbers under the paragraph are what make it
    # checkable rather than merely stated, so their absence is worth a line
    # even though the disclosure itself is the assertion.
    g = rx_hits(raw, re.compile(
        r"on this study that is \d+ outlined structures?,\s*drawn on \d+ of "
        r"the \d+ slices"))
    if g:
        ev.append("recount: %s" % g[0][2])
    else:
        ev.append("NOTE the per-study slice recount under the paragraph is "
                  "not on this page")
    if missing:
        return Result(case, "A1 semi-oracle disclosure on the rater page",
                      "FAIL", ev,
                      "missing from the page: " + "; ".join(missing))
    return Result(case, "A1 semi-oracle disclosure on the rater page",
                  "PASS", ev)


# A2 ------------------------------------------------------------------------
AUDIT_LABEL = "corrected by IPKG audit 2026-09-29, confirmed by the pipeline team's re-issue 2026-10-01"
SITE_WORDS = ("Head of pancreas", "Body of pancreas", "Tail of pancreas")


def a2_subsite(case, raw):
    """pancreas_125 shows the corrected sub-site AND says where it came from.

    On the other three cases the same assertion is the negative one: the
    delivered rule wrote a pancreatic sub-site on the LiTS liver and the
    KiTS kidney too, and the overlay voids those - so a pancreatic sub-site
    word may not be published as a fact on a liver or a kidney page either.
    """
    name = "A2 corrected sub-site shown with its provenance"
    if case == "pancreas_125":
        ev, missing = [], []
        for want, what in ((("Head of pancreas",), "the corrected word"),
                           (("corrected",), "a correction marker"),
                           ((AUDIT_LABEL,), "the audit's labelled sentence"),
                           (("the delivered record says",),
                            "the delivered value beside it")):
            got = None
            for w in want:
                g = hits(raw, w)
                if g:
                    got = g[0][2]
                    break
            if got:
                ev.append("%s: %s" % (what, got))
            else:
                missing.append("%s (%s)" % (what, " / ".join(want)))
        # The delivered word has to be the one that is quoted, not merely
        # some word: a label that says "the delivered record says" and then
        # prints the corrected value would read as provenance and be none.
        g = rx_hits(raw, re.compile(r"the delivered record says\s*[\"']?tail"))
        if g:
            ev.append("delivered value quoted verbatim: %s" % g[0][2])
        else:
            missing.append('the delivered word "tail" quoted as delivered')
        if missing:
            return Result(case, name, "FAIL", ev,
                          "missing: " + "; ".join(missing))
        return Result(case, name, "PASS", ev)

    # liver / kidney / multi-organ: a pancreatic sub-site must not be
    # asserted, and where the withdrawal is stated that is what we want to
    # see rather than silence.
    ev, bad = [], []
    for w in SITE_WORDS:
        for s, e, snip in hits(raw, w):
            ctx = near(raw, s, e, ("withdrawn", "does not apply",
                                   "corrected by IPKG audit", "no sub-site"))
            if ctx:
                ev.append("withdrawn, as it should be: %s" % snip)
            else:
                bad.append(snip)
    if bad:
        return Result(case, name, "FAIL", ev + ["BARE: %s" % b for b in bad],
                      "a pancreatic sub-site is stated as a fact on a page "
                      "whose organ has none")
    if not ev:
        # Nothing on the page to be right or wrong about. See Result.
        return Result(case, name, "N/A",
                      ["no pancreatic sub-site word is published on this "
                       "page, so there was no occurrence to judge"],
                      "the surface this assertion is about is not on this "
                      "page; it is tested on volume-76, case_00067 and "
                      "pancreas_125, where the word does appear")
    return Result(case, name, "PASS", ev)


# A3 ------------------------------------------------------------------------
# WHAT COUNTS AS SAYING SO. Four surfaces say it in four registers and all
# four are legitimate: the phenotype row corrects ("corrected by IPKG audit
# ... the delivered record says 'tail'"), the provenance drawer withdraws the
# code ("withdrawn - no replacement concept"), the organ gate voids ("does not
# apply to the liver"), and the graph's reasoning paths - which describe OTHER
# patients' delivered records and so are not ours to correct - flag ("sub-site
# pending re-issue - IPKG audit 2026-09-29").
#
# "pending re-issue" was NOT on this list on the first run with the <details>
# drawers open, and A3 duly reported 40 bare facts on pancreas_125. Every one
# of them was a reasoning path that did carry a marker, in a wording this list
# did not know. That is worth writing down rather than quietly widening: an
# assertion phrased from the code rather than from the screen fails on wording
# and calls it a defect, and a reader who trusts it once will not trust it
# again.
CORRECTION_CONTEXT = ("corrected", "withdrawn", "the delivered record says",
                      "does not apply", "confirmed by the pipeline team's re-issue",
                      "pending the pipeline team re-issue",
                      "pending re-issue", "pending the pipeline team confirmation",
                      "as delivered")


def a3_no_bare_tail(case, raw):
    """No page presents a pancreatic sub-site noun phrase as a bare fact."""
    name = "A3 no sub-site noun phrase presented as a bare fact"
    ev, bad = [], []
    for w in SITE_WORDS:
        for s, e, snip in hits(raw, w):
            ctx = near(raw, s, e, CORRECTION_CONTEXT)
            if ctx:
                ev.append("%r carries %s -- %s" % (w, "/".join(ctx), snip))
            else:
                bad.append("%r with no correction context: %s" % (w, snip))
    # collapsed for the same reason A3b is: the ten packaged neighbours put
    # forty near-identical reasoning paths on one page, and forty lines of
    # "this one is fine too" is not more convincing than three and a count
    if bad:
        return Result(case, name, "FAIL", _collapse(ev) + _collapse(bad),
                      "%d occurrence(s) stand as bare facts" % len(bad))
    if not ev:
        return Result(case, name, "N/A",
                      ["no sub-site noun phrase on this page, so there was "
                       "no occurrence to judge"],
                      "the surface this assertion is about is not on this "
                      "page")
    return Result(case, name, "PASS", _collapse(ev))


# THE RETRIEVAL SURFACES PRINT THE SUB-SITE AS A TABLE CELL, which is a
# different question from A3 and gets its own assertion: a cell has no room
# for a sentence, so a correction either reaches the value or it does not
# reach that surface at all. The pinned top-10 renders
# "pancreas <tab> pancreatic sub-site <tab> tail <tab> tail <tab> ok", the
# console's rows render "differing: pancreas pancreatic sub-site (tail here,
# body there)", and the first value in both is THIS study's.
_CELL = re.compile(r"pancreatic sub-site\s+(head|body|tail)\s+(head|body|tail)")
_HERE = re.compile(r"pancreatic sub-site \((head|body|tail) here,")
_NEIGH_PHRASE = re.compile(r"(head|body|tail) of the pancreas")


def a3b_retrieval_cells(case, raw, expect=None):
    """The retrieval surfaces' own sub-site values, for THIS study.

    `expect` is the corrected word where the audit has one; the assertion is
    that a cell describing this study does not print a word the same page has
    already corrected. Neighbours' values are reported too but not failed on:
    they are other studies' delivered records, and correcting the whole pool
    on a rater's screen is a re-issue and not a caption. (This sentence said
    "a pool of 1,334" until 2026-10-04; the pool is 1,311 and has moved six
    times, so it is no longer written down here at all - A7 reads the live
    size off ipkg_final_cases and off the page.)
    """
    name = "A3b retrieval surfaces report this study's sub-site corrected"
    ev, bad = [], []
    for s, e, snip in rx_hits(raw, _CELL, before=30, after=40):
        word = _CELL.search(fold(raw)[s:e]).group(1)
        if expect and word != expect:
            bad.append("comparison row gives this study %r where the page "
                       "shows %r: %s" % (word, expect, snip))
        else:
            ev.append("comparison row: %s" % snip)
    for s, e, snip in rx_hits(raw, _HERE, before=40, after=40):
        word = _HERE.search(fold(raw)[s:e]).group(1)
        if expect and word != expect:
            bad.append("difference line gives this study %r where the page "
                       "shows %r: %s" % (word, expect, snip))
        else:
            ev.append("difference line: %s" % snip)
    # THIS NOTE USED TO ASSERT A MECHANISM IT NEVER CHECKED.
    #
    # It read: "N retrieved-study description(s) print a sub-site from the
    # delivered pool record, uncorrected". A browser cannot see which record
    # a word came from, and nothing here looked: the sentence was a fixed
    # string appended after a regex count, and it rode inside a PASSing
    # assertion for weeks. It was also false. segx_queries draws those rows
    # from corrected_pool(), so the word on the card is the RESOLVED one -
    # the discriminator is on the page itself, where the one neighbour whose
    # sub-site the rule withdrew prints no word at all while a delivered
    # record would have printed one. The claim then propagated verbatim into
    # cutover/OUTSTANDING_AT_CUTOVER.md, with the count wrong as well.
    #
    # A harness that writes down a mechanism it has not measured is worse
    # than one that stays quiet, because the sentence gets quoted. So this
    # now records only what was counted, and the provenance question is
    # asked properly in A3c, where it can be answered.
    n_neigh = len(rx_hits(raw, _NEIGH_PHRASE))
    if n_neigh:
        ev.append("%d retrieved-study description line(s) print a pancreatic "
                  "sub-site; they are other patients and are not failed on "
                  "here. Which record supplied each word is NOT readable "
                  "from the DOM and is not claimed - see A3c" % n_neigh)
    if bad:
        # THE TEN ROWS OF THE PACKAGED TOP-10 SAY THE SAME THING TEN TIMES,
        # and ten identical lines of evidence bury the seventh assertion
        # under the fourth. The count is the finding; three rows are enough
        # to show what the cell looks like.
        return Result(case, name, "FAIL", ev + _collapse(bad),
                      "the correction layer has not reached the retrieval "
                      "surfaces")
    if not ev:
        # THE ONE THAT MATTERS MOST. This assertion is the reason A3b was
        # written - the retrieval cells were the surface the correction had
        # NOT reached - and on three of the four cases there is no such cell
        # to read. Reported as a pass it looked like three quarters of a
        # green; it is one case tested and three not applicable.
        return Result(case, name, "N/A",
                      ["no retrieval surface on this page prints a "
                       "sub-site, so no cell was judged"],
                      "this assertion is live on pancreas_125 only, where "
                      "the corrected word has somewhere to appear")
    return Result(case, name, "PASS", _collapse(ev))


# A4 ------------------------------------------------------------------------
# THE METRIC VOCABULARY. "semi-oracle" is deliberately absent: it names the
# method and belongs on the rater page. So is "similarity" - S is this record
# against another record, never either against a reference, and calling it an
# accuracy figure would be the same category error the audit was raised to
# fix, in the opposite direction.
METRIC_WORDS = (r"\bdice\b", r"\bdsc\b", r"\bjaccard\b", r"\biou\b",
                r"\bhausdorff\b", r"\bhd95\b", r"\bassd\b", r"\bnsd\b",
                r"\bsurface distance\b", r"\bfalse negative rate\b")
METRIC_NUMBER = re.compile(
    r"\b(?:accuracy|accurate to|overlap|agreement with (?:the )?reference)"
    r"\s*(?:score)?\s*[:=]?\s*0?\.\d{2,}")
# The audit's own figures, as literals. A number can reach a page without its
# name - a chip, a tooltip, a table cell - and these are the ones that would:
# the six organ Dice measured on 09-29, the pipeline team's two reported tumour figures for
# the cases that carry them, the recomputed pancreas_125 value, the
# as-delivered headline and the box-alone ablation.
#
# EXTENDED 2026-10-04, AND THE REASON IS THE DEFECT THIS LIST IS FOR. A
# hand-typed list of literals goes stale the week the numbers move, and then
# the sweep reports "none of the N audit figures appear" about figures that
# no longer exist anywhere - a sentence that is true, reassuring and empty.
# Four sets moved this week and every one of them is a number that must not
# reach a rater:
#   * the FLARE23 twin-clean cohort is n = 14, not 47: raw 0.2193, gated
#     0.2565. The figures they REPLACE - 0.2890 and 0.3215 - are more
#     dangerous than the new ones, because a stale cache or an unrebuilt
#     figure is exactly how they would come back, so both pairs are here;
#   * 0.904 is WITHDRAWN - the FLARE23 tumour figure on n = 8, of which 4
#     are train and 1 val. A withdrawn number on a page is worse than a
#     wrong one, because it was once correct and still looks it;
#   * the retrieval mAP moved 0.972 -> 0.967 under the corrected gold
#     standard. Both, for the same reason as the pair above.
AUDIT_NUMBERS = ("0.945", "0.811", "0.867", "0.934", "0.889", "0.979",
                 "0.921", "0.983", "0.9315", "0.8610", "0.682", "0.537",
                 # FLARE23 twin-clean, n = 14 (current) and n = 47 (retired)
                 "0.2193", "0.2565", "0.2890", "0.3215",
                 # withdrawn: n = 8, 4 train + 1 val
                 "0.904",
                 # retrieval mAP, corrected gold standard and the old one
                 "0.967", "0.972")


_NEIGH_CARD = re.compile(r"(?m)^\s*\d+\.\s+(\S+)\s+[·|-]")


def a3c_withheld_subsite(case, raw):
    """A withheld pancreatic sub-site is not reported as absent.

    THE SAME DEFECT AS A WITHHELD LESION CLAIM RENDERED AS F, one surface
    along. A neighbour card's sub-site clause is blank for two unrelated
    reasons - the collection never recorded one, or the precedence rule
    WITHDREW a word the delivery did supply - and the card gave a single
    sentence to both, so a rater was told a lesion has no sub-site when what
    we actually have is a measurement we declined to make.

    The expectation comes from the module and the check is made in the DOM,
    which is the pattern A7 already uses for the pool size: the neighbour
    case ids are read off the page, the resolution is asked what it did to
    each one, and if it withdrew any then the page has to SAY withheld. It
    does not forbid the absent sentence - a page can legitimately carry both
    a withdrawn neighbour and a genuinely unrecorded one - because an
    assertion that over-claims is how the note this replaces went wrong.
    """
    name = "A3c a withheld pancreatic sub-site is not reported as absent"
    ids = _NEIGH_CARD.findall(raw)
    if not ids:
        return Result(case, name, "N/A", [],
                      "no numbered neighbour cards on this page")
    try:
        import ipkg_final_cases as IFC                        # noqa: PLC0415
        import ipkg_scorer_fixed as SF                        # noqa: PLC0415
        pool = {r.get("case_id"): r for r in IFC.pool_records()}
    except Exception as e:                                    # noqa: BLE001
        return Result(case, name, "BLOCKED",
                      ["%s: %s" % (type(e).__name__, e)],
                      "the resolution could not be read, so what the page "
                      "ought to say is unknown")
    seen, withdrawn = [], []
    for cid in ids:
        rec = pool.get(cid)
        o = ((rec or {}).get("organs") or {}).get("pancreas") or {}
        if not rec or not o:
            continue
        seen.append(cid)
        try:
            r = SF.resolve_site(rec.get("dataset"), cid, "pancreas",
                                o.get("anatomic_location")) or {}
        except Exception:                                     # noqa: BLE001
            continue
        if str(r.get("status") or "").lower() == "voided":
            withdrawn.append("%s (delivered %r, %s)"
                             % (cid, r.get("delivered"),
                                r.get("source") or "no source"))
    ev = ["%d numbered neighbour card(s), %d of them with a pancreas record"
          % (len(ids), len(seen))]
    if not withdrawn:
        ev.append("the rule withdrew none of their pancreatic sub-sites, so "
                  "there is nothing withheld on this page to test")
        return Result(case, name, "N/A", ev,
                      "no withheld sub-site on this page to test the rule "
                      "against")
    ev.append("the rule WITHDREW the pancreatic sub-site on %d of them: %s"
              % (len(withdrawn), "; ".join(withdrawn)))
    said = hits(raw, "WITHHELD, not absent")
    absent = hits(raw, "No sub-site within the pancreas on the retrieved")
    ev.append("the page says 'WITHHELD, not absent' %d time(s) and reports "
              "an absent sub-site %d time(s)" % (len(said), len(absent)))
    if not said:
        return Result(case, name, "FAIL", ev,
                      "a sub-site this rule withdrew is on the page as "
                      "absent, or as nothing at all; withheld is not "
                      "negative and the page has to say which it is")
    return Result(case, name, "PASS", ev)


def a4_no_scores(case, raw):
    """No accuracy figure anywhere a rater can read."""
    name = "A4 no Dice or accuracy score on the rater page"
    ev, bad = [], []
    for pat in METRIC_WORDS:
        for s, e, snip in rx_hits(raw, re.compile(pat)):
            bad.append("metric word %s: %s" % (pat, snip))
    for s, e, snip in rx_hits(raw, METRIC_NUMBER):
        bad.append("scored claim: %s" % snip)
    for num in AUDIT_NUMBERS:
        for s, e, snip in rx_hits(raw, re.compile(re.escape(num))):
            bad.append("audit figure %s on the page: %s" % (num, snip))
    if not bad:
        ev.append("none of %d metric patterns and none of the %d audit "
                  "figures appear in %d characters of page text"
                  % (len(METRIC_WORDS) + 1, len(AUDIT_NUMBERS), len(raw)))
        # Say what IS on the page in the shape of a number, so a reader can
        # see the sweep looked and did not merely fail to look.
        sim = rx_hits(raw, re.compile(r"similarity s\s*[\n\r ]*\d\.\d{4}"),
                      before=0, after=20)
        if sim:
            ev.append("the only four-decimal figures are retrieval "
                      "similarity, e.g. %s" % sim[0][2])
        return Result(case, name, "PASS", ev)
    return Result(case, name, "FAIL", ev + bad,
                  "%d accuracy-bearing string(s) on a rater page" % len(bad))


# A5 ------------------------------------------------------------------------
MSD_CUTPOINT = "3,496"
SELF_REF_KEYS = ("which is this lesion's own count",
                 "is not an independent finding")


def a5_burden(case, raw):
    """The burden word says what it rests on, and pancreas_125's says that
    the cut-point it clears is its own voxel count."""
    name = "A5 burden line does not present a self-referential cut-point " \
           "as a finding"
    ev, missing = [], []
    basis = rx_hits(raw, re.compile(
        r"lesion-(?:size|burden) tercile within [a-z0-9 ]+"), before=20,
        after=200)
    if basis:
        ev.append("basis: %s" % basis[0][2])
    else:
        missing.append("the tercile basis caption under the burden word")
    if case == "pancreas_125":
        cut = hits(raw, MSD_CUTPOINT, before=120, after=200)
        if not cut:
            missing.append("the %s cut-point figure (so the sentence that "
                           "qualifies it cannot be checked)" % MSD_CUTPOINT)
        for k in SELF_REF_KEYS:
            g = hits(raw, k, before=90, after=40)
            if g:
                ev.append("qualifier: %s" % g[0][2])
            else:
                missing.append(k)
        # and the qualifier has to be beside the number, not elsewhere on a
        # 20,000-character page
        if cut and not missing:
            s, e, _ = cut[0]
            ctx = near(raw, s, e, SELF_REF_KEYS, window=320)
            if len(ctx) < len(SELF_REF_KEYS):
                missing.append("the qualifier is on the page but not within "
                               "320 characters of the cut-point figure")
    if missing:
        return Result(case, name, "FAIL", ev, "missing: " + "; ".join(missing))
    return Result(case, name, "PASS", ev)


# A6 ------------------------------------------------------------------------
NEIGHBOUR_KEYS = ("Expert outline (reference)",
                  "every contour in this panel was drawn by a person",
                  "is a MODEL OUTPUT",
                  "How these contours were made")
OPEN_LABELS = ("Compare CT", "Open")
# The half of the symmetry that is on the case page itself, with no press:
# the index panel says the contours are a model's, in the panel heading and
# in the disclosure. Checked separately so a BLOCKED neighbour half still
# reports what WAS verified rather than nothing at all.
INDEX_IS_MODEL = ("what the model found in them",
                  "The contours are the model's")
# Where the neighbour outlines have to be for the control to be drawn at all:
# segx_queries._openable asks segx_caseview.resolve, which looks for
# <preds>/<arm>/<study>__gt_unified.nii.gz under the per-collection
# totalsegmentator__* / vista3d__* arms. A filesystem look is out of
# character for a UI harness and is here for one reason: a BLOCKED line a
# reader has to go and diagnose is a BLOCKED line nobody acts on.
PREDS_DIR = os.path.join(BASE, "mmkg_cache_universal", "preds")


def _why_no_neighbour():
    """One sentence naming what this machine is missing, or ""."""
    try:
        arms = sorted(d for d in os.listdir(PREDS_DIR)
                      if os.path.isdir(os.path.join(PREDS_DIR, d)))
    except OSError:
        return "%s does not exist on this machine" % os.path.relpath(
            PREDS_DIR, BASE)
    outlines = [a for a in arms
                if a.startswith(("totalsegmentator__", "vista3d__"))]
    if outlines:
        return ""
    return ("%s carries only the %d AUSAM arm folder(s) (%s); the neighbour "
            "outlines the control needs are the per-collection "
            "totalsegmentator__* / vista3d__* arms, and none is here, so "
            "segx_caseview.resolve() reports every retrieved study as not on "
            "this machine and segx_queries._openable draws no button"
            % (os.path.relpath(PREDS_DIR, BASE), len(arms), ", ".join(arms)))


def a6_neighbour(case, pg, marker, shots, stem):
    """Press Compare CT and check both panels name whose contour they draw.

    Nothing about this can be read off the case page as it loads: the panel
    exists only after a press. So the harness looks for the control, and when
    there is none it says what it looked for and what it found instead -
    BLOCKED, not PASS, because an assertion about a panel nobody can open has
    not been tested.
    """
    name = "A6 neighbour panel and index case labelled symmetrically"
    here = _main_text(pg)
    # what the index half says about itself, press or no press
    index_ev = []
    for k in INDEX_IS_MODEL:
        g = hits(here, k, before=20, after=60)
        if g:
            index_ev.append("index case names itself a model output: %s"
                            % g[0][2])
    found = []
    for label in OPEN_LABELS:
        loc = pg.get_by_role("button", name=re.compile(re.escape(label)))
        if loc.count():
            found.append((label, loc))
    if not found:
        n_rows = len(rx_hits(here, re.compile(r"shared organs:\s*\S")))
        ev = index_ev + [
            "no button matching %s anywhere on the page"
            % " / ".join(repr(x) for x in OPEN_LABELS),
            "the closest-studies panel did draw %d neighbour row(s), so the "
            "rows are ranked and only the control is missing" % n_rows]
        why_missing = _why_no_neighbour()
        if why_missing:
            ev.append("CAUSE %s" % why_missing)
        return Result(
            case, name, "BLOCKED", ev,
            "the neighbour panel cannot be opened on this build, so the "
            "reference half of the symmetry could not be read off a screen")
    label, loc = found[0]
    try:
        loc.first.scroll_into_view_if_needed()
        loc.first.click()
    except Exception as e:                                    # noqa: BLE001
        return Result(case, name, "BLOCKED",
                      ["%r could not be pressed (%s)" % (label,
                                                         type(e).__name__)],
                      "the control is on the page but did not respond")
    raw, why = _settle(pg, quiet=2.5, cap=240, marker=marker)
    _walk_down(pg, shots, stem + "_neighbour", max_shots=4)
    raw2, _ = _settle(pg, quiet=1.5, cap=90, marker=marker)
    raw = raw2 or raw
    ev, missing = list(index_ev), []
    if not index_ev:
        missing.append("the index panel does not name itself a model output "
                       "(%s)" % " / ".join(INDEX_IS_MODEL))
    for k in NEIGHBOUR_KEYS:
        g = hits(raw, k, before=30, after=110)
        if g:
            ev.append("%s (x%d): %s" % (k, len(g), g[0][2]))
        else:
            missing.append(k)
    # SYMMETRY IS A COUNT, not a presence. The provenance has to be on the
    # strip at the top and on BOTH halves of the panel - the coronal slice
    # and the rotating scene - because a label on the half that happened to
    # have a caption is the defect it was written to fix.
    # WAIT FOR THE THING BEING ASSERTED, NOT FOR QUIET.
    #
    # Both settles above stop when the page has been still for a moment, and
    # on the two largest volumes - volume-76 at 512x512x285 and case_00067 -
    # the rotating 3-D half draws AFTER that lull: the panel goes quiet, this
    # counted 1 of the 3 labels, and A6 failed on pages that were about to be
    # correct. pancreas_125 and FLARE23_0405 passed in the same run because
    # they are smaller. A quiet page is not evidence that a slow panel has
    # finished, so the count is polled to its expected value instead, with a
    # bound - and the bound is what keeps this an assertion: a label that
    # never arrives still fails, and the wait it cost is printed either way
    # so a panel that is quietly getting slower is visible rather than
    # absorbed. Re-running the walk until it went green would have hidden
    # exactly that.
    n = len(hits(raw, "Expert outline (reference)"))
    waited = 0.0
    while n < 3 and waited < 60:
        pg.wait_for_timeout(2000)
        waited += 2.0
        raw3, _ = _settle(pg, quiet=1.0, cap=20, marker=marker)
        raw = raw3 or raw
        n = len(hits(raw, "Expert outline (reference)"))
    if waited:
        ev.append("the reference provenance reached %d label(s) after a "
                  "further %.0fs - the 3-D half draws after the page goes "
                  "quiet on the larger volumes" % (n, waited))
    if n < 3:
        missing.append("the reference provenance appears %d time(s) after "
                       "waiting %.0fs; the strip and both halves of the "
                       "panel make 3" % (n, waited))
    else:
        ev.append("the reference provenance appears %d times: strip, slice "
                  "half and 3-D half" % n)
    if why:
        ev.append("NOTE %s" % why)
    if missing:
        return Result(case, name, "FAIL", ev, "missing: " + "; ".join(missing))
    return Result(case, name, "PASS", ev)


# A7 ------------------------------------------------------------------------
# THE POOL SIZE AS A READER SEES IT, which is not the same question as the
# pool size in the files. check_pool_statements.py sweeps the tree and
# check_figure_pixels sweeps the PNGs; neither can see the number a running
# server actually serves, and that is the one that was wrong on 2026-10-02:
# segx_testcase._kg is behind @st.cache_resource, so a process started
# before a corpus swap serves the OLD pool for the life of that process with
# a correct footer printed beside it. Two different numbers, on one screen,
# both drawn by the same app.
#
# The pool has been re-counted six times in three weeks - 1,425, 1,366,
# 1,347, 1,334, 1,309, 1,311 - so the live value is never typed here. It is
# read from ipkg_final_cases, which computes it as len(pool_records()) on
# every access.
#
# WHAT COUNTS AS A STATEMENT is borrowed from check_pool_statements rather
# than written again: one definition of "this sentence states the pool", or
# the two drift inside a week and the guard that is supposed to end the
# drift becomes a source of it. The st.metric tile is read separately
# because a metric prints its value on its own line with no word of context
# and matches none of those patterns - which is exactly how the 10-02
# defect survived a page dump that contained it in plain sight.
KG_TILE = "comparable studies"
_KG_TILE_RE = re.compile(re.escape(KG_TILE) + r"\s+([\d,]{3,})\b", re.I)


def _live_pool():
    """(size, how) or (None, why not). Never a literal in this file."""
    try:
        sys.path.insert(0, BASE)
        import ipkg_final_cases as IFC                        # noqa: PLC0415
        return int(IFC.pool_size()), ("ipkg_final_cases.pool_size(), which "
                                      "is len(pool_records()) on every "
                                      "access and types nothing")
    except Exception as e:                                    # noqa: BLE001
        return None, "ipkg_final_cases could not be read (%s)" % e


def _pool_statements(raw):
    """Every pool size this page states, via check_pool_statements' patterns.

    That function returns (unit, value, pool_size, offset). `pool_size` is
    already normalised onto the pool convention - "N candidates" is the pool
    minus the query, so it reads as N + 1 - which is the off-by-one the
    module exists to avoid, so the normalised value is what is compared and
    the raw one is what is quoted.
    """
    try:
        sys.path.insert(0, BASE)
        import check_pool_statements as CPS                   # noqa: PLC0415
        out = []
        for unit, value, as_pool, at in CPS.statements(raw):
            out.append((unit, value, as_pool,
                        _snip(raw, at, at + len(str(value)), 70, 70)))
        return out, ""
    except Exception as e:                                    # noqa: BLE001
        return [], "check_pool_statements could not be imported (%s)" % e


def a7_pool(case, raw, pg):
    """No surface on this page states a pool size other than the live one."""
    name = "A7 every pool size on the page is the live one"
    live, how = _live_pool()
    if live is None:
        return Result(case, name, "BLOCKED", [], how)
    ev, bad = ["the live pool is %s (%s)" % ("{:,}".format(live), how)], []
    said, why = _pool_statements(raw)
    if why:
        ev.append("NOTE %s" % why)
    for unit, value, as_pool, txt in said:
        if as_pool == live:
            ev.append("%s statement reads %s: %s"
                      % (unit, "{:,}".format(value), txt[:150]))
        else:
            bad.append("%s statement reads %s (a pool of %s) where the live "
                       "pool is %s: %s"
                       % (unit, "{:,}".format(value),
                          "{:,}".format(as_pool), "{:,}".format(live),
                          txt[:150]))
    # the one tile a text sweep cannot see
    tile = None
    m = _KG_TILE_RE.search(re.sub(r"[ \t]+", " ", raw))
    if m:
        tile = int(m.group(1).replace(",", ""))
        if tile == live:
            ev.append("the %r tile reads %s, drawn from pool_metrics() "
                      "behind @st.cache_resource and not from any caption"
                      % (KG_TILE, "{:,}".format(tile)))
        else:
            bad.append("the %r tile reads %s where the live pool is %s. "
                       "This is the cached-graph defect: the tile comes "
                       "from pool_metrics() inside a @st.cache_resource, "
                       "so a server started before the corpus swap serves "
                       "the old pool for the life of the process"
                       % (KG_TILE, "{:,}".format(tile),
                          "{:,}".format(live)))
    if bad:
        return Result(case, name, "FAIL", ev + bad,
                      "%d surface(s) on this page state a pool size that is "
                      "not the live one" % len(bad))
    if not said and tile is None:
        return Result(case, name, "N/A",
                      ev + ["this page states no pool size and the %r tile "
                            "is not on it" % KG_TILE],
                      "nothing on this page states a pool size, so there "
                      "was no statement to judge")
    if tile is None:
        ev.append("NOTE the %r tile was not on the page; it is the only "
                  "surface whose number comes from pool_metrics() rather "
                  "than from FROZEN_STATE or a caption, so this run did "
                  "not test the cached-graph path" % KG_TILE)
    return Result(case, name, "PASS", ev)


# A8 ------------------------------------------------------------------------
# A WITHHELD CLAIM IS NOT A NEGATIVE FINDING, and the application already
# says so in its own verdict legend: "N not asserted - the lesion claim the
# question rests on was withheld by the claim clean-up, so the answer is
# withdrawn rather than answered; withheld is not negative, and N never says
# the organ is clear."
#
# That legend is a promise to a clinician and this assertion is the only
# thing that holds the page to it. It has two halves and the second is the
# one that earns the N/A column:
#
#   1. the legend is on the page, including the clause that says withheld is
#      not negative. A grid that prints an N with no legend is a letter
#      nobody can read.
#   2. every question whose claim IS withheld answers N and not F. On the
#      shipping build the verdict header counts them - "T 3 . F 13 . U 0 .
#      N 0" - and on all four cases N is zero, because the displayed arm is
#      the IPKG record and the clean-up acts on the pipeline team's round-2 record, which
#      this build does not serve. So the second half is N/A here, and saying
#      PASS would be claiming a test that did not happen. The same assertion
#      against the artefact IS live and IS failing: see ipkg_validate
#      V11.3/claims, where nine questions resting on a withheld claim are
#      answered F and six of them were turned from T into F.
VERDICT_LEGEND = ("not asserted",
                  "withheld is not negative",
                  "never says the organ is clear")
_VERDICT_TALLY = re.compile(r"\bt\s+(\d+)\s*.\s*f\s+(\d+)\s*.\s*u\s+(\d+)"
                            r"\s*.\s*n\s+(\d+)", re.I)


def a8_withheld_not_negative(case, raw):
    name = "A8 a withheld lesion claim does not render as a negative finding"
    ev, missing = [], []
    for k in VERDICT_LEGEND:
        g = hits(raw, k, before=40, after=90)
        if g:
            ev.append("legend: %s" % g[0][2])
        else:
            missing.append(k)
    m = _VERDICT_TALLY.search(fold(re.sub(r"[ \t]+", " ", raw)))
    counts = None
    if m:
        counts = dict(zip(("T", "F", "U", "N"), (int(x) for x in m.groups())))
        ev.append("verdict tally on this page: T %d / F %d / U %d / N %d"
                  % (counts["T"], counts["F"], counts["U"], counts["N"]))
    else:
        ev.append("NOTE the T / F / U / N tally line was not found, so the "
                  "count of withheld answers could not be read")
    if missing:
        return Result(case, name, "FAIL", ev,
                      "the page carries verdicts but not the clause that "
                      "says a withheld claim is not a negative one: "
                      + "; ".join(missing))
    if counts is not None and counts["N"] == 0:
        return Result(
            case, name, "N/A",
            ev + ["the legend is present and correct; no answer on this "
                  "page is withheld, so the half of the assertion that "
                  "matters - that a withheld claim answers N and not F - "
                  "was not exercised",
                  "the same assertion against the artefact IS failing: "
                  "ipkg_validate V11.3/claims, FLARE23_0405, nine "
                  "questions resting on a withheld organ claim answered F"],
            "no withheld claim on this page to test the rule against")
    return Result(case, name, "PASS", ev)


# A9 ------------------------------------------------------------------------
# NO RAW DIAGNOSTIC ON A CLINICIAN'S SCREEN, and this assertion exists
# because the walk that preceded it opened the Knowledge graph drawer on all
# four cases, wrote "drawer 'Knowledge graph' opened in 4s" in its notes, and
# never looked at what was inside it. What is inside it, twice on every case
# page, is this:
#
#     The comparison corpus could not be read: [Errno 2] No such file or
#     directory: '<app-tree>\ipkg_delivered\kg_graphs\
#     corpus_flare23.json'
#
# A Python errno and an absolute path on the author's machine, rendered to a
# rater. It is not a wrong number, so nothing in A1 to A8 could see it; it is
# a surface that has failed open, and a surface that fails open is the one
# thing a reviewer will remember about the build.
#
# WHAT IS NOT FLAGGED. These pages quote artefact names on purpose - the
# whole provenance design is "name the file this came from" - so
# record_containment_v2.json, ipkg_delivered/code/extract_phenotypes.py and
# FLARE23_0405.nii.gz are evidence, not leakage. The patterns below are
# written against the machine rather than the filename: a drive letter with
# two or more separators after it, a unix system root, and Python's own
# error furniture. A relative path a caption chose to print matches none of
# them.
_SEG = r"[\w .$()+-]+"
RAW_DIAGNOSTIC = (
    ("a filesystem path from the machine this is running on",
     re.compile(r"[A-Za-z]:(?:[\\/]" + _SEG + r"){2,}")),
    ("a unix system path",
     re.compile(r"(?<![\w.])/(?:home|mnt|srv|opt|Users|tmp|var)/[\w./-]{3,}")),
    ("a python errno", re.compile(r"\[errno\s*-?\d+\]")),
    ("a python traceback", re.compile(r"traceback \(most recent call last\)")),
    ("a python exception class",
     re.compile(r"\b(?:filenotfound|keyerror|valueerror|typeerror|"
                r"attributeerror|indexerror|oserror|runtimeerror|"
                r"jsondecode)\w*\b")),
    ("a source location", re.compile(r"\.py[\"']?,\s*line\s*\d+")),
)


def a9_no_raw_diagnostics(case, raw):
    """Nothing on the page is addressed to the developer rather than the
    reader."""
    name = "A9 no raw diagnostic on a rater's screen"
    ev, bad = [], []
    for why, rx in RAW_DIAGNOSTIC:
        for s, e, snip in rx_hits(raw, rx, before=70, after=120):
            bad.append("%s: %s" % (why, snip))
    if bad:
        return Result(case, name, "FAIL", _collapse(bad),
                      "%d developer-facing string(s) are rendered on this "
                      "page; a rater reads them" % len(bad))
    ev.append("none of the %d diagnostic patterns appear in %d characters "
              "of page and component-frame text"
              % (len(RAW_DIAGNOSTIC), len(raw)))
    return Result(case, name, "PASS", ev)


# ---------------------------------------------------------------------------
# the walk
# ---------------------------------------------------------------------------
def walk_case(pg, url, case, label, phrase, drawers, want_shots=True):
    """One case page: open it, scroll it, open its drawers, assert."""
    stem = case.replace(".", "_")
    shots, notes = [], []
    t0 = time.time()
    pg.goto("%s/?case=%s" % (url, case), wait_until="domcontentloaded")
    _settle(pg, quiet=1.0, cap=60)
    ok, why = _gate(pg, phrase)
    if not ok:
        return ([Result(case, "A0 the case page opened", "BLOCKED", [], why)],
                shots, notes, "")
    raw, why = _settle(pg, quiet=2.5, cap=300, marker=READY_MARKER)
    if why and _at_gate(pg):
        # One retry, and it is not defensive padding: the gate is the one
        # thing on the way in that can silently swallow a click, and a walk
        # that asserts against the doorway reports five failures that are all
        # the same failure. Retried once, said out loud, and if it still does
        # not open the case is BLOCKED rather than assessed.
        notes.append("the gate was still on screen after the first settle; "
                     "retrying it once")
        ok, why2 = _gate(pg, phrase)
        if ok:
            raw, why = _settle(pg, quiet=2.5, cap=300, marker=READY_MARKER)
        else:
            why = why2 or why
    if why:
        notes.append("first settle: %s" % why)
    # THE PRECONDITION IS ITS OWN ASSERTION. Everything below reads the page
    # text, so a page that never finished loading makes every one of them a
    # statement about nothing; reporting that once, as A0, is the difference
    # between "the disclosure is missing" and "the page did not open".
    if READY_MARKER not in raw:
        if want_shots:
            shots.append(_shot(pg, "%s_00_blocked" % stem))
        txt = os.path.join(OUT_DIR, "%s.txt" % stem)
        with open(txt, "w", encoding="utf-8") as fh:
            fh.write(raw)
        # IS IT THIS CASE THAT DID NOT DRAW, OR IS THIS NOT THE RATER BUILD?
        # `?case=` is KIOSK-ONLY - seg_explorer says so where it reads it, and
        # the research viewer keeps its own pickers and ignores the parameter
        # entirely. A server started without IPKG_KIOSK therefore serves some
        # other module to every case id in turn, each one blocks after the
        # full settle, and all four dumps come out byte-identical and short.
        # That is twenty minutes of looking for a rendering fault that is not
        # there, so the walk says it instead.
        kiosk = _rx(case).search(fold(raw)) or READY_MARKER in raw
        hint = ("" if kiosk else
                " - and `?case=` is not being honoured at all, so this is "
                "probably not the rater build: the research viewer ignores "
                "it. Restart the server with IPKG_KIOSK=1 and re-run")
        return ([Result(case, "A0 the case page opened", "BLOCKED",
                        ["the page text stops at: %s"
                         % re.sub(r"\s+", " ", raw[-160:]).strip(),
                         "the case id %s is %son the page"
                         % (case, "" if kiosk else "NOT ")],
                        (why or "the case panels never drew") + hint)],
                shots, notes, txt)
    if want_shots:
        shots.append(_shot(pg, "%s_00_top" % stem))
    n = _walk_down(pg, shots if want_shots else None, stem)
    notes.append("scrolled the main pane in %d viewport steps" % n)
    raw, why = _settle(pg, quiet=2.0, cap=180, marker=READY_MARKER)
    if why:
        notes.append("settle after scrolling: %s" % why)

    # the drawers, then one dump that covers page and drawers together
    opened, missed = _open_drawers(pg, drawers, marker=READY_MARKER)
    for lab, secs in opened:
        notes.append("drawer %r opened in %.0fs" % (lab, secs))
    for lab, whyd in missed:
        notes.append("drawer %r NOT opened: %s" % (lab, whyd))
    n_det = _open_details(pg)
    if n_det:
        notes.append("opened %d <details> drawer(s) in the DOM, including "
                     "the provenance drawer under the disclosure" % n_det)
    pg.wait_for_timeout(600)
    _walk_down(pg, shots if want_shots else None,
               stem + "_drawers", max_shots=3)
    raw2, why2 = _settle(pg, quiet=2.0, cap=240, marker=READY_MARKER)
    if raw2:
        raw = raw2
    if why2:
        notes.append("settle after the drawers: %s" % why2)

    frames = _frame_text(pg)
    if frames:
        raw = raw + FRAME_HEAD + frames
        notes.append("read %d component frame(s) as well as the main pane"
                     % (frames.count("--- frame ")))
    txt_path = os.path.join(OUT_DIR, "%s.txt" % stem)
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(raw)

    expect = "head" if case == "pancreas_125" else None
    results = [a1_disclosure(case, raw),
               a2_subsite(case, raw),
               a3_no_bare_tail(case, raw),
               a3b_retrieval_cells(case, raw, expect),
               a3c_withheld_subsite(case, raw),
               a4_no_scores(case, raw),
               a5_burden(case, raw),
               a7_pool(case, raw, pg),
               a8_withheld_not_negative(case, raw),
               a9_no_raw_diagnostics(case, raw)]
    # targeted captures for the two panels the report quotes
    if want_shots:
        for phrase_, nm in ((DISCLOSURE_HEAD, "%s_disclosure" % stem),
                            ("WHAT THE RECORD STATES", "%s_record" % stem),
                            (AUDIT_LABEL[:28], "%s_correction" % stem)):
            p = _shot_at(pg, phrase_, nm)
            if p:
                shots.append(p)
    # A6 goes last: it presses a control and changes the page under us.
    results.append(a6_neighbour(case, pg, READY_MARKER,
                                shots if want_shots else None, stem))
    notes.append("walked in %.0fs" % (time.time() - t0))
    return results, [s for s in shots if s], notes, txt_path


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                         # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(
        description="Walk the four IPKG case pages and assert that the "
                    "2026-09-29 provenance audit reached the screen.")
    ap.add_argument("--url", default=os.environ.get("IPKG_UI_URL", URL))
    ap.add_argument("--case", action="append", default=None,
                    help="one display id; repeatable (default: all four)")
    ap.add_argument("--drawers", choices=("all", "core", "none"),
                    default="all")
    ap.add_argument("--no-shots", action="store_true")
    ap.add_argument("--headed", action="store_true",
                    help="watch it work, for when an assertion is in doubt")
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:                                    # noqa: BLE001
        print("Playwright is not importable in this interpreter (%s). It is "
              "installed in <app-workspace>\\.venv." % e)
        return 3

    os.makedirs(OUT_DIR, exist_ok=True)
    drawers = {"all": DRAWERS_ALL, "core": DRAWERS_CORE, "none": ()}[
        args.drawers]
    cases = [(c, l) for c, l in CASES
             if args.case is None or c in args.case]
    if not cases:
        print("No such case. The four are: %s"
              % ", ".join(c for c, _ in CASES))
        return 3
    phrase = _passphrase()

    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say("IPKG UI WALK  %s  %s" % (time.strftime("%Y-%m-%d %H:%M:%S"),
                                  args.url))
    say("drawers: %s" % (", ".join(drawers) if drawers else "none"))
    say("output:  %s" % OUT_DIR)
    say()

    every = []
    with sync_playwright() as p:
        b = p.chromium.launch(headless=not args.headed)
        pg = b.new_page(viewport={"width": 1600, "height": 1000})
        base_url = args.url.rstrip("/")
        for case, label in cases:
            say("=" * 78)
            say("%s   (%s)" % (case, label))
            say("=" * 78)
            try:
                res, shots, notes, txt = walk_case(
                    pg, base_url, case, label, phrase, drawers,
                    want_shots=not args.no_shots)
            except Exception as e:                            # noqa: BLE001
                res = [Result(case, "A0 the case page opened", "BLOCKED", [],
                              "the walk raised %s: %s" % (type(e).__name__, e))]
                shots, notes, txt = [], [], ""
            for n in notes:
                say("  . %s" % n)
            if txt:
                say("  . page text -> %s (%d chars)"
                    % (os.path.relpath(txt, BASE),
                       os.path.getsize(txt)))
            if shots:
                say("  . %d screenshots -> %s"
                    % (len(shots),
                       os.path.relpath(os.path.dirname(shots[0]), BASE)))
            say()
            for r in res:
                say("  %-7s %s" % (r.state, r.name))
                if r.why:
                    say("          why: %s" % r.why)
                for ev in r.evidence:
                    say("          %s" % ev)
                say()
            every.extend(res)

        b.close()

    n_pass = sum(1 for r in every if r.state == "PASS")
    n_fail = sum(1 for r in every if r.state == "FAIL")
    n_block = sum(1 for r in every if r.state == "BLOCKED")
    n_na = sum(1 for r in every if r.state == "N/A")
    say("=" * 78)
    say("SUMMARY  %d PASS  %d FAIL  %d BLOCKED  %d N/A"
        % (n_pass, n_fail, n_block, n_na))
    # THE N/A COUNT IS PART OF THE HEADLINE, not a footnote. "28 PASS, 0
    # FAIL" read as a tested build; the same walk is "23 tested and held,
    # 5 nothing to test", and a reader is entitled to the second sentence.
    say("         %d of %d assertions were actually exercised"
        % (n_pass + n_fail, len(every)))
    for r in every:
        if r.state != "PASS":
            say("  %-7s %s :: %s" % (r.state, r.case, r.name))
            if r.why:
                say("          %s" % r.why)
    say("=" * 78)

    with open(os.path.join(OUT_DIR, "ui_walk.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT_DIR, "ui_walk.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"url": args.url, "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "drawers": list(drawers),
                   "counts": {"pass": n_pass, "fail": n_fail,
                              "blocked": n_block, "not_applicable": n_na,
                              "exercised": n_pass + n_fail,
                              "total": len(every)},
                   "results": [r.as_dict() for r in every]},
                  fh, indent=1, ensure_ascii=False)
    return 1 if n_fail else (2 if n_block else 0)


if __name__ == "__main__":
    raise SystemExit(main())
