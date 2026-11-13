# Change and release process

Goal: every change to the dashboard can be traced (what, why, evidence), tested, reversed, and reproduced by someone who was not there.

## Roles

| Role | Who | Does |
|---|---|---|
| Owner | Fabio | Decides what changes; approves; pushes from GitHub Desktop; checks the live app after the push. |
| Developer | Claude (or any contributor) | Edits the local clone, runs the tests, writes the change note, the changelog entry and, where needed, the decision and evidence entries. Never pushes. |
| Automation | GitHub Actions (MRS Bot) | Daily MRS update, nightly and morning scans, gamma shadow, manual backfill. Commits data files only. |

## Version numbers

`MAJOR.MINOR.PATCH` in the `VERSION` file, shown in the page footer.

- **MAJOR**: the scoring model or the meaning of a headline number changes (e.g. MRS v3 → v4 scoring).
- **MINOR**: a new card, scan or rule; a change in what a card says (e.g. 6.2 allocation card, 6.3 market context card).
- **PATCH**: fixes, wording, reliability, no change in meaning (e.g. 6.3.1 messaging fixes, 6.3.2 load-health line).

The scoring model has its own version (`pipeline.SCORING_VERSION`, e.g. "MRS v4.0"); it changes only with a new calibration report.

## Every change: the checklist

1. **Scope.** One sentence: what changes and why. If it changes what the user is told (a number, a rule, a claim), name the evidence (study, date, status).
2. **Backup.** Copy each file about to change to `Claude outputs/<change>_backup_<date>/` (Git also keeps history; the backup covers unpushed local edits).
3. **Edit** the local clone (`MRS_Dashboard/app`). Rules:
   - no secrets in code (`.streamlit/secrets.toml` and token files are never read or committed);
   - workflow files (`.github/workflows/`) only with the owner's explicit OK;
   - no third-party method names on screen; wording impersonal and factual; every on-screen figure carries its status tag.
4. **Test.**
   - `python -m py_compile` on every changed file;
   - the test scripts (`test_allocation.py`, `test_scoring_v4.py`; target: a `tests/` folder run by GitHub on every push);
   - an AppTest run of `app.py` (no exceptions; changed cards render);
   - for logic changes, a test on real or synthetic data, written down in the change note.
5. **Record.**
   - `CHANGELOG.md`: the version, date and what changed (user-visible first).
   - `docs/DECISIONS.md`: when a rule, threshold, model or message changes, add the decision, the reason and the evidence.
   - `docs/EVIDENCE.md`: when a claim on screen is added, changed or retired.
   - `VERSION`: bump.
   - Project note `claude/mrs_v<version>_<topic>_<date>.md`: the detailed change note (files, md5s, tests).
6. **Push** (owner, GitHub Desktop). Commit message: `v<version>: <one line>` (e.g. `v6.3.2: load-health line, scanner coverage guard`).
7. **Verify live** after Streamlit Cloud redeploys (1–3 minutes): the footer shows the new version; the changed card looks right; the load-health line is green.
8. **Tag** the release on GitHub (Releases → Draft a new release → tag `v6.3.2`), pasting the changelog entry. This marks an exact restorable state.

## Rolling back

- A bad release: GitHub Desktop → History → right-click the release commit → **Revert changes in commit** → push. The revert is itself recorded.
- A single file: restore it from the backup folder or from an earlier commit, then follow the checklist as a PATCH.

## Data commits

The bots commit only data files (`mrs_history.csv`, `StockScanner/output/*`, `gamma_shadow.csv`). Code and data changes are never mixed in one commit. Before a manual push, pull first (GitHub Desktop: Fetch origin → Pull), so a bot commit and a manual change do not collide.

## Audits

At least once a quarter, and after any large change:

- static analysis (ruff);
- all tests;
- AppTest;
- a review of the production paths (daily update, scanner, workflows);
- an evidence check: every claim on screen still matches EVIDENCE.md and its study.

The report goes to `docs/audits/AUDIT_<date>.md`. Findings that need a decision go to the owner as yes / no questions.
