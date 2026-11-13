# Runbook

What to do when something looks wrong. Start from the load-health line under the decision strip: it lists every section that failed or is behind, with the reason.

| Symptom | Likely cause | Action |
|---|---|---|
| Health line: **Scanner (prices through <date>, behind the MRS)** | The overnight scan ran before Yahoo published the session's bars. | Wait for the 09:30 Berlin morning run, or GitHub → Actions → CAN SLIM Scanner → Run workflow (default universe). About 20–40 minutes. |
| Health line: **Scanner (last run had low coverage, the previous full scan is shown)** | Yahoo throttled downloads; the coverage guard kept the previous scan. | Nothing urgent: the next scheduled run publishes and clears the flag. To force it: run the scanner workflow manually. If it repeats for 2+ runs, check the run log for download errors. |
| Health line: **Scanner (scan_results.csv missing / empty)** or **covered only N stocks** | Universe download failed or the job failed. | Actions → CAN SLIM Scanner → open the last run → read the error; rerun. |
| Health line: **Sector table / Allocation card / Risk today panel (… not available (Yahoo))** | A live Yahoo fetch failed in the app (cached 1–3 h). | Reload in a few minutes. If it persists: Manage app → Reboot app on share.streamlit.io (clears the cache). |
| Health line: **MRS history (GitHub fetch failed …)** | GitHub raw URL not reachable; the app shows the copy bundled at the last deploy. | Reload later; the bundled copy can be a few sessions old. |
| MRS row missing for a session | Daily update failed, or (before 13 Nov 2026) Columbus / Veterans Day skipped. | Actions → MRS Daily Update → Run workflow. On a weekend it fills the last Friday. Manual inputs: dashboard sidebar form (Backfill workflow). |
| Daily update failed at "git push" | Another commit landed meanwhile (fixed 13 Nov 2026 with pull --rebase). | Rerun the workflow. |
| Industries: "behind the MRS" | The industry job runs after the scan; same Yahoo timing. | As for the scanner. |
| A card shows old wording or numbers after a push | Streamlit has not redeployed yet, or the cache is warm. | Wait 1–3 minutes; check the footer version; then Reboot app. |
| Watchlist changes not saved | GITHUB_TOKEN missing in the Streamlit secrets, or a SHA conflict (saved from two tabs). | Caption under the watchlist says which; reload and save again. |
| Gamma shadow value missing | The trading PC did not push after the close (Gamma Live off). | Shadow only; the score uses the zero gamma input. Push from the PC the next day if needed. |

**Scheduled-task reminders:**
- End of March (summer time): change the scanner morning cron from `30 8` to `30 7` (UTC) in `scanner.yml`.
- End of October: change it back.
