# MRS Dashboard: documentation

This folder is the reference for how the MRS Dashboard is built, run, changed and audited. The code lives one folder up; every change to it is recorded in Git (GitHub repository `MRS_Dashboard`) and in the files below.

| Document | What it answers |
|---|---|
| [PROCESS.md](PROCESS.md) | How a change is made, tested, released and recorded. The checklist for every release. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | What the parts are, how data flows through them, what runs when. |
| [DATA.md](DATA.md) | Every data source and file: where it comes from, who writes it, when, and how it fails. |
| [EVIDENCE.md](EVIDENCE.md) | Every claim and number on screen, the study behind it and its status (Tested / Provisional / Descriptive). |
| [RUNBOOK.md](RUNBOOK.md) | What to do when something looks wrong: symptom → cause → action. |
| [DECISIONS.md](DECISIONS.md) | Why things are the way they are: dated decisions with their reason and evidence. |
| [../CHANGELOG.md](../CHANGELOG.md) | What changed in each version. |
| [audits/](audits/) | Audit reports (code, data, evidence). |

Other files in the repository: `README.md` (scoring overview), `BACKTEST_METHODOLOGY.md` (the MRS backtest method), `VERSION` (the current app version, shown in the page footer).

Research behind the dashboard lives outside this repository: the Epistruct project notes (`claude/*.md`) and the study folders in `TRADING DESK/Studies/`. EVIDENCE.md links each claim to them.
