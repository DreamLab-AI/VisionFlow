---
title: Current focus
status: live
date: 2026-11-16
type: reference
---

# Current focus

The one page to start from when the question is "what is open, and where is it written down?". Everything else that tracks work is linked from here. If a list of open work is not reachable from this page, that is a defect in this page.

## What is open

Cross-repository links on this page are absolute, so they work on GitHub as well as in a checkout.

| Where | What it holds | State |
|---|---|---|
| [Live board](https://github.com/DreamLab-AI/VisionClaw/blob/main/docs/TODO-unified.md) (VisionClaw repository) | The open rows for VisionClaw and agentbox, grouped by what blocks it. It is the only list here whose status is kept current | Rebuilt 16 November 2026 from a row-by-row check made on 15 November |
| [Audit findings index](estate-review/2026-11-02-findings-index.md) | 173 findings from the 1 to 3 November audit programme | Frozen at 3 November: the status column here is historical. The high findings still open on 15 November have rows on the live board, and the board is where their status is kept |
| [Park register](registers/park-2026-10-30.md) | 174 rows parked on 30 October, each with a re-open trigger | Checked 15 November. Every row has a re-open trigger; two triggers were the dates 3 to 5 November, now past, and those rows need a new trigger or a decision |
| [Gap register](registers/gap-register-v1.3.md) | Estate gaps; three open | Checked 15 November |
| [Duplicate implementations](estate-review/2026-11-01-parallel-implementations.md) | 134 places the estate implements one thing more than once | Frozen at 1 November: spot checks on 15 November found some entries already fixed in code |
| [Diagram index](diagrams/README.md) and [VisionClaw register](diagrams/REGISTER-visionclaw.md) | Tensions, debt and drift found while cataloguing the code. Only the VisionClaw area has a collected register; for other areas the markers are in the topic files | Generated from the diagram corpus |

## How shared knowledge changes

The intended rule: a shared fact changes only when an applied change cites a human-signed decision bound to an agent-signed proposal, and the forum's governance contract is the one definition of those events. The [README](../README.md) states it. It is a design intent, not yet an enforced invariant: some paths record a decision without a human signature. Closing that gap is the main engineering thread.

## What is history

| Where | What it is |
|---|---|
| [Archived board](https://github.com/DreamLab-AI/VisionClaw/blob/main/docs/archive/TODO-unified-2026-09-07-to-2026-11-15.md) | Every dated reconciliation of the board from 7 September to 15 November 2026 |
| [Estate review archive](estate-review/archive/README.md) | Dated audit records that nothing live cites any longer |
| [Estate review](estate-review/README.md) | Dated audit records still cited by diagrams, decision records or other repositories |
| [Sprint register, 31 October](registers/sprint-2026-10-31.md) | The closed sprint |

## Documents that describe plans, not present state

[Roadmap](roadmap.md) and the [18-month business roadmap](business/roadmap-2026-11-to-2028-04.md) with its decision record (ADR-2021, proposed) were written before the reference deployment they assume was deferred. Read them as proposals.

## Keeping this page true

- One open item has one id, and one list owns its status. Other lists may link to it; they do not restate its status.
- A dated record is finished when it is written, wherever it sits. Whether a record is history is a property of the record, not of its folder: some historical records stay in `estate-review/` only because a diagram citation is stamped to that path, and they move when that citation is next re-verified.
- A list that is no longer maintained says so here, with the date.
