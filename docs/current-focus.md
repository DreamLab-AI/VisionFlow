---
title: Current focus
status: live
date: 2026-11-16
type: reference
---

# Current focus

The one page to start from when the question is "what is open, and where is it written down?".

## What is open

**One place: the [open-work register](open-work/README.md).** It holds every open, parked and recently closed item in the estate, each with one id. Its front page lists what is waiting on the owner; its [by-seam view](open-work/by-seam.md) files every item under the contract between components that would have to hold for it to close.

The register replaced five separate lists on 16 November 2026. They are frozen and listed under history below. If a list of open work exists anywhere else, that is a defect: add its items to the register and freeze it.

| View | What it is |
|---|---|
| [Front page](open-work/README.md) | Counts, the owner's decision queue, totals by seam, by component and by blocker |
| [By seam](open-work/by-seam.md) | Every item under its seam, live items first, then items recorded as open that nobody has re-checked |
| [Rebuild notes, 16 November](open-work/rebuild-notes-2026-11-16.md) | The review branches, now merged: what each needs rebuilt, set or run to take effect |
| [Branch and worktree ledger, 17 November](open-work/branches-2026-11-17.md) | What is merged, the two held commits, older unmerged branches, stashes and worktrees |
| [VisionClaw and agentbox board](https://github.com/DreamLab-AI/VisionClaw/blob/main/docs/TODO-unified.md) | The register's live items for those two components, generated into the VisionClaw repository at the path people already use |
| [Diagram index](diagrams/README.md) and [VisionClaw diagram register](diagrams/REGISTER-visionclaw.md) | Tensions, debt and drift recorded while cataloguing the code. These describe the code as read; an item becomes work only when it is added to the open-work register |

Two words the register uses carefully. **Live** means a check on the date shown found the item open, or fixed on a branch that is not merged. **Unverified** means an older list recorded it as open and nobody has looked since; it is not known to be open.

## How shared knowledge changes

The intended rule: a shared fact changes only when an applied change cites a human-signed decision bound to an agent-signed proposal, and the forum's governance contract is the one definition of those events. The [README](../README.md) states it. It is a design intent, not yet an enforced invariant: some paths record a decision without a human signature. The items that close that gap are under [Who decided](open-work/by-seam.md#decision-attribution) and [Governance events](open-work/by-seam.md#governance-events).

## What is history

Every record here is finished. None is updated.

| Where | What it is |
|---|---|
| [Archived board](https://github.com/DreamLab-AI/VisionClaw/blob/main/docs/archive/TODO-unified-2026-09-07-to-2026-11-15.md) | Every dated reconciliation of the to-do board from 7 September to 15 November 2026, with the definition of each row id |
| [Audit findings index](estate-review/archive/2026-11/2026-11-02-findings-index.md) | The 1 to 3 November audit programme's 173 findings as first recorded |
| [Duplicate implementations](estate-review/archive/2026-11/2026-11-01-parallel-implementations.md) | The 1 November report of 134 places the estate implements one thing more than once |
| [Park register](registers/park-2026-10-30.md) | 174 decision records parked on 30 October, each with a re-open trigger |
| [Gap register](registers/gap-register-v1.3.md) | Estate gaps as of July 2026 |
| [Sprint register, 31 October](registers/sprint-2026-10-31.md) | The closed sprint |
| [Estate review archive](estate-review/archive/README.md) | Dated audit records with no remaining reference |
| [Estate review](estate-review/README.md) | Dated audit records that stay at their paths only because something still cites them there |

## Documents that are no longer plans

The [18-month business roadmap](business/roadmap-2026-11-to-2028-04.md) and its decision record, ADR-2021, were rejected by the owner on 16 November 2026 and are kept as history. The [roadmap](roadmap.md) predates that ruling; read it as a proposal.

## Keeping this page true

- One open item has one id, and the register owns its status. Other documents may link to an item; they do not restate its status.
- A change that fixes an item updates its line in the register in the same pull request.
- A dated record is finished when it is written, wherever it sits. Whether a record is history is a property of the record, not of its folder.
