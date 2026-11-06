---
id: VC-57
title: Control centre information architecture, scope badges, dock and right-hand drawer
area: visionclaw
verified_commit: {visionclaw: 809a859aae9e3d96d4d6b99690378018e48e3ad8}
governing:
  - ../project/docs/BASELINE-architecture.md
  - ../project/docs/XR-client.md
adrs: [ADR-2189, ADR-2188, ADR-2183, ADR-2185]
sources:
  - ../project/docs/adr/ADR-2189-desktop-and-xr-menus-share-one-seven-section-information-architecture.md
  - ../project/docs/BASELINE-architecture.md
  - ../project/docs/XR-client.md
  - ../project/docs/explanation/control-center.md
  - ../project/client/src/features/control-center/registry/placement.ts
  - ../project/client/src/features/control-center/registry/manifest.ts
  - ../project/client/src/features/control-center/registry/settingsRegistry.ts
  - ../project/client/src/features/control-center/registry/macros.ts
  - ../project/client/src/features/control-center/registry/groups/look.ts
  - ../project/client/src/features/control-center/registry/ia.json
  - ../project/client/src/features/control-center/registry/settings-manifest.json
  - ../project/client/src/features/control-center/registry/__tests__/registry.test.ts
  - ../project/client/scripts/emit-settings-manifest.ts
  - ../project/xr-client/data/ia.json
  - ../project/xr-client/scripts/hud_ia.gd
  - ../project/xr-client/scripts/hud.gd
  - ../project/xr-client/tests/unit/test_hud_ia.gd
  - ../project/client/src/features/control-center/primitives/SettingRow.tsx
  - ../project/client/src/features/control-center/panels/SettingsPanel.tsx
  - ../project/client/src/features/control-center/hooks/useControlCenterHotkeys.ts
  - ../project/client/src/features/control-center/state/useEnsureGroupLoaded.ts
  - ../project/client/src/features/control-center/state/settingWrites.ts
  - ../project/client/src/features/control-center/hooks/useSettingField.ts
  - ../project/client/src/features/control-center/macros/useMacro.ts
  - ../project/client/src/features/visualisation/components/CommandInput.tsx
  - ../project/client/src/api/settings/endpoints.ts
  - ../project/client/src/api/settings/schemaMappings.ts
  - ../project/client/src/features/control-center/ControlCenter.tsx
  - ../project/client/src/features/control-center/macros/MacroBar.tsx
  - ../project/client/src/features/control-center/macros/useDockCompact.ts
  - ../project/client/src/features/control-center/drawer/RightDrawer.tsx
  - ../project/client/src/features/control-center/state/useControlCenterUI.ts
  - ../project/client/src/features/control-center/hooks/useExclusiveSidePanels.ts
  - ../project/client/src/features/control-center/integrationSlots.ts
  - ../project/client/src/features/graph/explore/index.ts
  - ../project/client/src/features/graph/hooks/useGraphFiltering.ts
  - ../project/client/src/features/graph/components/GlassEdges.tsx
  - ../project/client/src/store/settings/retiredPaths.ts
  - ../project/client/src/store/settingsStore.ts
  - ../project/src/settings/models.rs
  - ../project/src/config/path_access.rs
  - ../project/crates/visionclaw-domain/src/config/visualisation.rs
  - ../project/client/src/api/settings/__tests__/settingsRoundTrip.test.ts
  - ../project/client/src/api/settings/defaults.ts
  - ../project/client/src/features/control-center/registry/groups/system.ts
  - ../project/client/src/features/control-center/registry/groups/graphView.ts
  - ../project/client/src/features/control-center/registry/groups/atmosphere.ts
  - ../project/client/src/features/control-center/primitives/SettingAction.tsx
  - ../project/client/src/features/control-center/primitives/RendererInfo.tsx
  - ../project/client/src/features/settings/config/settings.ts
  - ../project/client/src/rendering/rendererFactory.ts
  - ../project/src/settings/api/settings_routes.rs
  - ../project/src/adapters/sqlite_settings_repository.rs
  - ../project/docs/adr/ADR-2187-a-cosmos-gl-2d-overview-is-built-as-a-time-boxed-reference-spike.md
---

## For developers

ADR-2189 regroups every registry field into seven sections that both clients share. Group files still declare fields; `placement.ts` decides where each is shown; `manifest.ts` derives the sections, a per-field storage scope from the server bucket a path routes to, the test manifest and the `ia.json` map the headset reads. This topic covers that spine, the dock (dials, fold, buttons), the one right-hand drawer, the rule that keeps the settings panel and the drawer from overlapping, the lane seam in `integrationSlots.ts`, and how removed paths are stripped from old data. VC-34 keeps the settings round trip; VC-31 the render tree that mounts the drawer; VC-56 the explore panel inside it; VC-36 the headset HUD.

## For the business

Every setting now sits in one of seven named places, the same on the desktop and in the headset, and each one says whether changing it affects only this browser, every viewer, or the shared layout everyone sees. On a narrow screen the dock folds its five dials into one menu instead of pushing controls off the edge, and the explore, memory and node panels share one drawer so none hides another. Settings that were removed are cleaned out of old saved data on load.

## VC-57.1 From group declaration to the seven sections and their consumers
```mermaid
flowchart TB
    subgraph decl["Declared (group files)"]
        GD["GROUP_DATA, ten source groups<br/>manifest.ts:27"]
        LOOK["look group, e.g. maxEdgesCeiling under Lighting and Rendering<br/>look.ts:23"]
    end
    subgraph place["Placed (placement.ts)"]
        SD["SECTION_DEFS, seven sections, hotkeys 1 to 7<br/>placement.ts:28"]
        PF["placeField(sourceGroup, field)<br/>placement.ts:210"]
    end
    subgraph built["Built (manifest.ts)"]
        BS["buildSections, sort by subsection then declaration<br/>manifest.ts:75-98"]
        SECD["SECTION_DATA<br/>manifest.ts:108"]
        REG["REGISTRY, icons attached<br/>settingsRegistry.ts:30"]
    end
    GD --> BS
    LOOK --> GD
    SD --> BS
    PF -->|"section and subsection per field (placement.ts:210)"| BS
    BS --> SECD --> REG
    REG --> RAIL["SettingsPanel rail<br/>SettingsPanel.tsx:190"]
    REG --> DOCK["dock section buttons<br/>MacroBar.tsx:322"]
    REG --> HK["digit hotkeys from registry order<br/>useControlCenterHotkeys.ts:96-99"]
    SECD --> BM["buildManifest, test ids and buckets<br/>manifest.ts:156"]
    SECD --> BI["buildIa, section to subsection to fields<br/>manifest.ts:238"]
    BI --> IA2["two committed ia.json copies (VC-57.3)<br/>emit-settings-manifest.ts:19-22"]
    REG --> LOAD["first open hydrates the section loadPaths<br/>useEnsureGroupLoaded.ts:24-29"]
```

**What it shows.** The registry pipeline after ADR-2189: declaration, placement, section build, then the four consumers (rail, dock, hotkeys, emitters). **Why it is this way.** ADR-2189 §1-§2 keep group files as the place a field is declared and make placement data, so the seven-section map can be emitted for the headset without React or icons.

**Drift (control-center.md vs code):** the explanation page still describes nine groups on hotkeys 1-9 (`control-center.md:100`, `:233`); the code has seven sections (`placement.ts:28`) and generates hotkeys from them (`useControlCenterHotkeys.ts:96`).

## VC-57.2 placeField: where one field is shown
```mermaid
flowchart TB
    IN["field plus its source group<br/>placement.ts:210"] --> P1{"path in PATH_PLACEMENT?<br/>placement.ts:211"}
    P1 -->|yes| OV["per-path override, e.g. maxEdgesCeiling to System Performance<br/>placement.ts:177, placement.ts:181"]
    P1 -->|no| P2{"motion group, subgroup starts Semantic?<br/>placement.ts:213"}
    P2 -->|yes| SEM["Layout, Semantic forces then the subgroup tail<br/>placement.ts:214"]
    P2 -->|no| P3["SUBGROUP_PLACEMENT by sourceGroup and subgroup<br/>placement.ts:112, placement.ts:216"]
    P3 -->|"no entry (placement.ts:217)"| THROW["throw, nothing drops out of the menus<br/>placement.ts:218"]
    OV --> CHK{"subsection declared on the section?<br/>placement.ts:221"}
    SEM --> CHK
    P3 --> CHK
    CHK -->|no| THROW2["throw<br/>placement.ts:222"]
    CHK -->|yes| OUT["section and subsection<br/>placement.ts:224"]
    T["INVARIANT: test b2 enumerates every field through it<br/>registry.test.ts:259"] -.-> THROW
```

**What it shows.** The three-step resolution for one field and the two throws that make an unplaced field a test failure. **Why it is this way.** ADR-2189 §7 maps old subgroups wholesale but moves a few mixed-concern paths individually; throwing keeps the "every control in exactly one place" rule mechanical.

**Debt (loose end):** `maxEdgesCeiling` ("Edge memory budget") is still declared in the look group under "Lighting & Rendering" with a stale "Phase 6" description (`look.ts:23`), though it caps edge capacity (`GlassEdges.tsx:244`); only the per-path override shows it under System › Performance (`placement.ts:181`), as ADR-2189 §7 asks (`ADR-2189-desktop-and-xr-menus-share-one-seven-section-information-architecture.md:218`).

## VC-57.3 The emitted documents and what pins each copy
```mermaid
flowchart TB
    EM["npm run gen:manifest<br/>emit-settings-manifest.ts:14"] --> MAN["settings-manifest.json, browser coverage phase<br/>emit-settings-manifest.ts:16, emit-settings-manifest.ts:25"]
    EM --> IAW["the same buildIa text written twice<br/>emit-settings-manifest.ts:26-27"]
    IAW --> C1["registry/ia.json, desktop copy<br/>registry/ia.json:1844"]
    IAW --> C2["data/ia.json, headset copy<br/>data/ia.json:1844"]
    B5["test b5, both copies deep-equal a fresh buildIa<br/>registry.test.ts:329-337"] -.->|pins| C1
    B5 -.->|pins| C2
    TE["test e, counts and per-section field counts only<br/>registry.test.ts:525, registry.test.ts:530-533"] -.->|"OPEN: labels, paths and test ids are not compared (registry.test.ts:530)"| MAN
    C2 --> LD["HudIa.load_doc, falls back to a built-in seven<br/>hud_ia.gd:88"]
    LD --> HUD["HUD tab bar built from it<br/>hud.gd:506, hud.gd:510"]
    FB["FALLBACK_SECTIONS and FALLBACK_SCOPES, hand-kept<br/>hud_ia.gd:26, hud_ia.gd:35"] -.-> LD
    GUT["GUT test, fallback equals the committed file<br/>test_hud_ia.gd:38, test_hud_ia.gd:47"] -.->|pins| FB
    XO["XR_ONLY exempts names the desktop lacks<br/>hud_ia.gd:75"] -.-> GUT2["GUT test, every XR subsection is the desktop's or XR-only<br/>test_hud_ia.gd:68-71"]
```

**What it shows.** One emitter, three committed outputs, and which test pins which copy. **Why it is this way.** ADR-2189 §2 follows the `semantic_fields.json` precedent: one generated document, two committed copies, a test that regenerates and compares, so desktop and headset cannot drift silently.

**Debt (parallel):** the IA exists as two committed copies, `registry/ia.json` and `xr-client/data/ia.json`, written by one emitter (`emit-settings-manifest.ts:19-22`); they are generated, never hand-edited.

**Invariant:** both copies deep-equal a fresh `buildIa()` or the Vitest suite fails (`registry.test.ts:329-337`).

**Debt (parallel):** the headset keeps a hand copy of the seven sections and the scope texts (`hud_ia.gd:26`, `:35`) beside `SCOPE_TEXT` (`manifest.ts:223`); only the GUT test ties them (`test_hud_ia.gd:38-47`).

**Tension (parallel):** the XR "Semantic forces" subsection passes the parity test only by being listed as XR-only (`hud_ia.gd:79`), though the desktop has eight "Semantic forces › …" subsections (`placement.ts:62-68`).

**Tension (ADR-2189 vs XR-client.md):** ADR-2189 still says the XR HUD reading `ia.json` and its GUT parity test are not built (`ADR-2189-desktop-and-xr-menus-share-one-seven-section-information-architecture.md:23`, `:314`); `XR-client.md:1061` records the XR side as built, and `hud.gd:510` loads it.

## VC-57.4 Scope badges from the server bucket
```mermaid
flowchart TB
    PATH["field path"] --> SBF["serverBucketFor<br/>manifest.ts:123"]
    SBF --> B1["physics, graphs and .physics.<br/>manifest.ts:125"]
    SBF --> B2["rendering only for the ten server RenderingSettings keys,<br/>other rendering paths such as agentColors go to visual<br/>SERVER_RENDERING_KEYS manifest.ts:117, manifest.ts:132"]
    SBF --> B2D["null for the per-device budgets maxEdgesCeiling,<br/>softwareFallback, labelLayoutEvery<br/>manifest.ts:131"]
    SBF --> B2Q["qualityGates, nodeFilter, constraints<br/>manifest.ts:134-136"]
    SBF --> B3["visual, other visualisation paths<br/>manifest.ts:141"]
    SBF --> B4["null, browser only<br/>manifest.ts:124, manifest.ts:142"]
    SBF -.-> DEAD["DEBT: two branches can never fire after the dotted prefixes above<br/>manifest.ts:139-140"]
    B1 --> SF["scopeFor, physics to shared, other bucket to server, null to browser<br/>manifest.ts:55-59"]
    B2 --> SF
    B2D --> SF
    B2Q --> SF
    B3 --> SF
    B4 --> SF
    SF --> SFF["scopeForField, an action may override<br/>manifest.ts:62-63"]
    SFF --> BADGE["ScopeBadge text and icon<br/>SettingRow.tsx:50, SettingRow.tsx:110"]
    TXT["SCOPE_TEXT, every server bucket reaches every viewer<br/>manifest.ts:223-227"] --> BADGE
    SF --> MAC["dial scope is the strongest of the paths it writes<br/>macros.ts:112-114"]
    MAC --> DIALS["layout dials vs view dials<br/>MacroBar.tsx:77-78"]
    INV["INVARIANT: only physics paths are shared, dial scope equals strongest<br/>registry.test.ts:306, registry.test.ts:317"] -.-> SF
    R1["write routing, batched, dotted prefixes<br/>endpoints.ts:362-376"] -.->|"mirrored by hand (manifest.ts:111-114)"| SBF
    R2["write routing, single path, undotted prefixes<br/>endpoints.ts:303-313"] -.-> SBF
    R3["isVisualSettingsPath and isServerRenderingPath<br/>schemaMappings.ts:177-181, schemaMappings.ts:158-161"] -.-> R1
    R4["serverBucketForPath, the routing copy in schemaMappings<br/>schemaMappings.ts:165-174"] -.-> R3
    PIN["INVARIANT: serverBucketFor equals serverBucketForPath<br/>over every registry path, settingsRoundTrip.test.ts:117-119"] -.-> SBF
    PIN -.-> R4
```

**What it shows.** How a row's badge and a dial's scope derive from the bucket a path would be PUT to. **Why it is this way.** ADR-2189 §2 wants every row to say where its value is stored; the implementation note corrected the wording because every server bucket is a single global document (`ADR-2189-desktop-and-xr-menus-share-one-seven-section-information-architecture.md:305`).

**Tension (parallel):** the badge's `serverBucketFor` copies the batched router's dotted prefixes (`manifest.ts:125`, `endpoints.ts:362`), but the single-path writer tests `.physics` and bare `qualityGates`/`nodeFilter` prefixes (`endpoints.ts:303`, `:307`, `:309`), so the two writers and the badge can disagree on a path such as `qualityGatesX.*`; no registered path hits the gap today.

**Debt (parallel):** bucket routing is written four times: the badge's `serverBucketFor` (`manifest.ts:123`), `serverBucketForPath` (`schemaMappings.ts:165`) and the two writers (`endpoints.ts:303-313`, `:362-376`); the rendering key set is copied by hand (`manifest.ts:117` beside `schemaMappings.ts:139`) and the device-local list is an inline literal (`manifest.ts:131`) beside `DEVICE_LOCAL_RENDERING_KEYS` (`schemaMappings.ts:149`), pinned to each other only over registry paths (`settingsRoundTrip.test.ts:117-119`).

**Drift (comment vs code):** `manifest.ts:127-128` says `maxEdgesCeiling`, `softwareFallback` and `labelLayoutEvery` live in `/visual`, but the next branch returns `null` for them, so they stay in the browser (`manifest.ts:131`).

**Drift (comment vs code):** the `SCOPE_TEXT` header says rendering lives in the settings actor (`manifest.ts:220-221`); the rendering handler now also stores a `rendering` row (`settings_routes.rs:1004-1007`, `sqlite_settings_repository.rs:1057`).

## VC-57.5 One write path for rows, dials and the Ask box
```mermaid
flowchart LR
    ROW["registry row<br/>useSettingField.ts:47, useSettingField.ts:70"] --> ASW["applySettingWrites<br/>settingWrites.ts:42"]
    DIAL["dock dial<br/>useMacro.ts:36"] --> ASW
    ASK["Ask box keyword rules<br/>CommandInput.tsx:492, CommandInput.tsx:561"] --> ASW
    ASW --> BATCH["one immer batch on the settings store<br/>settingWrites.ts:46-58"]
    ASW --> Q["autoSaveManager.queueChange per path<br/>settingWrites.ts:63-64"]
    ASW --> LM["layoutApi.setMode for layout-mode paths<br/>settingWrites.ts:65-66"]
    BATCH -.-> DQ["DEBT: updateSettings already queues, so each path is queued twice by design<br/>settingWrites.ts:60-62"]
```

**What it shows.** The three writing surfaces converge on one function with one store batch, one autosave queue and one layout side effect. **Why it is this way.** The menus audit found up to four writers per setting and an Ask box that PUT physics directly, bypassing the store and the Sync switch (`settingWrites.ts:12-14`); ADR-2189 §6 makes one writer.

## VC-57.6 The dock: dials, buttons and the narrow-width fold
```mermaid
flowchart LR
    CC["ControlCenter mounts GlassDock and MacroBar<br/>ControlCenter.tsx:115-116"] --> MB["MacroBar<br/>MacroBar.tsx:289"]
    MB --> HYD["hydrate every macro path<br/>MacroBar.tsx:290, useEnsureGroupLoaded.ts:42"]
    MB --> FOLD{"compact?<br/>MacroBar.tsx:292, MacroBar.tsx:303"}
    FOLD -->|no| DIALS["Dials, layout group then view group<br/>MacroBar.tsx:80"]
    FOLD -->|yes| DM["one Dials menu button<br/>MacroBar.tsx:97, MacroBar.tsx:100"]
    DIALS --> LD["Spread and Motion, shared<br/>MacroBar.tsx:77, macros.ts:36"]
    DIALS --> VD["Glow, Labels, Atmosphere, server<br/>MacroBar.tsx:78, macros.ts:49"]
    MB --> PT["Physics toggle and Reset layout<br/>MacroBar.tsx:119, MacroBar.tsx:139"]
    MB --> FIT["Fit split button, stored fit scope<br/>MacroBar.tsx:197, MacroBar.tsx:207"]
    MB --> LEG["Legend toggle<br/>MacroBar.tsx:241"]
    MB --> EXP["Explore panel button, filter badge<br/>MacroBar.tsx:259, MacroBar.tsx:264"]
    MB --> SEC["seven section buttons<br/>MacroBar.tsx:322"]
    LD -.-> IDS["DEBT: ids keep the old names density, luminosity, focus; tests pin them<br/>macros.ts:36, registry.test.ts:325-326"]
    SEC -.-> ONE["OPEN: only the dials fold; the view and section buttons never do<br/>MacroBar.tsx:303"]
```

**Debt (loose end):** the Atmosphere dial still drives `sceneEffects.fogOpacity` (`macros.ts:101`) and the Fog opacity row still offers it (`atmosphere.ts:26`), but nothing in the client draws fog from it since the GLSL fog plane was deleted; the only other mentions are its default and type (`defaults.ts:140`, `settings.ts:507`).

**What it shows.** What the dock renders and the single point where it folds. **Why it is this way.** ADR-2189 §3 orders the dock as shared dials, view dials, physics and reset, then Fit, Legend, Explore and the sections; the audit found the old dock 74 px off-screen at 1024 px (`useDockCompact.ts:9-10`).

## VC-57.7 useDockCompact: overflow-triggered fold with hysteresis
```mermaid
stateDiagram-v2
    [*] --> Wide
    Wide --> Compact : dock body overflows its capped width
    Compact --> Wide : viewport wider than the overflow width plus 48 px
    Compact --> Compact : any other resize
    note right of Wide
        check reads scrollWidth vs clientWidth
        useDockCompact.ts:29
        records innerWidth at overflow, useDockCompact.ts:30-32
    end note
    note right of Compact
        HYSTERESIS_PX = 48, useDockCompact.ts:15
        release test, useDockCompact.ts:33-35
        re-checked on every resize of the box, useDockCompact.ts:40
    end note
```

**What it shows.** The fold is driven by measured overflow, not a breakpoint, and only releases once the viewport has grown 48 px past the width where it overflowed. **Why it is this way.** Measuring makes an off-screen dock impossible at any font or content width, and the hysteresis stops flicker at the boundary (`useDockCompact.ts:6-8`).

**Drift (ADR-2189 §3 vs code):** the ADR fixes the fold "below 1180 px" (`ADR-2189-desktop-and-xr-menus-share-one-seven-section-information-architecture.md:147`); the code folds on measured overflow with no breakpoint (`useDockCompact.ts:29-33`).

## VC-57.8 The right-hand drawer's tab state
```mermaid
stateDiagram-v2
    [*] --> Closed
    Closed --> Explore : dock Explore button or slash hotkey
    Closed --> Memory : embedding cloud switched on
    Closed --> Node : node selected
    Explore --> Node : node selected, previous tab kept
    Memory --> Node : node selected, previous tab kept
    Node --> Explore : close Node with explore previous, deselects
    Node --> Memory : close Node with memory previous, deselects
    Node --> Closed : deselect with nothing previous
    Explore --> Closed : close or Esc inside the drawer
    Memory --> Closed : cloud switched off
    note right of Closed
        drawerTab null, DrawerTab type useControlCenterUI.ts:64
        openDrawer keeps previous unless leaving node, useControlCenterUI.ts:136-141
        closeDrawer on node restores previous and fires deselect, useControlCenterUI.ts:142-149
        dismissDrawerTab, useControlCenterUI.ts:151-158
    end note
    note right of Node
        tabs offered, explore always, memory while cloud on, node while a label is held
        RightDrawer.tsx:173-178
        selection events RightDrawer.tsx:104-127, cloud RightDrawer.tsx:132-138
        each panel has its own error boundary, RightDrawer.tsx:286, RightDrawer.tsx:300, RightDrawer.tsx:316
    end note
```

**What it shows.** The three tabs, what opens and dismisses each, and how closing the Node tab returns to the tab shown before it. **Why it is this way.** ADR-2189 §4 puts explore, memory explorer and node detail in one drawer because the memory panel and node detail used to cover the explore drawer and each other (`ADR-2189-desktop-and-xr-menus-share-one-seven-section-information-architecture.md:332`).

**Drift (comment vs code):** `useExclusiveSidePanels.ts:11-13` says the dock's Explore button brings a hidden drawer tab back, but `openDrawer('explore')` overwrites the stashed `previousDrawerTab` (`useControlCenterUI.ts:140`), so a hidden Node tab returns only when its tab button is clicked (`RightDrawer.tsx:176`) or another node is selected.

## VC-57.9 useExclusiveSidePanels: settings panel and drawer never overlap
```mermaid
sequenceDiagram
    participant UI as useControlCenterUI<br/>useControlCenterUI.ts:114
    participant X as useExclusiveSidePanels<br/>useExclusiveSidePanels.ts:41
    participant C as sidePanelsCollide<br/>useExclusiveSidePanels.ts:35
    participant W as window resize
    UI->>X: subscribe(state, prev) (useExclusiveSidePanels.ts:45)
    X->>X: return unless panel open and drawer tab set (useExclusiveSidePanels.ts:46)
    X->>C: edge + stored panel width + edge + drawer + edge above viewport? (useExclusiveSidePanels.ts:47)
    Note over C: drawer is min of 380 and viewport minus edges, useExclusiveSidePanels.ts:36-38
    alt drawer tab changed (opened last)
        X->>UI: closePanel (useExclusiveSidePanels.ts:51-53)
    else panel opened, section changed or width dragged
        X->>UI: drawerTab null, previousDrawerTab kept (useExclusiveSidePanels.ts:54-57)
    end
    W->>X: resize while both open and colliding (useExclusiveSidePanels.ts:62-64)
    X->>UI: closePanel, the drawer usually holds what was just asked for (useExclusiveSidePanels.ts:64)
    Note over UI,X: mounted once by ControlCenter, ControlCenter.tsx:82
```

**What it shows.** The last-opened-wins rule and the resize rule that resolve a collision between the left settings panel and the right drawer. **Why it is this way.** ADR-2189 §3-§4 forbid overlapping fixed surfaces; the stored width is used because during a drag the DOM still reports the old width (`useExclusiveSidePanels.ts:25-29`).

## VC-57.10 integrationSlots: the one seam to the render and explore lanes
```mermaid
flowchart LR
    subgraph slots["integrationSlots.ts"]
        RF["requestFit, FitTarget<br/>integrationSlots.ts:23-24"]
        OEN["OverviewEdgeNotice<br/>integrationSlots.ts:25"]
        EXP["ExplorePanel, GraphLegend, clearAllFilters, useActiveFilterCount<br/>integrationSlots.ts:26"]
        HF["HIDE_FILTERED_PATH<br/>integrationSlots.ts:27"]
    end
    LANE_E["graph/explore barrel<br/>index.ts:8-11"] --> EXP
    LANE_F["hideFiltered setting path<br/>useGraphFiltering.ts:40"] --> HF
    RF --> MBR["MacroBar Fit<br/>MacroBar.tsx:31, MacroBar.tsx:207"]
    EXP --> MBE["MacroBar Explore badge<br/>MacroBar.tsx:260"]
    EXP --> DRW["RightDrawer explore tab<br/>RightDrawer.tsx:29, RightDrawer.tsx:287"]
    EXP --> CCL["ControlCenter always mounts GraphLegend, it installs the encoder<br/>ControlCenter.tsx:191, ControlCenter.tsx:212"]
    RULE["rule: clear filters with clearAllFilters, never highlightMask.clearAll<br/>integrationSlots.ts:20-21"] -.-> EXP
    OPN["OPEN: the one-file rule is a convention; no test enforces it<br/>integrationSlots.ts:3-5"] -.-> slots
```

**What it shows.** Every control-centre dependency on the render (ADR-2180/2181) and explore (ADR-2183-2185) lanes comes through one re-export file. **Why it is this way.** The three lanes were built in parallel; one seam means a lane API change touches one file (`integrationSlots.ts:2-5`).

## VC-57.11 Retired paths: one client list, three strip points, separate server lists
```mermaid
flowchart TB
    subgraph client["client: retiredPaths.ts"]
        DP["DEAD_PATHS, 41: 37 from ADR-2189, 4 from the settings audit<br/>retiredPaths.ts:20-21, retiredPaths.ts:25-28"]
        RP["REPLACED_PATHS, 9: 4 plus the five bloom paths glow replaces<br/>retiredPaths.ts:74, retiredPaths.ts:83-87"]
        MP["MERGED_PATHS, 3<br/>retiredPaths.ts:100-104"]
        RN["RENAMED_PATHS, edge colour moves to agent mode<br/>retiredPaths.ts:111-113"]
        ALL["ADR_2189_REMOVED_PATHS<br/>retiredPaths.ts:116"]
        STRIP["stripRemovedPaths, renames first then delete and prune<br/>retiredPaths.ts:127-160"]
    end
    DP --> ALL
    RP --> ALL
    MP --> ALL
    ALL --> STRIP
    RN --> STRIP
    STRIP --> S1["localStorage rehydrate merge<br/>settingsStore.ts:47, settingsStore.ts:59"]
    STRIP --> S2["server load boundary<br/>endpoints.ts:247"]
    ALL --> S3["test b3, none is still a control<br/>registry.test.ts:285"]
    STRIP -.-> DR["header misnames the server-load strip site, registered once in VC-30<br/>retiredPaths.ts:11-12"]
    subgraph server["server: separate lists"]
        PA["RETIRED_SETTING_KEYS, nodes.enableMetadataShape<br/>path_access.rs:306"]
        DV["node blob drops enableMetadataShape<br/>visualisation.rs:548"]
        QG["quality gates still carry showClusters and showCommunities<br/>models.rs:64, models.rs:70"]
    end
    QG -->|"served, then stripped (endpoints.ts:247)"| S2
```

**What it shows.** How the 53 removed paths leave old data on both load paths, and the separate server-side retirements. **Why it is this way.** ADR-2189 §6 deletes paths nothing reads but promises that blobs written by older clients still load, so stripping is tolerant and happens at every boundary (`retiredPaths.ts:122-126`).

**Tension (parallel):** the client retires `qualityGates.showClusters` and `showCommunities` (`retiredPaths.ts:102-103`), but the server still models them as live quality-gate fields with no reader (`models.rs:64`, `:70`), so every load serves them and the client strips them again (`endpoints.ts:247`); the same holds for `qualityGates.ruvectorEnabled`, retired by the settings audit (`retiredPaths.ts:26`) but still a server field (`models.rs:73`).

**Debt (parallel):** retirement is three lists with no shared source: the client's `ADR_2189_REMOVED_PATHS` (`retiredPaths.ts:116`), the server's `RETIRED_SETTING_KEYS` (`path_access.rs:306`) and the domain node-blob filter (`visualisation.rs:548`).

## VC-57.12 What the settings-integrity pass changed in the menus
```mermaid
flowchart TB
    subgraph edges["Look › Edges holds every edge appearance control"]
        LE["look Edges rows, thickness, Edge Opacity (all edges), three mode colours<br/>look.ts:14-19"]
        LP["look Edges and Graph-Type Visuals placed in Look › Edges<br/>placement.ts:124-126"]
        OE["Overview Edges rows, Long-Edge Fade Floor among them<br/>graphView.ts:38, graphView.ts:43, placed placement.ts:168"]
        ENC["Encode has no Edges subsection<br/>placement.ts:48"]
    end
    subgraph scope["Scope follows where the value is actually kept"]
        DL["per-device budgets return null, badge reads This browser only<br/>manifest.ts:131, manifest.ts:226"]
        AC["agentColors and other non-server rendering keys go to visual<br/>manifest.ts:132"]
    end
    subgraph gone["Retired and redirected"]
        RT["node opacity, ruvectorEnabled, auth.enabled, auth.required<br/>retiredPaths.ts:25-28"]
        BL["bloom.* replaced by glow.*<br/>retiredPaths.ts:83-87"]
        ASKN["Ask box answers that node opacity is not a setting<br/>CommandInput.tsx:306-312"]
        ASKG["Ask box glow or bloom writes glow.enabled only<br/>CommandInput.tsx:345, CommandInput.tsx:350"]
    end
    HF["Hide filtered nodes toggle in Explore › Filters<br/>graphView.ts:57, placement.ts:191"]
    LE --> LP --> OE
    LP --> ENC
    ENC --> DL --> AC
    AC --> RT --> BL --> ASKN --> ASKG
    ASKG --> HF
    WG["System › Performance WebGPU Renderer row says it switches WebGPU and WebGL<br/>system.ts:15, placement.ts:150"]
    HF --> WG
    WG --> ACT["toggle-webgpu now forces three's WebGL2 backend, then reloads<br/>SettingAction.tsx:79-82, rendererFactory.ts:59-62"]
    WG -.-> WD["DRIFT: the label and description predate the single WebGPURenderer<br/>system.ts:15"]
```

**What it shows.** The menu side of the 2026-11-06 settings audit: edge appearance gathered under Look › Edges, scopes that match where a value is kept, the retired paths and their Ask-box redirects, and the explore hide-filtered toggle. **Why it is this way.** The audit found edge opacity split from its fade floor and per-device budgets labelled as server-saved (`placement.ts:123-124`, `schemaMappings.ts:145-150`).

**Drift (registry vs renderer):** the WebGPU Renderer row still says it switches between WebGPU and WebGL renderers (`system.ts:15`); the desktop has one renderer, and the toggle forces its WebGL2 backend (`SettingAction.tsx:80-81`, `rendererFactory.ts:59-62`).

**Debt (loose end):** the renderer info row shows TSL Materials and Node Bloom badges (`RendererInfo.tsx:45-46`) from flags that are now always true (`rendererFactory.ts:196-197`, `:206-207`), and the type still contrasts node bloom with EffectComposer (`settings.ts:158`).

**Tension (ADR-2189 vs ADR-2187):** ADR-2189 places the ADR-2187 overview route flag under System › Developer (`ADR-2189-desktop-and-xr-menus-share-one-seven-section-information-architecture.md:104`, `:257`); ADR-2187 is rejected (`ADR-2187-a-cosmos-gl-2d-overview-is-built-as-a-time-boxed-reference-spike.md:5`) and only Debug Logging feeds Developer (`placement.ts:153`).
