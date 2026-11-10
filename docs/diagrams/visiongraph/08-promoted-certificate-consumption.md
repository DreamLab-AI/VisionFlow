---
id: VG-08
title: Publisher consumes the promoted certificate generation
area: visiongraph
governing: []
adrs: [ADR-VG-002]
sources:
  - ../visionGraph/.github/workflows/publish.yml
verified_commit: d8277151a1404b456b5c7206187f7e4b50f25be1
---

## For developers

This scoped successor covers the certificate destination in VG-03.5. The workflow pins Ontomark 0.7.3 with its released binary checksum and uses an output-child certificate path compatible with the transactional compiler. It also consumes the official VOWL 0.2.3 GitHub bundle. No workflow deployment is inferred here.

## For the business

The verifier, review artefact and published site all take the certificate from the ontology generation's own output directory. Publication includes the full certificate, subject to the existing site size gates.

## VG-08.1 Verify and publish the certificate from the promoted bundle

```mermaid
flowchart TB
    BUILD["certificate below site-data output<br/>publish.yml:179"] --> VERIFY["Lean checker reads promoted files<br/>publish.yml:229"]
    VERIFY --> REPORT["checker report beside certificate<br/>publish.yml:230"]
    REPORT --> REVIEW["retain complete review artefact<br/>publish.yml:237"]
    REPORT --> SITE["copy whole generation to site<br/>publish.yml:344"]
```

**What it shows:** the checker and downstream copies use one certificate location within the compiler output. **Why it is this way:** the new compiler refuses external certificate directories so its generation and warrant share one promotion point. Formal checking establishes the declared proof fragment, not the factual truth of every authored statement. **Invariant:** verification reads the certificate under the same `site-data` tree that is published (`publish.yml:229`).
