---
id: PS-02
title: The website audit and its SSRF filter
area: prose-sanitiser
governing: [../prose-sanitiser/README-CLI.md, ../prose-sanitiser/docs/UPSTREAM-DELTA.md]
adrs: []
sources:
  - ../prose-sanitiser/crates/cli/src/bin/audit_website.rs
  - ../prose-sanitiser/crates/cli/src/audit/website/mod.rs
  - ../prose-sanitiser/crates/cli/src/audit/website/net.rs
  - ../prose-sanitiser/crates/cli/src/exit.rs
  - ../prose-sanitiser/crates/cli/src/audit/mod.rs
  - ../prose-sanitiser/crates/media/src/image/mod.rs
  - ../prose-sanitiser/crates/media/src/image/harness.rs
  - ../prose-sanitiser/crates/media/src/image/tools.rs
  - ../prose-sanitiser/README-CLI.md
  - ../prose-sanitiser/docs/UPSTREAM-DELTA.md
  - ../prose-sanitiser/docs/prose-sanitiser-external-review-2026-09-03.md
verified_commit: 266063408b57f52672b5b061491120373d72ffef
---
## For developers

`audit-website` crawls the URLs listed in a sitemap, downloads each one and runs
the same inspection the local directory audit uses
(`../prose-sanitiser/crates/cli/src/audit/website/mod.rs:1`-`6`). The input is a
URL that somebody else nominated, and so is every URL inside the sitemap it
points at. The module therefore validates every hop before a socket is opened:
only http and https, no credentials in the URL, the same origin as the start
(plus the ordinary http to https upgrade), and only globally routable addresses
(`../prose-sanitiser/crates/cli/src/audit/website/net.rs:1`-`7`).

The filter has two halves. `net.rs` decides whether a URL and the addresses its
host resolves to are acceptable. `mod.rs` then connects only to the addresses
that were validated, by handing the HTTP client a resolver that answers with
nothing else (`../prose-sanitiser/crates/cli/src/audit/website/mod.rs:54`-`68`).
That pinning is what stops a DNS answer that changes between the check and the
connect. The binary drives it: it finds or accepts a sitemap, collects the
same-site URLs, fetches each one under a byte cap and a timeout, and reports
(`../prose-sanitiser/crates/cli/src/bin/audit_website.rs:38`-`127`).

The defaults are a 4 MiB body cap, 15 second timeout, 200 pages and 5
redirects (`../prose-sanitiser/crates/cli/src/audit/website/mod.rs:21`-`33`).
A decompressed sitemap is capped at 64 MiB and that cap is not configurable
(`:26`-`30`).

## For the business

The tool downloads whatever a website's sitemap lists, so a hostile or mistaken
sitemap could try to make it fetch addresses inside the operator's own network,
such as an internal admin page or a cloud metadata service. The filter refuses
those. It will only talk to public internet addresses, only to the site it was
pointed at, and it never sends credentials embedded in a URL.

What an operator can rely on: a page that fails to download is listed in the
report, and the run carries on. What they cannot rely on today is the exit code
meaning "every page was checked", which the topic records below.

## PS-02.1 One audit-website run

```mermaid
sequenceDiagram
    autonumber
    participant BD as body<br/>audit_website.rs:38
    participant DS as discover_sitemap<br/>website/mod.rs:328
    participant CU as collect_urls<br/>website/mod.rs:371
    participant FE as fetch<br/>website/mod.rs:243
    participant IR as inspect_remote<br/>website/mod.rs:310
    BD->>BD: neither --sitemap nor --base is exit 2 (audit_website.rs:40-45)
    BD->>DS: --base only: look for a sitemap (audit_website.rs:51)
    DS-->>BD: a URL, or none which is exit 2 (audit_website.rs:53-58)
    BD->>CU: sitemap URL, timeout, max pages (audit_website.rs:69)
    CU-->>BD: same-site URLs, an error is exit 2 (audit_website.rs:69-74)
    BD->>BD: no URLs is exit 2 (audit_website.rs:75-77)
    loop each URL, at most max pages
        BD->>FE: URL, timeout, byte cap, no expected origin (audit_website.rs:82)
        FE-->>BD: bytes and Content-Type, or an error string
        BD->>IR: bytes, content type (audit_website.rs:84)
        Note over BD: a fetch error is kept in a failures list<br/>and the loop carries on (audit_website.rs:86)
    end
    BD->>BD: aggregate, then exit 1 only if a file is actionable (audit_website.rs:90-126)
```

**What it shows.** Every failure before the crawl starts is exit 2. Once the
loop starts, a download failure is recorded in `urls_failed` and does not reach
the exit code, which depends only on the actionable-file count.

**Why it is this way.** The summary block is shared with the directory audit, so
both binaries report and exit the same way on what they did inspect
(`../prose-sanitiser/crates/cli/src/audit/mod.rs:175`-`176`).

**Tension (exit contract vs crawl):** the contract defines exit 2 as unreadable
input (`../prose-sanitiser/crates/cli/src/exit.rs:11`, see PS-01), but a page
that fails to download is only appended to `failures`
(`../prose-sanitiser/crates/cli/src/bin/audit_website.rs:86`) and the exit is
`from_flag(actionable > 0)` (`:126`). A crawl in which every page was refused,
for example by the address filter, exits 0.

**Debt:** `audit_website.rs:71` passes the literal `2` where the file uses
`exit::ERROR` elsewhere (`../prose-sanitiser/crates/cli/src/bin/audit_website.rs:42`).

## PS-02.2 One fetch, hop by hop

```mermaid
flowchart TB
    START["fetch url, timeout, max bytes, allowed origin<br/>website/mod.rs:243"]
    VT["validated_target: URL policy, then public addresses<br/>website/mod.rs:253"]
    SET["expected origin is set once, from the first hop,<br/>when the caller gave none<br/>website/mod.rs:254"]
    PIN["new agent per hop: redirects(0), timeouts,<br/>PinnedResolver answering only the validated IPs<br/>website/mod.rs:262"]
    REQ["GET with User-Agent remove-ai-marks-audit/1.0<br/>website/mod.rs:271"]
    ST{"status<br/>website/mod.rs:275"}
    ERR["not 2xx or a redirect: Err HTTP status<br/>website/mod.rs:277"]
    RD{"301, 302, 303, 307 or 308 with a Location header?<br/>website/mod.rs:285"}
    HOP{"hop already at MAX_REDIRECTS?<br/>website/mod.rs:287"}
    TOO["Err too many redirects<br/>website/mod.rs:288"]
    JOIN["join_url resolves a relative Location, then the loop<br/>revalidates the new URL as the next hop<br/>website/mod.rs:290"]
    BODY["read at most max bytes plus one<br/>website/mod.rs:297"]
    BIG{"more than max bytes?<br/>website/mod.rs:301"}
    OVER["Err exceeds, nothing is truncated or inspected<br/>website/mod.rs:302"]
    OK["Ok bytes, Content-Type<br/>website/mod.rs:304"]
    START --> VT --> SET --> PIN --> REQ --> ST
    ST -->|error| ERR
    ST -->|2xx or redirect| RD
    RD -->|yes| HOP
    HOP -->|yes| TOO
    HOP -->|no| JOIN
    JOIN --> VT
    RD -->|no| BODY --> BIG
    BIG -->|yes| OVER
    BIG -->|no| OK
    OPEN["DEBT: the expected origin never advances, so after an<br/>http to https upgrade a hop back to http is still allowed<br/>website/mod.rs:254"]
    SET --> OPEN
```

**What it shows.** Each hop is validated and connected independently, and the
client is built with automatic redirects switched off so the code, not the
library, decides whether to follow one. A redirect status with no `Location`
falls through to the body read and is treated as an ordinary response.

**Why it is this way.** A response over the cap is abandoned rather than
truncated, so a partial file is never inspected
(`../prose-sanitiser/crates/cli/src/audit/website/mod.rs:19`-`21`). The client
resolver is replaced so that Host and TLS server name still come from the URL
while the socket address comes from the validated list (`:54`-`59`).

**Invariant:** a redirect cannot leave the validated origin, because the new URL
goes back through `validated_target` before any connection is made
(`../prose-sanitiser/crates/cli/src/audit/website/mod.rs:252`-`253`, then `:290`).

**Debt:** the redirect policy allows http, then https, then http, because the
expected origin is set once and compared by equality or the upgrade rule
(`../prose-sanitiser/crates/cli/src/audit/website/mod.rs:254`-`256`,
`../prose-sanitiser/crates/cli/src/audit/website/net.rs:128`-`137`). The
external review of 2026-09-03 recorded the same observation
(`../prose-sanitiser/docs/prose-sanitiser-external-review-2026-09-03.md:71`).

## PS-02.3 The URL policy and the address check

```mermaid
flowchart TB
    U["url_origin url<br/>net.rs:113"]
    P["parse_url: minimal parser, anything it cannot parse<br/>confidently is rejected<br/>net.rs:36"]
    S{"scheme is http or https?<br/>net.rs:115"}
    C{"user:password@ present?<br/>net.rs:118"}
    H{"host is non-empty?<br/>net.rs:121"}
    EX{"expected origin given?<br/>net.rs:312"}
    OA{"same origin, or http 80 to https 443 on the same host?<br/>net.rs:128"}
    XO["Err cross-origin URL is not allowed<br/>net.rs:314"]
    R["resolve_public_addresses<br/>net.rs:271"]
    LIT{"host is an IP literal, else DNS lookup<br/>net.rs:275"}
    EVERY{"every address is_global? one miss refuses the name<br/>net.rs:293"}
    NOP["Err refusing non-public address<br/>net.rs:294"]
    NONE["Err resolved to no IP addresses<br/>net.rs:300"]
    GOOD["origin and the validated addresses<br/>net.rs:321"]
    U --> P --> S
    S -->|no| REJ["Err unsupported URL scheme<br/>net.rs:116"]
    S -->|yes| C
    C -->|yes| CRED["Err credentials in URLs are not allowed<br/>net.rs:119"]
    C -->|no| H
    H -->|no| NOH["Err URL has no hostname<br/>net.rs:122"]
    H -->|yes| EX
    EX -->|yes| OA
    OA -->|no| XO
    OA -->|yes| R
    EX -->|no| R
    R --> LIT --> EVERY
    EVERY -->|no| NOP
    EVERY -->|yes| ANY{"at least one address?<br/>net.rs:300"}
    ANY -->|no| NONE
    ANY -->|yes| GOOD
```

**What it shows.** The checks run in a fixed order, cheapest first: scheme,
credentials, host, origin, and only then DNS and the address test. A cross-origin
URL is refused before any lookup is made.

**Why it is this way.** A name with one private answer among several public ones
is refused outright, not filtered to its public answers, because it is not a
name the audit should follow
(`../prose-sanitiser/crates/cli/src/audit/website/net.rs:266`-`270`). An
IPv4-mapped IPv6 address is unwrapped to its IPv4 form before the test
(`:286`-`292`). The host is lowercased and loses a trailing dot, so
`Example.COM.` and `example.com` are the same origin (`:85`).

**Invariant:** a URL with credentials is rejected, never transmitted
(`../prose-sanitiser/crates/cli/src/audit/website/net.rs:118`-`120`).

**Invariant:** the only cross-origin move allowed is the same host from port 80
http to port 443 https, and not the reverse
(`../prose-sanitiser/crates/cli/src/audit/website/net.rs:132`-`136`).

## PS-02.4 Which IPv6 addresses count as public

```mermaid
flowchart TB
    A["is_global address<br/>net.rs:142"]
    V4["IPv4, or IPv6 mapped to IPv4: is_global_v4<br/>net.rs:146"]
    V4R["refused: 0.0.0.0/8, loopback, private, link-local,<br/>broadcast, documentation, multicast, 100.64/10,<br/>192.0.0/24, 198.18/15, 240/4<br/>net.rs:154"]
    N64{"64:ff9b::/96 NAT64<br/>net.rs:207"}
    N64A["public only if the embedded IPv4 is public<br/>net.rs:208"]
    N64L["64:ff9b:1::/48 local-use NAT64: refused<br/>net.rs:211"]
    S6["2002::/16 6to4: public only if the embedded IPv4 is<br/>net.rs:215"]
    IE["2001::/23: refused, except the registered<br/>globally reachable sub-allocations<br/>net.rs:225"]
    D3["3fff::/20 documentation: refused<br/>net.rs:229"]
    TR["::ffff:0:0:0/96 IPv4-translated: refused<br/>net.rs:234"]
    IS["ISATAP identifier with a non-public IPv4: refused<br/>net.rs:240"]
    CP["IPv4-compatible ::a.b.c.d: judged by the embedded IPv4<br/>net.rs:247"]
    GEN["then the plain ranges: unspecified, loopback, multicast,<br/>fc00::/7, fe80::/10, fec0::/10, 2001:db8::/32, 100::/64<br/>net.rs:251"]
    A --> V4 --> V4R
    A --> N64
    N64 --> N64A
    N64 --> N64L
    N64L --> S6 --> IE --> D3 --> TR --> IS --> CP --> GEN
```

**What it shows.** An IPv6 address can carry an IPv4 destination in several
transition forms. Each form is judged by the IPv4 it carries, or refused where it
can tunnel to an arbitrary host, before the ordinary IPv6 ranges are tested.

**Why it is this way.** Without the transition rules an attacker-controlled AAAA
record such as `64:ff9b::7f00:1` would pass as global while a translator
delivered the connection to an internal IPv4 host
(`../prose-sanitiser/crates/cli/src/audit/website/net.rs:201`-`205`). Teredo is
refused outright rather than unpacked (`:219`-`224`). Rust's own `is_global` is
unstable, so the ranges are spelled out (`:139`-`141`).

**Invariant:** every refused family above is exercised by a test with concrete
addresses, including the cloud metadata address
(`../prose-sanitiser/crates/cli/src/audit/website/net.rs:414`-`436`).

## PS-02.5 Sitemap discovery and collection

```mermaid
flowchart TB
    DISC["discover_sitemap base URL<br/>website/mod.rs:328"]
    STD["try base/sitemap.xml, then base/sitemap_index.xml,<br/>each fetched same-origin and parsed<br/>website/mod.rs:332"]
    ROB["else robots.txt, first Sitemap line only, 1 MiB cap<br/>website/mod.rs:343"]
    OFF{"candidate origin allowed?<br/>website/mod.rs:358"}
    NONEF["no sitemap: Ok None, which is exit 2 upstream<br/>website/mod.rs:361"]
    COL["collect_urls sitemap URL<br/>website/mod.rs:371"]
    REC["recurse: stop at max pages or depth above 3<br/>website/mod.rs:401"]
    PS["parse_sitemap: gzip inflated under a 64 MiB cap,<br/>namespace prefixes stripped, loc entries unescaped<br/>website/mod.rs:149"]
    XO{"each loc same-origin?<br/>website/mod.rs:409"}
    ABORT["whole crawl fails: cross-origin sitemap URL<br/>website/mod.rs:410"]
    IDX{"root element is sitemapindex?<br/>website/mod.rs:418"}
    NEST["recurse one level deeper<br/>website/mod.rs:419"]
    ADD["collect the page URL, stop at max pages<br/>website/mod.rs:421"]
    DISC --> STD --> ROB --> OFF
    OFF -->|no| NONEF
    COL --> REC --> PS --> XO
    XO -->|no| ABORT
    XO -->|yes| IDX
    IDX -->|yes| NEST --> REC
    IDX -->|no| ADD
```

**What it shows.** Discovery tries two standard paths, then `robots.txt`.
Collection is a bounded recursion over sitemap indexes. One off-site entry
anywhere aborts the crawl, rather than being skipped.

**Why it is this way.** A `robots.txt` that names an off-site sitemap is a
redirect by another name, so it is refused
(`../prose-sanitiser/crates/cli/src/audit/website/mod.rs:359`-`361`). The
decompression is stopped at the cap while inflating, not checked afterwards,
because it is the zip-bomb guard (`:26`-`30`, `:153`-`162`).

**Debt:** the recursion limits depth and the final page count but not the number
of sitemaps fetched, so a wide tree of nested indexes costs one request per index
(`../prose-sanitiser/crates/cli/src/audit/website/mod.rs:401`,
`../prose-sanitiser/docs/prose-sanitiser-external-review-2026-09-03.md:70`).

**Drift:** `docs/UPSTREAM-DELTA.md` lists SSRF hardening for the crawler as an
open item, saying `audit-website` does not yet refuse private hosts or pin
redirects (`../prose-sanitiser/docs/UPSTREAM-DELTA.md:37`). The code refuses
non-public addresses and revalidates every redirect
(`../prose-sanitiser/crates/cli/src/audit/website/net.rs:293`-`295`).

## PS-02.6 What happens to a downloaded asset

```mermaid
sequenceDiagram
    autonumber
    participant IR as inspect_remote<br/>website/mod.rs:310
    participant GK as guess_kind<br/>website/mod.rs:71
    participant SF as scan_file<br/>audit/mod.rs:66
    participant II as inspect_image<br/>image/mod.rs:112
    participant SY as run_synthid_score<br/>harness.rs:122
    IR->>GK: URL, bytes, Content-Type (website/mod.rs:311)
    GK-->>IR: a kind, from the header, then the URL suffix, then magic bytes (website/mod.rs:79-145)
    IR->>IR: write the bytes to a fresh temp dir as asset.ext (website/mod.rs:313-319)
    IR->>SF: temp path, the URL as the display name (website/mod.rs:320)
    SF->>II: an image kind goes to inspect_image with no scorer dir (audit/mod.rs:98)
    II->>SY: REVERSE_SYNTHID_DIR, when set, still runs the scorer (image/mod.rs:165)
    Note over SY: DRIFT: website/mod.rs:5 says external tools<br/>are not invoked for remote URLs
```

**What it shows.** A download is classified, written to a temporary file and run
through the local `scan_file` pipeline, so the website audit reuses the directory
audit's inspection exactly. Classification prefers the `Content-Type` header,
then the URL suffix, then the first bytes.

**Why it is this way.** The module header chooses to reuse one pipeline so the
two audits cannot diverge
(`../prose-sanitiser/crates/cli/src/audit/website/mod.rs:3`-`6`). The c2patool
and exiftool cross-checks are behind the non-default `external-verify` feature
and report unavailable otherwise
(`../prose-sanitiser/crates/media/src/image/tools.rs:11`-`13`).

**Drift:** the module header says the optional external tools are not invoked for
remote URLs (`../prose-sanitiser/crates/cli/src/audit/website/mod.rs:5`-`6`).
`inspect_image` calls the SynthID scorer whenever `REVERSE_SYNTHID_DIR` is set
(`../prose-sanitiser/crates/media/src/image/mod.rs:165`, then
`../prose-sanitiser/crates/media/src/image/harness.rs:123`-`125`), so a downloaded
image can start that subprocess. The external review made the same observation
for the earlier tool set
(`../prose-sanitiser/docs/prose-sanitiser-external-review-2026-09-03.md:69`).
