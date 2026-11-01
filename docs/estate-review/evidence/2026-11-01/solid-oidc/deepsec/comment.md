## 🔎 deepsec found 5 findings

🟡 MEDIUM ×4 · 🟣 BUG ×1

<sub>scope: `git-diff:working` · run `20261101200112-ef31c994be2369dc`</sub>

### 🟡 MEDIUM · `crates/solid-pod-rs/src/oidc/jwks.rs:L100-L411`

**OidcConfigCache / JwksCache grow without bound; stale entries never evicted**
<sub>slug: `other-cache-exhaustion` · confidence: medium</sub>

OidcConfigCache and JwksCache store entries in a HashMap with no maximum capacity and no eviction. get() checks the TTL and returns None when stale but never removes the expired entry (lines 114-122, 158-166); put() only ever inserts (lines 124-134, 168-178). The cache key is canonical_issuer(issuer), i.e. an attacker-influenced issuer URL string. An attacker who can drive CachedFetcher::config / jwks / verify_access_token_cached with a stream of distinct issuer URLs (each a reachable public host returning a discovery doc whose issuer field matches — e.g. attacker-hosted) causes both maps to…

**Recommendation:** Bound the caches with a maximum entry count and an eviction policy (LRU or periodic sweep of expired entries), and/or constrain the fetcher to an operator-configured allowlist of trusted issuers before performing any network fetch or cache insert.

---

### 🟡 MEDIUM · `crates/solid-pod-rs/src/oidc/jwks.rs:L229-L313`

**SSRF error messages embed resolved internal IPs and are wrapped verbatim into PodError**
<sub>slug: `other-info-disclosure` · confidence: low</sub>

On a blocked or failed SSRF check, SsrfError's Display embeds the resolved IP and classification (e.g. "host 'x' (resolved to 10.1.2.3) blocked: Private"; DnsFailure vs BlockedClass are distinguishable). jwks.rs wraps these verbatim into PodError::Nip98 at lines 227-230 (issuer) and 310-313 (jwks_uri). The access-token verification path documents PodError::Nip98 as surfaced to clients as a 401/invalid_token challenge. Because the caller chooses the issuer URL, a reachable error string (or merely a distinguishable error outcome) turns the pod into a blind internal-DNS reconnaissance oracle: th…

**Recommendation:** Map SSRF failures to a generic, non-specific error for any client-facing surface (log the detailed resolved-IP string server-side only). Do not propagate resolved IPs or the DNS-failure/blocked-class distinction to callers who control the target URL.

---

### 🟡 MEDIUM · `crates/solid-pod-rs/src/oidc/jwks.rs:L201-L216`

**SSRF/DNS-rebinding pin bypassed when an HTTP proxy is configured in the environment**
<sub>slug: `ssrf` · confidence: medium</sub>

pinned_client() builds a fresh reqwest::Client with .resolve(host, approved_ip:port) and .redirect(none) to force the TCP connect at the SSRF-approved IP, defeating DNS rebinding between the policy check and the connect. However reqwest::Client::builder() honours the standard HTTP_PROXY / HTTPS_PROXY / ALL_PROXY environment variables by default, and `default-features = false` in Cargo.toml (reqwest 0.12, features json+rustls-tls) does NOT disable that — proxy-from-env is runtime behaviour, not a cargo feature. When any such proxy var is present in the pod's environment, reqwest routes the dis…

**Recommendation:** Add .no_proxy() to the pinned_client builder so the SSRF-approved IP pin is always authoritative, or, if proxying is a deliberate deployment requirement, re-run the SSRF policy against the proxy's resolution and document that outbound OIDC fetches must not traverse an untrusted proxy.

---

### 🟡 MEDIUM · `crates/solid-pod-rs/src/oidc/jwks.rs:L273-L343`

**Unbounded response body on discovery and JWKS fetch enables memory-exhaustion DoS**
<sub>slug: `expensive-api-abuse` · confidence: medium</sub>

Both fetch_oidc_config (resp.json(), lines 273-276) and fetch_jwks (resp.json(), lines 340-343) deserialize the entire HTTP response body into memory with no byte cap. The documented 'bounded timeout' (FETCH_TIMEOUT 10s) limits wall-clock, not response size. Any host that passes the SSRF classification — i.e. any public IP, including one the attacker controls and points at via a token's issuer — can return (or stream, within 10s) an arbitrarily large body, driving the pod to allocate until OOM. Because the issuer that selects the target host is attacker-influenced and this fetch runs before t…

**Recommendation:** Cap the readable body (e.g. stream with a hard max length / Content-Length precheck, or reqwest response .bytes() with a size guard) before JSON parsing; reject oversized discovery docs and JWKS. A few hundred KB is a generous ceiling for both.

---

### 🟣 BUG · `crates/solid-pod-rs/src/oidc/jwks.rs:L303-L410`

**JWKS path never consults the discovery (config) cache, double-fetching on every JWKS miss**
<sub>slug: `other-logic-bug` · confidence: high</sub>

CachedFetcher::jwks() checks jwks_cache, and on a miss calls fetch_jwks (line 410), which unconditionally calls the *uncached* fetch_oidc_config (line 303). The OidcConfigCache populated by CachedFetcher::config() is therefore never consulted on the token-verification path. Every JWKS cache miss re-runs the full discovery fetch — a second SSRF resolve+connect and HTTP round-trip — even when a fresh discovery document is already cached. This defeats the stated purpose of caching the discovery doc (amortising the SSRF handshake, module header point 5) and doubles outbound request volume and lat…

**Recommendation:** Have the cached JWKS path reuse CachedFetcher::config() (or an internal cached discovery lookup) to obtain the jwks_uri, rather than calling the uncached fetch_oidc_config, so the discovery cache is honoured.

---

<sub>🤖 generated by [deepsec](https://github.com/vercel/deepsec)</sub>