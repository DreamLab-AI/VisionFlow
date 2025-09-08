# 🚨 CRITICAL SECURITY ALERT

## Malware Detected in NPM Dependencies

**Date:** 2025-09-08
**Severity:** CRITICAL
**Status:** Partially Mitigated

### Affected Packages with Malware

The following packages have been compromised with malware:
- `ansi-regex` (all versions)
- `ansi-styles` (all versions)
- `color-name` (all versions)
- `color-convert` (versions >=1.6.0)
- `supports-color` (all versions)

### Impact

These packages are fundamental dependencies used by:
- @testing-library/react
- @testing-library/dom
- @vitest/coverage-v8
- vitest
- Pretty much any testing framework

### Mitigation Applied

1. **Updated to latest versions** of testing frameworks (vitest 2.1.8)
2. **Added npm overrides** to force safe versions:
   ```json
   "overrides": {
     "ansi-regex": "6.1.0",
     "ansi-styles": "6.2.1",
     "color-name": "2.0.0",
     "color-convert": "2.0.1",
     "supports-color": "9.4.0"
   }
   ```

### Remaining Issues

Despite overrides, some packages still pull in malicious versions through complex dependency chains. The testing libraries appear to be the main vector.

### Recommendations

1. **IMMEDIATE ACTION REQUIRED:**
   - Consider temporarily removing testing dependencies until the supply chain attack is resolved
   - Monitor GitHub advisories for updates
   - Consider using alternative testing frameworks

2. **Alternative Testing Solutions:**
   - Use Jest directly instead of vitest
   - Use React Testing Library alternatives
   - Run tests in isolated containers

3. **Security Monitoring:**
   - Run `npm audit` regularly
   - Use Snyk or similar tools for continuous monitoring
   - Enable GitHub Dependabot alerts

### Temporary Workaround

To completely remove the vulnerable packages, you can temporarily remove testing dependencies:

```bash
npm uninstall @testing-library/react @testing-library/jest-dom @vitest/coverage-v8 @vitest/ui vitest
```

Then run tests using Docker or isolated environments only.

### References

- https://github.com/advisories/GHSA-jvhh-2m83-6w29 (ansi-regex)
- https://github.com/advisories/GHSA-p5rr-crjh-x7gr (ansi-styles)
- https://github.com/advisories/GHSA-m99c-cfww-cxqx (color-name)
- https://github.com/advisories/GHSA-pj3j-3w3f-j752 (supports-color)

### Status

This is an active supply chain attack. The malicious packages appear to be exfiltrating data or providing backdoor access. DO NOT run tests on production systems until this is resolved.