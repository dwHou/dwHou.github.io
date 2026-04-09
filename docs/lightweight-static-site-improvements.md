# Lightweight Static Site Improvements

## Implemented Improvements

### 1. Fixed the contact-page email unscramble trigger
- Files: `contact.html`
- Benefit dimension: bug fix
- Why it was worth changing:
  The email unscramble link markup had malformed attributes, which made the interaction fragile and could prevent the intended click behavior from working correctly.

### 2. Removed duplicate jQuery loads on lightweight utility pages
- Files: `contact.html`, `demo.html`
- Benefit dimension: loading performance, maintainability
- Why it was worth changing:
  Both pages loaded jQuery twice. Removing the duplicate request reduces unnecessary network work and lowers the chance of debugging confusion around script ordering.

### 3. Upgraded shared third-party asset URLs to HTTPS on top-level utility pages
- Files: `about.html`, `contact.html`, `demo.html`
- Benefit dimension: bug prevention, loading reliability
- Why it was worth changing:
  These pages still depended on insecure or protocol-relative third-party asset URLs. Converting them to HTTPS reduces mixed-content risk and improves reliability on modern browsers.

### 4. Added a lightweight local verification script
- Files: `tools/verify-static-site.mjs`
- Benefit dimension: maintainability, regression safety
- Why it was worth changing:
  The repository now has a small local check that validates the accepted improvements without introducing a build system or new frontend runtime.

## Intentionally Unchanged Areas

### 1. Mature profile/about visual design
- Why unchanged:
  The profile/research presentation is already a polished UI area. This change did not identify a lightweight visual improvement strong enough to justify touching it.

### 2. Homepage card structure and bilingual content layout
- Why unchanged:
  The homepage is large and content-heavy, but no lightweight edit was clearly better without risking semantic or visual drift.

### 3. Legacy demo pages under `posts/`
- Why unchanged:
  They are specialized experiences, and this change only targets top-level, high-signal improvements.

### 4. Analytics setup and historical content links
- Why unchanged:
  While some analytics and external links are old, changing them without a stronger product need could affect behavior without meaningful user-facing benefit.

## Local Verification

Run:

```bash
node tools/verify-static-site.mjs
```

This checks:
- the fixed contact email trigger markup
- single jQuery inclusion on `contact.html` and `demo.html`
- HTTPS usage for the upgraded shared third-party URLs
- preservation of key page affordances such as the homepage language toggle and the about-page profile theme
