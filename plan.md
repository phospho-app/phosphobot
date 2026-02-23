# Rebranding Plan: phospho/phosphobot → Factory Intelligence

## Strategy: Minimal Surgical Changes for Easy Upstream Sync

The key insight: we want to **minimize merge conflicts** when syncing with `phospho-app/phosphobot` upstream. To achieve this, we:

1. **Create a single branding constants file** (`dashboard/src/lib/branding.ts`) that centralizes all brand strings
2. **Only modify UI-visible branding** — leave Python package names, import paths, and internal config untouched
3. **Replace hardcoded brand strings** with imports from the constants file
4. **Add a Factory Intelligence favicon/logo** to replace the phospho one

This approach means most files only gain a single import line change, and the branding file itself is new (no conflict possible).

---

## Files to Create (1 new file)

### `dashboard/src/lib/branding.ts`
Centralized branding constants:
- `BRAND_NAME` = "Factory Intelligence"
- `PRODUCT_NAME` = "Factory Intelligence" (was "phosphobot")
- `BRAND_URL` = your website URL
- `DOCS_URL` = your docs URL
- `GITHUB_URL` = your GitHub URL
- `DISCORD_URL` = your Discord URL (or remove)
- `SUPPORT_EMAIL` = your contact email
- `PRO_URL` = your pro/upgrade URL (or remove pro callouts)
- `HOTSPOT_SSID` / `HOTSPOT_PASSWORD` = network defaults

---

## Files to Modify (14 dashboard files + 1 HTML)

### Tier 1: Core Branding (most visible, change first)

| # | File | What Changes | Lines |
|---|------|-------------|-------|
| 1 | `dashboard/index.html` | Page title: "phospho dashboard" → "Factory Intelligence" | L7 |
| 2 | `dashboard/src/components/layout/topbar.tsx` | Header text "phosphobot" → brand constant; docs.phospho.ai URL; phospho.ai/pro URL; billing.phospho.ai URL; contact email | L247, L254, L192, L204, L215, L269 |
| 3 | `dashboard/src/components/layout/footer.tsx` | GitHub link; "phosphobot" in brew update commands; install URL | L96, L155-165, L186-189 |
| 4 | `dashboard/src/pages/auth/AuthForm.tsx` | "phosphobot account" → "Factory Intelligence account" | L22, L25, L77 |
| 5 | `dashboard/src/pages/DashboardPage.tsx` | Import of PhosphoProCallout (keep or replace with own callout) | L1, L248 |

### Tier 2: Feature Pages

| # | File | What Changes | Lines |
|---|------|-------------|-------|
| 6 | `dashboard/src/pages/BrowsePage.tsx` | "phosphobot" breadcrumb label; `~/phosphobot/recordings` tooltip | L567, L572 |
| 7 | `dashboard/src/pages/AIControlPage.tsx` | "phosphobot records data in radians" tooltip text | L399-401 |
| 8 | `dashboard/src/pages/AITrainingPage.tsx` | Import of PhosphoProCallout; phospho.ai/pro link | L1, L281, L519 |
| 9 | `dashboard/src/pages/AdminSettingsPage.tsx` | "phospho-app organization"; "phospho pro" text/link; phospho.ai/pro URL | L179, L183, L189-191 |
| 10 | `dashboard/src/pages/NetworkPage.tsx` | Hotspot SSID "phosphobot"; password "phosphobot123" | L80, L203-206 |
| 11 | `dashboard/src/pages/calibration/calibration-sequence.tsx` | docs.phospho.ai URL; "phospho.ai customer"; robots.phospho.ai URL | L98, L242-248 |
| 12 | `dashboard/src/pages/control/VRControlPage.tsx` | Import of PhosphoVRCallout (references Phospho Pro) | L1 |

### Tier 3: Internal/Config

| # | File | What Changes | Lines |
|---|------|-------------|-------|
| 13 | `dashboard/src/components/callout/phospho-pro.tsx` | Entire component text references "phospho" — rebrand or replace | All |
| 14 | `dashboard/src/components/callout/phospho-vr.tsx` | "Phospho Pro" / "phospho teleoperation" / Meta Store references | All |
| 15 | `dashboard/src/components/common/add-robot-connection.tsx` | Robot type id "phosphobot" / name "Remote phosphobot server" | L51-52 |
| 16 | `dashboard/src/lib/hooks.ts` | Zustand store key "phosphobot-global-store" (internal, low impact) | L182 |

### Tier 4: Static Assets

| # | File | What Changes |
|---|------|-------------|
| 17 | `dashboard/public/favicon.ico` (or added to index.html) | Add Factory Intelligence favicon |
| 18 | `phosphobot/resources/swagger-ui/favicon.png` | Replace with Factory Intelligence favicon (API docs page) |

---

## What NOT to Change (to maintain upstream sync)

- **Python package name** (`phosphobot/`) — renaming would break all imports across 57+ files
- **CLI entry point** (`phosphobot` command) — defined in pyproject.toml, would break user installs
- **Config paths** (`~/phosphobot/recordings`) — would require migration scripts
- **Python internal references** — sentry DSN, posthog keys, supabase URLs are all backend config
- **README.md, CONTRIBUTING.md** — these get overwritten on every sync anyway
- **Cloud frontend** (`cloud/frontend/`) — separate deployment, not part of local dashboard
- **Dataset viewer** (`dataset-viewer/`) — no phospho brand references found

---

## Color/Theme Changes

The current accent color is **green-500** (used throughout for buttons, icons, status indicators).

Based on the Factory Intelligence logo, we can optionally change the accent color. The current theme in `index.css` uses neutral/achromatic CSS variables, and green is applied via Tailwind utility classes (`text-green-500`, `bg-green-500`).

**Recommendation**: Keep this as a separate follow-up change since the green accent appears in 50+ places across the dashboard components. For now, the text branding change is the highest-impact, lowest-conflict change.

---

## Implementation Order

1. Create `branding.ts` constants file
2. Update `index.html` title
3. Update `topbar.tsx` (most visible branding)
4. Update `footer.tsx`
5. Update `auth/AuthForm.tsx`
6. Update callout components (`phospho-pro.tsx`, `phospho-vr.tsx`)
7. Update remaining pages (BrowsePage, AIControlPage, AITrainingPage, AdminSettingsPage, NetworkPage, CalibrationPage, VRControlPage)
8. Update `add-robot-connection.tsx` and `hooks.ts`
9. Add favicon
10. Commit and push
