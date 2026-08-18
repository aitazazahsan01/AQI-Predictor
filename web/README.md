# Pearls AQI — website

The public front end: a statically exported Next.js site built on the
**Modernist** design system.

## How it gets its data

The site has no server, no database and no credentials. Everything it renders
comes from one file:

```
web/public/data/forecast.json
```

which the Python pipeline writes:

```
python scripts/export_web_data.py          # or:  ./run.sh export-web
```

That script is the only thing that talks to Hopsworks, loads the models and
runs SHAP, and it runs where the secrets already live — inside GitHub Actions,
straight after the nightly training job. It commits the refreshed snapshot back
to `main`, which triggers a redeploy.

The payload shape is defined by `src/inference/snapshot.py` and mirrored in
[`src/lib/types.ts`](src/lib/types.ts). `schema_version` is checked at build
time: a mismatch fails the build rather than shipping a page with empty panels.

## Local development

```bash
npm install
npm run dev        # http://localhost:3000
```

`forecast.json` is committed, so the site runs offline with whatever snapshot
was published last. To refresh it against live data, run the export script from
the repository root first.

Other scripts:

| Command | What it does |
| --- | --- |
| `npm run build` | Static export into `out/` |
| `npm run typecheck` | `tsc --noEmit` |

## Deployment

**Vercel** is what this deploys to. Import the repository, then set:

| Setting | Value |
| --- | --- |
| Root Directory | `web` |
| Framework Preset | Next.js *(auto-detected)* |
| Build Command | *(leave default)* |
| Environment variables | *(none — `BASE_PATH` stays unset)* |

Vercel redeploys on every push to `main`, which includes the nightly
`data: refresh forecast snapshot` commit from the pipeline. Nothing else needs
wiring up.

**Any other static host** works too: `npm run build` produces a plain folder of
HTML in `out/`. If you serve it from a subpath (GitHub Pages serves at
`/<repo>/`), build with `BASE_PATH=/<repo>` so asset URLs resolve.

## Design system

The look comes from the Claude Design project **Modernist**: flat, architectural,
set entirely in Archivo, near-mono red on a light ground, zero corner radius,
strong 2px rules, photography in black and white.

- [`src/styles/modernist.css`](src/styles/modernist.css) is **vendored from that
  project — treat it as read-only.** The single local change is the removed
  Google Fonts `@import`; `next/font` self-hosts Archivo instead and rebinds
  `--font-heading` / `--font-body` to it.
- [`src/styles/app.css`](src/styles/app.css) adds only page scaffolding the
  system does not ship: the shell, section rhythm and modular grid.
- Component styles are CSS Modules that read exclusively from the tokens
  (`--color-*`, `--space-*`, `--rule`). No raw hex, no invented font.

**One deliberate exception.** The AQI category colours (`#00E400` green through
`#7E0023` maroon) are inlined from the payload rather than themed. They are a
published US EPA standard, not a brand decision — recolouring them to fit the
palette would misinform the reader.

Charts are hand-built SVG rather than a charting library: flat, square-capped
and ruled, which is what the system asks for and what a library would fight.
