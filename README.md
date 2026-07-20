# Wafer Map Generator

Synthetic wafer bin map generator for semiconductor pre-sales demos. Configure wafer geometry and a spatial defect signature, then export CSV, PNG, ZIP, and STDF files suitable for loading into analytics tools such as Exensio.

## What it does

Most presales demos run without real customer data. This tool lets you answer questions like:

> "What if we had edge ring failures on 300 mm wafers?"

You describe the scenario (form or natural language), the app generates realistic-looking wafer maps and test data, and you download the outputs for demo ingestion.

## Features

- **37 spatial signatures** — Edge Ring, Center Cluster, Scratch families, Reticle Pattern (hard & soft repeaters), Striping (lens tilt), Mixed Mode, and more
- **Spec-compliant geometry** — 150/200/300 mm wafers with auto flat/notch, 1–10 mm edge exclusion, die aspect-ratio validation (1:2 to 2:1), 0.05–0.2 mm scribe street, auto stepping field
- **Yield model** — direct yield % or defect density via `Y = e^(-A·D)` (Poisson)
- **CP1/CP2/CP3 insertions** — retest cascade where CP2/CP3 keep 90–99.9% of prior passers
- **Story 1: ECID matching / FT traceability** — full spec 1.a-g coverage:
  - 1:1 and sweeper lots, blank-ECID join traps, wrong-bin / wrong-XY assembly errors
    (~100 and ~1000 GDPW), simple vs subtle FT fail rates
  - ECID encoding variants — plain concatenated `(lot)(wafer)(x)(y)`, ROT13 "encrypted"
    (unique but not map-readable), or split into 4 separate test items that must be
    re-concatenated to join (spec 1.b)
  - GDBN / missed-CP-cluster low yield at FT — dramatic (clean CP, FT-only donut
    pattern) and good-die-bad-neighborhood (CP scratch + adjacent passers risk FT
    fail) cases (spec 1.g)
  - Multi-die product traceability (Case B) — 3-component packaged products with
    full or partial (1-of-3 untraceable) component ECID traceability (spec 1.c)
  - Downloads: CP CSV (+ECID / split items), FT units, match ground truth, multi-die
    products CSV, manifest
- **Configurable bins** — 16/64/256 hardbins, softbins ×4/×16/×64
- **Test items** — 100..1M tests, pass/fail vs parametric split, five parametric data shapes, verbose test-name modes for UI stress testing
- **Fab realism** — FYYWWSSSS lot numbers, multi-lot time sequences for trend charts, 1–600 s test time, multi-site (1–16 sites from GDPW) with layout patterns and site-to-site yield loss
- **Manual configuration tab** — every parameter above as a form control
- **Stories tab** — named Story 1 scenarios with knobs
- **AI Chat Assistant tab** — natural language input via Azure OpenAI (GPT-4.1) with keyword-parser fallback
- **Exports**
  - CSV with die-level data (`Insertion`, `Bin`, `HardBin`, `SoftBin`, `Site`, `ECID`, coordinates, bin metadata)
  - Optional FT unit CSV + ECID match ground-truth CSV (Story 1)
  - Optional long-format per-test CSV
  - PNG/SVG/JPEG/TIFF wafer map grid and per-wafer ZIP
  - STDF v4 binary per lot per insertion (FAR, MIR, SDR, WIR/WRR, PIR/PTR/PRR, TSR, HBR/SBR, MRR)

## Project structure

| File | Role |
|------|------|
| `app.py` | Streamlit web UI (chat + manual + Stories) |
| `generator.py` | The pipeline: signature → yield → CP cascade → bins → optional Story 1 FT |
| `geometry.py` | Die grid, spec limits, auto flat/notch, auto stepping field |
| `signatures.py` | Spatial defect patterns and internal bin definitions |
| `yield_model.py` | Direct/defect-density yield targets, CP retest cascade, S2S |
| `binning.py` | Internal bin → hardbin/softbin mapping |
| `ecid.py` | ECID burn / blank rules, encoding variants (plain/ROT13), split-test-item helpers |
| `assembly.py` | Assembly picks: 1:1, sweeper, wrong-bin, wrong-XY |
| `gdbn.py` | Story 1g: GDBN / missed-CP-cluster low yield at FT |
| `multidie.py` | Story 1c: multi-die (Case B) packaged product traceability |
| `final_test.py` | FT outcomes + match ground-truth tables |
| `story1_presets.py` | Story 1 scenario presets (GDPW, knobs) for all sub-stories |
| `test_items.py` | Test names, parametric/pass-fail values, data shapes |
| `fab.py` | Fab lot numbers, lot schedules, multi-site, test-time math |
| `renderer.py` | Draws wafer map images |
| `llm_agent.py` | Parses chat prompts into generation parameters |
| `stdf_writer.py` | Builds STDF binary output |
| `tests/test_story1_ecid.py` | Hard edge-case tests for Story 1 (1:1/sweeper/wrong-bin/wrong-XY, ECID encoding) |
| `tests/test_story1_gdbn.py` | Tests for Story 1g (GDBN / missed-CP-cluster) |
| `tests/test_story1_multidie.py` | Tests for Story 1c (multi-die / Case B traceability) |
| `Spec_Implementation_Audit.md` | Spec-by-spec map of what is implemented, where, and how |

## Spec coverage & known limitations

See [`Spec_Implementation_Audit.md`](Spec_Implementation_Audit.md) for a full spec-by-spec
breakdown. Summary:

- **All Must-Have items (1–12) are implemented**, plus most Nice-To-Haves (lot numbers,
  lot sequence, test time, multi-site, site-to-site yield loss).
- **Known limitations:**
  - *Splits / child lots (`.01`, `.02`)* — `fab.make_lot_id()` supports the suffix, but it
    is not yet wired into the UI or generation pipeline.
  - *Repair (virgin vs repaired good die)* — intentional placeholder; not implemented.
  - *Parametric values* are currently drawn from a **uniform** distribution (not Gaussian);
    a documented simplification.
  - *Story 1c Case C/D (IDM/Foundry with ALPS/factory data)* — the spec itself marks
    this "TK" (needs ALPS data); not implemented. Case B's component populations are
    independent Bernoulli-yield pools rather than full spatial wafer maps, per the
    spec's own "simple case" guidance.

## Requirements

- Python 3.10+ (tested with 3.14 via `py` launcher on Windows)
- Dependencies in `requirements.txt`

## Setup

```powershell
cd SpacialSignatures
py -m pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` with your Azure OpenAI credentials (optional — keyword parser works without them):

```env
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

**Do not commit `.env`.** It is listed in `.gitignore`.

## Run the app

On Windows, use the Python launcher:

```powershell
py -m streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

## Deploy for Steve (Streamlit Community Cloud)

The easiest way to share a live link is [Streamlit Community Cloud](https://share.streamlit.io) (free).

### 1. Deploy the app

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with **GitHub** (`SaiRohitaBhaskaruniPDF`).
2. Click **Create app**.
3. Select repository: `SaiRohitaBhaskaruniPDF/WaferMapGenerator`
4. Branch: `main`
5. Main file: `app.py`
6. Click **Deploy**.

If the repo is **private**, grant Streamlit access first:
- Streamlit Cloud → your username → **Settings** → **Linked accounts** → connect private repo access.

### 2. Add Azure secrets (for AI Chat tab)

In the deployed app → **Settings** → **Secrets**, paste:

```toml
AZURE_OPENAI_API_KEY = "your-key"
AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT = "your-deployment-name"
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
```

Click **Save**. The app will reboot with AI chat enabled. Manual tab works without secrets.

### 3. Share with Steve

| Option | How |
|--------|-----|
| **Private app + invite** | App settings → **Sharing** → add Steve's email as **Viewer** (best if repo stays private) |
| **Public app link** | App settings → set visibility to **Public** → send URL `https://your-app.streamlit.app` |

Private GitHub repo → app is private by default. Steve needs an invite unless you make the app public.

### 4. After you push code updates

Streamlit redeploys automatically when you `git push` to `main`.

## Usage

### Manual tab

1. Set wafer diameter, die size, edge exclusion, and signature
2. Enter lot ID and number of wafers
3. Click **Generate**
4. Review maps, yield summary, and CSV preview
5. Download CSV, PNG, ZIP, or STDF

### AI Chat tab

1. Configure Azure credentials in `.env` or the in-app expander
2. Type a request, e.g. *"Give me 6 wafers with edge ring failures on 300 mm wafers — lot DEMO_A01"*
3. Click **Send**
4. Download the generated outputs

## How the data is generated

All output is **synthetic** — not from a real fab or external dataset.

1. `geometry.py` places dies on a grid and keeps only positions inside `(radius - edge_exclusion)`
2. `signatures.py` assigns pass/fail bins using geometry rules plus seeded randomness (e.g. edge ring ≈ 88% fail rate in the outer band)
3. `app.py` assembles rows into CSV and calls the renderer/STDF writer

Example: 300 mm wafer, 10×10 mm dies, 3 mm edge exclusion → **673 dies per wafer**.

## Demo workflow (Exensio)

```
Customer question → AI chat or manual config → Generate → Download STDF → Ingest in Exensio → Gallery demo
```

## License

Internal / project use — confirm with your team before external distribution.
