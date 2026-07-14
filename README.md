# Wafer Map Generator

Synthetic wafer bin map generator for semiconductor pre-sales demos. Configure wafer geometry and a spatial defect signature, then export CSV, PNG, ZIP, and STDF files suitable for loading into analytics tools such as Exensio.

## What it does

Most presales demos run without real customer data. This tool lets you answer questions like:

> "What if we had edge ring failures on 300 mm wafers?"

You describe the scenario (form or natural language), the app generates realistic-looking wafer maps and test data, and you download the outputs for demo ingestion.

## Features

- **33 spatial signatures** — Edge Ring, Center Cluster, Scratch families, Reticle Pattern (hard & soft repeaters), Striping (lens tilt), Mixed Mode, and more
- **Spec-compliant geometry** — 150/200/300 mm wafers with auto flat/notch, 1–10 mm edge exclusion, die aspect-ratio validation (1:2 to 2:1), 0.05–0.2 mm scribe street, auto stepping field
- **Yield model** — direct yield % or defect density via `Y = e^(-A·D)` (Poisson)
- **CP1/CP2/CP3 insertions** — retest cascade where CP2/CP3 keep 90–99.9% of prior passers
- **Configurable bins** — 16/64/256 hardbins, softbins ×4/×16/×64
- **Test items** — 100..1M tests, pass/fail vs parametric split, five parametric data shapes, verbose test-name modes for UI stress testing
- **Fab realism** — FYYWWSSSS lot numbers, multi-lot time sequences for trend charts, 1–600 s test time, multi-site (1–16 sites from GDPW) with layout patterns and site-to-site yield loss
- **Manual configuration tab** — every parameter above as a form control
- **AI Chat Assistant tab** — natural language input via Azure OpenAI (GPT-4.1) with keyword-parser fallback
- **Exports**
  - CSV with die-level data (`Insertion`, `Bin`, `HardBin`, `SoftBin`, `Site`, coordinates, bin metadata)
  - Optional long-format per-test CSV
  - PNG/SVG/JPEG/TIFF wafer map grid and per-wafer ZIP
  - STDF v4 binary per lot per insertion (FAR, MIR, SDR, WIR/WRR, PIR/PTR/PRR, TSR, HBR/SBR, MRR)

## Project structure

| File | Role |
|------|------|
| `app.py` | Streamlit web UI (chat + manual form) |
| `generator.py` | The pipeline: signature → yield → CP cascade → bins → exports |
| `geometry.py` | Die grid, spec limits, auto flat/notch, auto stepping field |
| `signatures.py` | Spatial defect patterns and internal bin definitions |
| `yield_model.py` | Direct/defect-density yield targets, CP retest cascade, S2S |
| `binning.py` | Internal bin → hardbin/softbin mapping |
| `test_items.py` | Test names, parametric/pass-fail values, data shapes |
| `fab.py` | Fab lot numbers, lot schedules, multi-site, test-time math |
| `renderer.py` | Draws wafer map images |
| `llm_agent.py` | Parses chat prompts into generation parameters |
| `stdf_writer.py` | Builds STDF binary output |

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

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with **GitHub** (`SaiRohitaBhaskaruni01`).
2. Click **Create app**.
3. Select repository: `SaiRohitaBhaskaruni01/WaferMapGenerator`
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
