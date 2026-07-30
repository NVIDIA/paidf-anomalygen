# Cosmos AnomalyGen — Use-Case Tutorials (UC1 / UC2 / UC3)

Step-by-step, run-from-scratch tutorials for the three reference use cases. Each use
case is self-contained in its own folder and follows the same seven-notebook flow.

> **Prerequisite — build the `cosmos-predict2` environment first.** These tutorials
> assume the `cosmos-predict2` conda environment already exists (PyTorch + CUDA 12.8,
> flash-attn, Transformer Engine, Apex, …). If it does not, run the top-level
> [`../0-setup-cuda128.ipynb`](../0-setup-cuda128.ipynb) once to create it, then start
> with `UC<n>/0-setup.ipynb`.

| Use case | Subject | Anomaly types | Mask placement | Dataset |
|---|---|---|---|---|
| [UC1](./UC1/) | PCB (printed circuit board) | `IC+bridge`, `passive_component+excess_solder`, `passive_component+missing` | `cad` (CAD-mask ROI) | Auto-download (Hugging Face) |
| [UC2](./UC2/) | Metal surface (Magnetic Tile) | `metal_surface+MT_{Blowhole,Break,Crack,Fray,Uneven}` | `free` (whole-image ROI) | Auto-download (public GitHub) |
| [UC3](./UC3/) | Mobile phone screen (glass) | `Phone+{oil,scratch,stain}` | `text` (Qwen-VL + SAM2 ROI) | Masks auto (HF) + **manual Roboflow images** |

## The seven-notebook flow (same in every use case)

| # | Notebook | What it does |
|---|---|---|
| 0 | `0-setup` | Environment check, Hugging Face auth, download base + finetuned checkpoints |
| 1 | `1-dataset-preparation` | Fetch & organize the dataset into the expected layout |
| 2 | `2-training` | Fine-tune the AnomalyGen adapters (or reuse the released checkpoint) |
| 3 | `3-auto-mask-placement` | Place defect masks on clean images → build the generation testcase |
| 4 | `4-generation` | Generate synthetic anomaly images, then evaluate & (optionally) filter |
| 5 | `5-pseudo-labeling` | Produce COCO annotations, class folders, and (optionally) captions |
| 6 | `6-agentic-flow` | Run steps 2→5 from a single Claude Code prompt (with per-sample quality search) |

Start with **`UC<n>/0-setup.ipynb`** and run the notebooks in order. All commands run in
the `cosmos-predict2` conda environment (via `conda run -n cosmos-predict2 …` in cells,
or `conda activate cosmos-predict2` in a terminal).

> **UC3 note.** The phone-screen anomaly/clean images require a one-time manual Roboflow
> download (see `UC3/1-dataset-preparation.ipynb`); masks and `defect_spec.jsonl` come
> from Hugging Face automatically.
