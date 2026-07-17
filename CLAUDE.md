# Project conventions for Claude

This repo wraps an upstream codebase (PAIDF AnomalyGen) with a skill set under
`.agents/skills/`. Stay inside the skill layer for everything except confirmed
upstream bugs.

## Where you can edit freely

- `.agents/skills/<name>/` — `SKILL.md`, references, assets.
- `scripts/utilities/` — packaged helper scripts (canonical location;
  available in container and on host via `${ANOMALYGEN_SCRIPTS}`).
- `tutorial/` — user-facing docs.
- Mid-product / pipeline artifacts the user owns:
  - `ag_inference/<name>/{allocation.json,amp_samples.json,testcase.jsonl,…}`
  - `results/<name>/{rounds,searched,filtered}/draws.json,per_sample.csv,…`
  - Generated configs under `ag_configs/`.
- `datasets/<name>/defect_spec.jsonl` and similar user-supplied data.
- Ad-hoc shell / Python invoked at runtime (one-liners, scratch scripts in
  `/tmp/`, etc.).

## Where NOT to edit (unless the user explicitly approves a core fix)

- `cosmos_predict2/` — upstream library code.
- `scripts/anomaly_gen/` — upstream training/inference scripts.

These are upstream-tracked. If you find a real bug there, surface it to the
user and ask for sign-off before patching. Don't make perf or convenience
edits in these dirs — fix it skill-side instead (helper script, JSONL/config
override, mid-product edit).

## Workflow patterns the skill set already supports

These are derivable from the code but easy to miss; surface them when a user
asks for the corresponding outcome:

- **Generate only specific defects:** trim `defect_spec.jsonl` to just the
  desired defect lines and run with `mode=inference_only`. `prep-testcase`
  allocates all `num_SDG` to the listed defects; `sdg-inference/validate_jsonl.py`
  enforces that each requested defect is in the trained checkpoint's
  supported set (`<ckpt>/ag_config.yaml → dataloader_train.dataset.anomaly_types`).
- **Custom per-defect counts (non-proportional):** run `prep_testcase.sh`,
  then overwrite `ag_inference/<name>/allocation.json` before the AMP /
  `build_jsonl.py` steps run. Or run prep-testcase per-defect with separate
  `name=`s and `num_SDG=` values, then concatenate the testcase JSONLs.
- **List supported defects in a checkpoint:** read
  `<checkpoint_dir>/ag_config.yaml → dataloader_train.dataset.anomaly_types`
  (each entry is `[texture, anomaly]` → render as `texture+anomaly`).
- **Skip Phase 1 / reuse a checkpoint:** `mode=inference_only` requires
  `checkpoint_dir` and `step`. Path follows
  `results/anomaly_gen/<name>/<name>_training_FP32_lr<lr>_bs=<bs>_<MODEL_SIZE>_<H>x<W>`
  with `MODEL_SIZE` in upper-case (`2b` → `2B`, `14b` → `14B`).

## Guiding users when their request doesn't fit cleanly

When a user's request doesn't map onto the documented `mode=full` /
`inference_only` / `finetune_only` paths, **don't just run it and let the
pipeline halt with an error.** Pre-validate against on-disk state, then
either redirect to a supported workflow or call out limits explicitly.

The pattern, in order:

1. **Inspect the relevant on-disk state first** — `<checkpoint>/ag_config.yaml`,
   `<dataset>/<TEXTURE>/{anomaly_image,mask}/<TYPE>/`, the user's
   `defect_spec.jsonl`, prior `results/<name>/` dirs. A single `cat` / `ls`
   often surfaces the mismatch before any GPU time is burned.
2. **Compare against the user's request** — supported defects vs. requested,
   trained `image_size` vs. requested, `step` value vs. saved checkpoints,
   defect mask folders present vs. listed in `defect_spec`, etc.
3. **If there's a mismatch, present workflow options instead of running** —
   re-run as `mode=full` with extended data, run as a separate isolated
   model, trim the spec, or pick a different `step`. Be specific about
   trade-offs (time, quality, whether old defects are preserved).
4. **If no skill-side option exists, say so plainly** and stop — don't
   silently fudge it or invent a workaround. The user's interface is the
   skill set; out-of-scope requests get an honest "not supported, here's
   the closest thing the skill can do."

Common mismatches to watch for and the right redirect:

| User request / input | Where it would fail | Surface this instead |
|---|---|---|
| `defect_spec` lists a defect not in the checkpoint's `ag_config.yaml` | `validate_jsonl.py` halt | Three options: `mode=full` retrain on extended set; isolated new model with separate `name=`; trim spec to supported defects |
| `dataset_dir/<TEXTURE>/mask/<TYPE>/` empty for a listed defect | `allocate_samples.py` "mask count = 0" | Ask for masks or trim the spec |
| `mode=inference_only` with a `step` not on a `save_iter` boundary | Cryptic `torch.load` FileNotFoundError | `ls <ckpt>/checkpoints/model/iter_*.pt`, present the actual saved steps, ask user to pick |
| Different `image_size` or `model_size` than what was trained | DiT shape mismatch / checkpoint key mismatch | Read trained config, surface the mismatch, offer to run at the trained values |
| Per-defect counts that aren't proportional to training mask counts | Default proportional allocation | Either per-defect runs concatenated, or hand-edit `ag_inference/<name>/allocation.json` between `allocate_samples.py` and `build_amp_samples.py` |
| Add one new defect to an existing checkpoint without retraining everything | Not skill-supported | Be honest: the skill doesn't do warm-start fine-tune. Offer full retrain on the extended defect set, or an isolated new model just for the new defect. |

Default to **inspecting first, asking second, running third** — especially
for `mode=inference_only` (the user has a specific intent there, and the
pipeline has the most ways to silently misalign with it).

## Long-running execution guidance

The full pipeline (`mode=full`) is multi-hour. A single subagent invocation
typically can't supervise a 4+ hour run end-to-end — agents tend to pause
themselves once a long background job is launched. For multi-hour
orchestration:

- Drive the loop from the parent conversation rather than a single subagent.
- Fire long jobs with `Bash --run_in_background` and poll via `BashOutput` /
  `Monitor` / `ScheduleWakeup`.
- If you do dispatch subagents, keep their scope narrow (one phase at a
  time) and expect to chain them.

## Docker build strategy

`docker/Dockerfile` is a **single-stage** build. One Dockerfile bundles the
compile-heavy dependencies (flash-attn, transformer-engine, apex,
opencv-python-headless, vllm), the full `requirements-conda-cuda128.txt`, and
the application layer. The base image is `nvidia/cuda:12.8.2-devel-ubuntu24.04`
(`ARG PYTORCH_BASE`). There is **no** `docker/Dockerfile.base`, no separate
base-image job, and no content-hash base tag — if you see a reference to any of
those, it is stale.

**Where the heavy wheels come from.**

The CUDA-extension wheels (flash-attn, transformer-engine, apex,
opencv-python-headless) are pre-built and published to this project's **GitLab
PyPI registry**. CI passes a token-bearing index URL into the build via
`--build-arg GITLAB_PYPI_INDEX_URL=...` (see `.gitlab-ci.yml`). When that arg is
unset (local builds without a token), each compile-heavy `RUN` **falls back to a
source compile** — apex from `git+https://github.com/NVIDIA/apex.git`, opencv
from source with `WITH_FFMPEG=OFF` (never the vulnerable PyPI wheel). A local
`docker build` therefore still works without registry access; it is just slow.

**CI job (single stage):**

| Job | Stage | Runner | What it does |
|---|---|---|---|
| `docker-build-push` | `build` | `[horde, docker]` (DinD, `docker:28.4.0-cli`, 4 h timeout) | `docker build -f docker/Dockerfile` with the GitLab PyPI index URL, then push to both `$PERF_IMAGE` and `$SDG_IMAGE` |

**Tag strategy: `VERSION-SHA.LABEL` (no manual bump, no content hash).**

The image tag is computed in the job script:

```
IMAGE_TAG="$VERSION-$CI_COMMIT_SHORT_SHA.$LABEL"
```

- `VERSION` — parsed from `pyproject.toml` (`version = ...`).
- `CI_COMMIT_SHORT_SHA` — the commit.
- `LABEL` — `mr<IID>` on merge-request pipelines, else `main`.

Layer caching uses a floating `:cache` tag: the job pulls `$PERF_IMAGE:cache`,
builds with `--cache-from $PERF_IMAGE:cache --build-arg BUILDKIT_INLINE_CACHE=1`,
and re-pushes `:cache`. Rebuilds are therefore incremental — only layers whose
inputs changed recompile (the heavy compile layers stay cached unless their pins
move).

**When / how the image is (re)built:**

| Trigger | When |
|---|---|
| MR pipeline | `docker-build-push` is **manual** (`when: manual`, `allow_failure`) — click the play button in the `build` stage |
| Protected branch | Runs automatically on `anomalyGen` and `main` |

**Local build:** `docker build -f docker/Dockerfile -t paidf-anomalygen:cuda12.8 .`
from the repo root — add `--build-arg GITLAB_PYPI_INDEX_URL=<token-url>` to pull
the pre-built wheels, or omit it for the (slower) source-compile fallback. This
is the same image CI builds.

**When helping a user with Docker build issues:**

- A failing heavy-dep install (flash-attn / TE / apex / opencv) almost always
  means `GITLAB_PYPI_INDEX_URL` is unset or its token expired, so the build fell
  through to a source compile. That is expected for local builds (slow but
  correct); in CI it points at a bad/expired `CI_JOB_TOKEN` or a registry outage.
- `FROM` / "image not found" → the NGC base `nvidia/cuda:12.8.2-devel-ubuntu24.04`
  could not be pulled (login / network), not a missing project base image.
- Don't look for `docker/Dockerfile.base`, a `build-base` stage, or an A100
  base-image job — none exist. The single `docker-build-push` job is the whole
  build.

## Engineering docs

- `.agents/skills/anomalygen/SKILL.md` (orchestrator) +
  `.agents/skills/anomalygen/references/{setup,finetune,prep-testcase,sdg-inference,inference,eval,sdg-refine}.md`
  for per-phase detail.

When updating any of these, keep claims consistent across `SKILL.md` and the
per-phase `references/*.md` — past dry-run audits caught drift points (path
case, `--seeds` flag, `clean_dir` plumbing, mode-validation gates) that would
have caused real stops.
