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

## Docker base image strategy

`docker/Dockerfile` is intentionally lightweight (~15 min): it only installs
vllm and copies source.  The compile-heavy dependencies (flash-attn,
transformer-engine, apex, plus all of `requirements-conda-cuda128.txt`) live
in `docker/Dockerfile.base`, built once and stored in the registry as
`$PERF_IMAGE:$BASE_IMAGE_TAG`.

**Tag strategy: content hash (no manual bump).**

The full base tag is `base-<12-char sha256 of Dockerfile.base + requirements-conda-cuda128.txt>`.
Both CI jobs compute the hash with the same shell command, so they always
agree on which base tag corresponds to a given commit:

```
INPUT_HASH=$(cat docker/Dockerfile.base requirements-conda-cuda128.txt | sha256sum | cut -c1-8)
BASE_IMAGE_TAG="${BASE_IMAGE_TAG_PREFIX}-${INPUT_HASH}"
```

Any change to either input automatically produces a new tag → CI builds and
pushes a new base image → `docker-build-push` pulls it.  No manual tag bump
is ever needed.

**Runner split (since this repo's A100 + horde topology is the constraint):**

| Job | Stage | Runner | Why |
|---|---|---|---|
| `build-base-image` | `build-base` | `10.63.147.87-A100` | Compile-heavy: `MAX_JOBS=32 NVCC_THREADS=8` for flash-attn / TE / apex needs the dedicated A100 host's cores + RAM |
| `docker-build-push` | `build` | `[horde, docker]` (DinD) | Lightweight: `FROM` pre-baked base + install vllm + COPY source.  Frees the A100 host for actual training/inference work |

**When a developer needs to (re)build the base image:**

| Trigger | When |
|---|---|
| First-time setup | The computed hash doesn't exist in the registry yet |
| After editing `Dockerfile.base` or `requirements-conda-cuda128.txt` | Hash changes → auto-trigger on protected branches |
| Weekly schedule | Automatic — picks up NGC base image security patches |

**How to trigger:**

- **Manually (MRs):** In the CI pipeline, find the `build-base-image` job in
  the `build-base` stage and click the play button.
- **Automatically (protected branches):** Push a commit that modifies
  `docker/Dockerfile.base` or `requirements-conda-cuda128.txt`.

**Guard rails already in place:**

- `build-base-image` skips the build if `$PERF_IMAGE:$BASE_IMAGE_TAG` already
  exists in the registry (`docker manifest inspect` guard — no duplicate work).
- `docker-build-push` does the same `docker manifest inspect` check up front
  and exits with a clear, actionable error message (not a cryptic Docker
  "FROM image not found") if the base tag is missing.
- `tests/test_docker_base_image_strategy.py` enforces all of the above
  invariants statically — no Docker daemon needed to run.

**When helping a user with Docker build issues, check this first:**

```bash
INPUT_HASH=$(cat docker/Dockerfile.base requirements-conda-cuda128.txt | sha256sum | cut -c1-8)
docker manifest inspect $PERF_IMAGE:base-$INPUT_HASH
```

If the tag is absent, direct them to one of the two trigger options above.
Don't suggest rebuilding `docker/Dockerfile.base` locally — that takes hours
on a CPU-only machine.  Use CI.

## Engineering docs

- `.agents/skills/anomalygen/SKILL.md` (orchestrator) +
  `.agents/skills/anomalygen/references/{setup,finetune,prep-testcase,sdg-inference,inference,eval,sdg-refine}.md`
  for per-phase detail.

When updating any of these, keep claims consistent across `SKILL.md` and the
per-phase `references/*.md` — past dry-run audits caught drift points (path
case, `--seeds` flag, `clean_dir` plumbing, mode-validation gates) that would
have caused real stops.
