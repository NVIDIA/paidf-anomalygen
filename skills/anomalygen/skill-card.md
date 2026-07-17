## Description: <br>
PAIDF AnomalyGen pipeline — fine-tune, generate synthetic anomaly images (SDG), evaluate quality (nn_score), and per-sample search across full, finetune_only, and inference_only modes. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers generating synthetic anomaly images for training anomaly detection models using a few-shot diffusion-based pipeline. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Setup (checkpoint download, HF_TOKEN)](references/setup.md) <br>
- [Fine-tune (Phase 0–1)](references/finetune.md) <br>
- [Inference (Phases 2–7)](references/inference.md) <br>
- [Prep-testcase (AMP routing)](references/prep-testcase.md) <br>
- [SDG inference (multi-GPU, NCCL)](references/sdg-inference.md) <br>
- [Eval (nn_score, FID)](references/eval.md) <br>
- [SDG refine (search, re-AMP)](references/sdg-refine.md) <br>
- [Datasets (UC1/UC2/UC3)](references/datasets.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Files, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Generated images, per-sample CSV metrics, eval logs] <br>

## Evaluation Agents Used: <br>
- claude-code <br>
- codex <br>



## Evaluation Tasks: <br>
Evaluated against 3 internal skill-activation tasks in the astra-sandbox environment (NVSkills-Eval external profile). <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies that the agent loaded the expected skill and workflow. <br>
- `skill_efficiency`: Checks routing quality, decoy avoidance, and redundant tool usage. <br>
- `accuracy`: Grades final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Checks whether the overall user task completed successfully. <br>
- `behavior_check`: Verifies expected behavior steps, including safety expectations. <br>
- `token_efficiency`: Compares token usage with and without the skill. <br>



## Evaluation Results: <br>
| Dimension | Num | `claude-code` | `codex` |
|---|---:|---:|---:|
| Security | 3 | 100% (+0%) | 100% (+0%) |
| Correctness | 3 | 89% (+64%) | 66% (+31%) |
| Discoverability | 3 | 90% (+55%) | 84% (+40%) |
| Effectiveness | 3 | 64% (+60%) | 21% (+8%) |
| Efficiency | 3 | 70% (+33%) | 78% (+32%) |

## Skill Version(s): <br>
1.0.0 (source: pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
