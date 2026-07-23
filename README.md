# gpt-oss_scoring

Score UPD-style unscored model-response JSON files with local **gpt-oss-120b** via **vLLM**, submitted as Slurm jobs (no always-on model server).

## Setup

```bash
# Conda env
source /project/3dllms/melgin/conda/etc/profile.d/conda.sh
conda env create -f environment.yml   # creates gpt-oss-scoring
# or: conda activate gpt-oss-scoring  # if already created

# Important: keep transformers==4.56.1 (vLLM 0.10.1.1 breaks on transformers 5.x).
# If the env already exists and scoring fails on all_special_tokens_extended:
#   conda activate gpt-oss-scoring && pip install 'transformers==4.56.1'

# Download weights (~61G MXFP4) into ./models/gpt-oss-120b
chmod +x scripts/download_gpt_oss_120b.sh
./scripts/download_gpt_oss_120b.sh
```

## Score one file

```bash
chmod +x slurm_score_model_responses.sh multi-slurm_score_model_responses.sh

./slurm_score_model_responses.sh --partition h100 \
  --answer_key /project/3dllms/melgin/UPD-3D/answer_keys/3D-FRONT.json \
  /path/to/unscored_model_responses/some_run/file.json
```

`--partition` accepts `a30`, `l40s`, `h100`, or full names `mb-a30`, `mb-l40s`, `mb-h100`.

GPU defaults:

| Alias | Partition | GPUs | Mem  |
|-------|-----------|------|------|
| h100  | mb-h100   | 1    | 128G |
| l40s  | mb-l40s   | 2    | 192G |
| a30   | mb-a30    | 4    | 256G |

Optional `--output <dir>` overrides the default output location.

**Default output:** from the input JSON path,  
`../../scored_model_responses/<input_folder>_oss120/<basename>_oss120_scored.json`

## Score every JSON in a folder (one job per file)

```bash
./multi-slurm_score_model_responses.sh /path/to/unscored_model_responses/some_run \
  --partition l40s \
  --answer_key /project/3dllms/melgin/UPD-3D/answer_keys/3D-FRONT.json
```

## Behavior notes

- Reasoning effort defaults to **high**; only the final answer (`T` or `F`) is kept as the score.
- Decoding is greedy (`temperature=0`).
- Empty model responses are scored `F` without a model call.
- `--answer_key` is required. It is applied for filenames ending in `standard.json` or containing `_aad_`, `_iasd_`, or `_oe_solvable`. For `_aad_` / `_iasd_` (base, additional_option, additional_instruction), the graded correct answer is `There is no correct answer / none of the above, or <answer key>`. Other files use “unanswerable / none of the above”.
- Invalid non-`T`/`F` outputs fail the job and write `FAILED_*_oss120_scored.json`. Retries are not used under greedy decoding (`temperature=0`), since they would repeat the same tokens.
- vLLM memory knobs auto-tune by GPU size (A30 is tight for 120B): lower `max_num_seqs` / `max_model_len`, and `enforce_eager` on ≤24GB GPUs. Override with `--max-num-seqs`, `--max-model-len`, `--gpu-memory-utilization`, `--enforce-eager` / `--no-enforce-eager`.
- Logs: `./slurm_logs/score_oss120_<jobid>.out`
