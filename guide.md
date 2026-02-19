# Qwen-Audio Feature Extraction Pipeline

## 1. Project Structure
Ensure your working directory has the following files:
* **`config.yaml`**: The central configuration file for folders, batch sizes, and prompts.
* **`qwen_parallel_with_config.py`**: The main extraction script.
* **`loader.py`**: Handles directory mapping and audio loading.
* **`submit_job.slurm`**: The SLURM submission script.

---

## 2. Configuration (`config.yaml`)
Before running, adjust your `config.yaml` to specify your target dataset, save preferences, and text prompts.

```yaml
run_type: "syntax"
audio_folder: "BLIMP_KOKORO"
output_base_dir: "output_features"

# folder_mode:
# "CREATE" preserves the relative subdirectory structure from your audio_folder
# "LET" dumps all .npy files flat into a single directory
folder_mode: "CREATE"

batch_size: 4
max_samples: 10000

# Prompts are executed sequentially for every audio batch loaded into memory
prompts:
  - eval_type: "01"
    text: ""
  - eval_type: "02"
    text: "Listen to the following sentence. Is it grammatically correct or ungrammatical? Answer with 'Correct' or 'Incorrect'."
```

## 3. SLURM Submission Script (submit_job.slurm)

Run the following:
```sbatch submit_job.slurm```

Monitoring Your Job:

* Check Queue Status: ```squeue -u $USER```
* Cancel a Job: ```scancel <JOB_ID>```
* View Real-Time Logs: ```check your Python logger file```