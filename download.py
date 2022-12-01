import wandb
import hydra
import pandas as pd

# 🤔 This dumps your entire metrics into one csv file (in just four lines)
@hydra.main(version_base=None, config_name="config", config_path="./")
def download(cfg):
    print("Downloading. This might take a while... ☕️")
    api = wandb.Api(timeout=30)

    # We reuse your wandb settings from the main.py here
    runs = api.runs(f"{cfg.wandb.entity}/{cfg.wandb.project}")
    data = [{**run.config, **row} for run in runs for row in run.scan_history()]
    return pd.DataFrame(data).to_csv("results/runs.csv")


if __name__ == "__main__":
    download()
