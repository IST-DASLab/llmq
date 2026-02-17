#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["wandb", "plotly[express]", "pandas"]
# ///


import argparse
import datetime
import json
import time
from pathlib import Path
from typing import Optional

import wandb


def log_line(run: "wandb.Run", entry: dict):
    kind = entry["log"]
    del entry["log"]
    step = entry["step"]
    del entry["step"]
    del entry["time"]  # TODO can we associate a datetime with step?
    if kind == "step":
        tps = entry["step_tokens"] / (entry["duration_ms"] / 1000)
        del entry["step_tokens"]
        run.log({f"train/{k}": v for k, v in entry.items()}, step=step)
        run.log({"train/tokens_per_second": tps}, step=step)
    elif kind == "eval":
        tps = entry["eval_tokens"] / (entry["duration_ms"] / 1000)
        del entry["eval_tokens"]
        run.log({f"eval/{k}": v for k, v in entry.items()}, step=step)
        run.log({"eval/tokens_per_second": tps}, step=step)
    elif kind == "gpu":
        del entry["throttle"]  # can't log this nicely?
        del entry["id"]        # not useful?
        if entry["fan"] == 0:  # indicates not recorded
            del entry["fan"]
        entry["dram_free"] /= 1024**2   # MiB
        entry["pcie_rx"] /= 1024**2     # MiB/s
        entry["pcie_tx"] /= 1024**2     # MiB/s
        run.log({f"gpu/{k}": v for k, v in entry.items()}, step=step)
    elif kind == "cmd":
        # TODO figure out if we can actually put this in the _wandb config object
        # where is belongs
        run.config["cmd"] = entry["cmd"]
    elif kind == "gpu-model":
        if entry["rank"] == 0:
            run.config["gpu"] = entry
        else:
            run.config[f"gpu-{entry['rank']}"] = entry
    elif kind == "allocator":
        import plotly.express as px
        names = [alloc["name"] for alloc in entry["stats"]]
        amounts = [round(alloc["device"] / 1024 / 1024, 1) for alloc in entry["stats"]]

        fig = px.pie(
            names=names,
            values=amounts,
            title=f"GPU Allocations",
        )
        run.log({"allocations": fig}, step=step)
    elif kind == "dataset":
        pass
        # run.config["dataset"] = entry
    elif kind in ["option", "info"]:
        pass
    elif kind == "message":
        print(entry["message"])
    elif kind == "abs-maxes":
        for stats in entry["abs_maxes"]:
            run.log({f"abs_maxes/{stats['name']}": stats['value']}, step=step)
    elif kind == "sol":
        if entry["rank"] != 0:
            return
        import plotly.express as px
        names = ["Blocks", "LM-Head", "Attention"]
        amounts = [entry["blocks"], entry["lm_head"], entry["attention"]]

        fig = px.pie(
            names=names,
            values=amounts,
            title=f"FLOPs",
        )
        run.log({"ops": fig}, step=step)
    else:
        raise RuntimeError(f"Unknown kind {kind}")

def convert_log(file_name: str, *, name: Optional[str], project: str, notes: str="", tags: list[str] = None):
    while name is None:
        log_data = json.loads(Path(file_name).read_text())
        for entry in log_data:
            if entry["log"] == "option":
                opt_name = entry["name"]
                opt_value = entry["value"]
                if opt_name == "name":
                    name = opt_value
        time.sleep(10)

    entry_count = 0
    running = True
    with wandb.init(
            project=project,
            name=name,
            notes=notes,
            tags=tags,
    ) as run:
        while running:
            content = Path(file_name).read_text()
            log_data = json.loads(content)
            for i, entry in enumerate(log_data):
                if i < entry_count:
                    continue
                if entry["log"] == "info" and entry["type"] == 'message' and entry["message"].startswith("Done. validation loss"):
                    running = False
                log_line(run, entry)
                entry_count += 1
            time.sleep(10)

def main():
    parser = argparse.ArgumentParser(description="Plot training run")
    parser.add_argument("--log-file", type=str, help="Log file", default="log.json")
    parser.add_argument("--project", help="WandB project name")
    parser.add_argument("--name", help="Name for the run", default=None)
    args = parser.parse_args()
    convert_log(args.log_file, project=args.project, name=args.name)


if __name__ == "__main__":
    main()
