import argparse
import datetime
import json
from typing import Optional

import wandb

def convert_log(file_name: str, *, name: Optional[str], project: str, notes: str="", tags: list[str] = None):
    log_data = json.load(file_name)

    with wandb.init(
            project=project,
            name=name,
            notes=notes,
            tags=tags,
    ) as run:
        for entry in log_data:
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
                del entry["id"]        # not useful
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
                amounts = [round(alloc["amount"] / 1024 / 1024, 1) for alloc in entry["stats"]]

                fig = px.pie(
                    names=names,
                    values=amounts,
                    title=f"GPU Allocations",
                )
                run.log({"allocations": fig}, step=step)
            elif kind == "dataset":
                pass
                #run.config["dataset"] = entry
            elif kind == "option":
                pass
            elif kind == "checkpoint":
                pass
            else:
                raise RuntimeError(f"Unknown kind {kind}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training run")
    parser.add_argument("--log-file", type=argparse.FileType("r"), help="Log file", default="log.json")
    parser.add_argument("--project", help="WandB project name")
    parser.add_argument("--name", help="Name for the run", default=None)
    args = parser.parse_args()

    convert_log(args.log_file, project=args.project, name=args.name)
