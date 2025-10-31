import json


class WandbLog:
    def __init__(self, run):
        self.run = run

    def log_line(self, entry: dict):
        kind = entry["log"]
        del entry["log"]
        step = entry["step"]
        del entry["step"]
        del entry["time"]  # TODO can we associate a datetime with step?
        if kind == "step":
            tps = entry["step_tokens"] / (entry["duration_ms"] / 1000)
            del entry["step_tokens"]
            self.run.log({f"train/{k}": v for k, v in entry.items()}, step=step)
            self.run.log({"train/tokens_per_second": tps}, step=step)
        elif kind == "eval":
            tps = entry["eval_tokens"] / (entry["duration_ms"] / 1000)
            del entry["eval_tokens"]
            self.run.log({f"eval/{k}": v for k, v in entry.items()}, step=step)
            self.run.log({"eval/tokens_per_second": tps}, step=step)
        elif kind == "gpu":
            del entry["throttle"]  # can't log this nicely?
            del entry["id"]        # not useful?
            if entry["fan"] == 0:  # indicates not recorded
                del entry["fan"]
            entry["dram_free"] /= 1024**2   # MiB
            entry["pcie_rx"] /= 1024**2     # MiB/s
            entry["pcie_tx"] /= 1024**2     # MiB/s
            self.run.log({f"gpu/{k}": v for k, v in entry.items()}, step=step)
        elif kind == "cmd":
            # TODO figure out if we can actually put this in the _wandb config object
            # where is belongs
            self.run.config["cmd"] = entry["cmd"]
        elif kind == "gpu-model":
            if entry["rank"] == 0:
                self.run.config["gpu"] = entry
            else:
                self.run.config[f"gpu-{entry['rank']}"] = entry
        elif kind == "allocator":
            import plotly.express as px
            names = [alloc["name"] for alloc in entry["stats"]]
            amounts = [round(alloc["amount"] / 1024 / 1024, 1) for alloc in entry["stats"]]

            fig = px.pie(
                names=names,
                values=amounts,
                title=f"GPU Allocations",
            )
            self.run.log({"allocations": fig}, step=step)
        elif kind == "dataset":
            pass
            #run.config["dataset"] = entry
        elif kind == "option":
            pass
        elif kind == "checkpoint":
            pass
        else:
            raise RuntimeError(f"Unknown kind {kind}")

    def make_callback(self):
        def callback(entry: str):
            entry = json.loads(entry)
            self.log_line(entry)
        return callback


def setup_wandb(project_name=None, config=None, **kwargs):
    import wandb

    # Initialize a wandb run with optional parameters
    run = wandb.init(
        project=project_name or "LLMQ",
        config=config or {},
        **kwargs
    )

    # Create and return WandbLog instance with the run
    return WandbLog(run)
