"""Hyperparameter grid: every combination is one `cifar-train` run in its own container.

Grid config (YAML):

    base: configs/simple_cnn.yaml          # run config every combination starts from
    run_name: "cnn-lr{train.lr}-bs{train.batch_size}"   # {key} -> value of the combination
    fixed:                                  # optional: overrides shared by all runs
      train.epochs: 15
    grid:                                   # keys -> values; all combinations are run
      train.lr: [0.0003, 0.001, 0.003]
      train.batch_size: [64, 128]

By default each run is `docker compose run --rm trainer cifar-train ...` (one container per
run); `--local` runs `cifar-train` in the current environment instead.
"""

import argparse
import itertools
import json
import logging
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from hw_product_management_2026.data.common import setup_logging
from hw_product_management_2026.data.upload import git_commit
from hw_product_management_2026.training.config import load_config

log = logging.getLogger(__name__)

PLACEHOLDER = re.compile(r"\{([A-Za-z0-9_.]+)\}")


def format_value(value: Any) -> str:
    """Value as it appears in run names: [32, 64] -> 32-64, True -> true."""
    if isinstance(value, list | tuple):
        return "-".join(format_value(v) for v in value)
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def render_run_name(template: str, values: Mapping[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise SystemExit(f"run_name placeholder {{{key}}} is not in the grid or fixed")
        return format_value(values[key])

    return PLACEHOLDER.sub(replace, template)


def to_override(key: str, value: Any) -> str:
    # JSON is valid YAML, so OmegaConf parses lists, numbers and bools back unchanged.
    return f"{key}={json.dumps(value)}"


def build_runs(grid_cfg: Mapping[str, Any]) -> list[tuple[str, list[str]]]:
    """All combinations of the grid -> [(run_name, overrides)]."""
    grid: Mapping[str, Sequence[Any]] = grid_cfg.get("grid") or {}
    fixed: Mapping[str, Any] = grid_cfg.get("fixed") or {}
    if not grid:
        raise SystemExit("grid config has no `grid` section")
    template = grid_cfg.get("run_name") or "-".join(f"{{{key}}}" for key in grid)
    runs = []
    for combo in itertools.product(*grid.values()):
        values = {**fixed, **dict(zip(grid, combo, strict=True))}
        name = render_run_name(template, values)
        overrides = [to_override(k, v) for k, v in values.items()]
        runs.append((name, [*overrides, to_override("run_name", name)]))
    names = [name for name, _ in runs]
    if len(set(names)) != len(names):
        raise SystemExit("run_name template gives duplicate names; add more placeholders")
    return runs


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True, help="grid config (YAML)")
    parser.add_argument("--local", action="store_true", help="run without docker")
    parser.add_argument("--dry-run", action="store_true", help="only print the commands")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)
    raw = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if not isinstance(raw, dict) or "base" not in raw:
        raise SystemExit(f"{args.config}: grid config must have `base`")
    grid_cfg = {str(k): v for k, v in raw.items()}
    base = Path(str(grid_cfg["base"])).as_posix()
    runs = build_runs(grid_cfg)
    # Validate every combination before starting anything long.
    for _, overrides in runs:
        load_config(Path(base), overrides)

    env = dict(os.environ)
    if not env.get("GIT_COMMIT"):
        info = git_commit()
        env["GIT_COMMIT"] = str(info["commit"] or "unknown")
        env["GIT_DIRTY"] = str(info["dirty"]).lower()
    if env.get("GIT_DIRTY") == "true":
        log.warning("working tree is dirty: runs will be logged with commit %s", env["GIT_COMMIT"])

    if args.local:
        prefix = [sys.executable, "-m", "hw_product_management_2026.training.train"]
    else:
        prefix = ["docker", "compose", "run", "--rm", "trainer", "cifar-train"]
    failed = []
    for i, (name, overrides) in enumerate(runs, 1):
        cmd = [*prefix, "--config", base]
        for override in overrides:
            cmd += ["--set", override]
        log.info("[%d/%d] %s", i, len(runs), name)
        if args.dry_run:
            print(" ".join(cmd))
            continue
        if subprocess.run(cmd, env=env, check=False).returncode != 0:
            log.error("run %s failed", name)
            failed.append(name)
    if failed:
        log.error("%d/%d runs failed: %s", len(failed), len(runs), ", ".join(failed))
        return 1
    log.info("all %d runs finished", len(runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
