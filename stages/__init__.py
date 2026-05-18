"""
Per-stage pipeline functions.

Each module here exposes one entry point with a uniform shape:

    def run_<stage>(pdf: Path, paths: StagePaths, **stage_specific_kwargs) -> None

Stages communicate exclusively through artifacts on disk (paths come from
`stages.paths.StagePaths`), so any stage can be invoked independently as long
as its inputs already exist. The CLI (`cli.py`) wires these into Typer
subcommands; `run_v3.py` invokes the same functions in-process as a thin
back-compat shim.
"""
