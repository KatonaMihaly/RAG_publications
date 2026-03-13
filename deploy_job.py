"""
Deploy (or update) the nightly RAG Workflow from job.yml.

All configuration — parameters, schedule, task DAG, pip packages — lives in
job.yml.  This script is a thin deployer: it reads that file, uploads the
pipeline notebooks to the Databricks workspace, and creates or updates the job.

Job-level parameters defined in job.yml are automatically pushed down to every
task by Databricks via dbutils.widgets.  Each pipeline script reads them with
a _param() helper that calls dbutils.widgets.get() on Databricks and falls back
to os.environ for local runs.

Usage
-----
    .venv/bin/python3 deploy_job.py

Prerequisites
-------------
  - ~/.databrickscfg with host/token configured.
  - job.yml present in this directory.
"""

import base64
import sys
from pathlib import Path

import yaml
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import (
    JobSettings,
    JobParameterDefinition,
    Task,
    NotebookTask,
    TaskDependency,
    CronSchedule,
    PauseStatus,
    Source,
)
from databricks.sdk.service.workspace import ImportFormat, Language

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

JOB_YML      = Path(__file__).parent / "job.yml"
PIPELINE_DIR = Path(__file__).parent / "pipeline"
PROMPTS_DIR  = PIPELINE_DIR / "prompts"

WORKSPACE_NOTEBOOK_DIR = "/Users/{username}/rag_publications"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    with JOB_YML.open() as f:
        return yaml.safe_load(f)["job"]


def _extra_params(task_key: str) -> dict:
    """
    Task-specific parameters that cannot be stored in job.yml because they
    contain multi-line content (prompt files).  Only evaluate_rag needs these.
    """
    if task_key == "evaluate_rag":
        return {
            "SYSTEM_PROMPT":     (PROMPTS_DIR / "research_assistant_system_prompt.md").read_text().strip(),
            "FEW_SHOT_EXAMPLES": (PROMPTS_DIR / "few_shot_examples.json").read_text().strip(),
        }
    return {}


def _notebook_content(script_path: Path, pip_packages: list[str]) -> str:
    """
    Prepend a %pip install magic cell when the task has extra packages.

    Scripts use _param() / dbutils.widgets.get() directly — no injection needed.
    """
    content = script_path.read_text()
    if not pip_packages:
        return content
    pip_cell = "# MAGIC %pip install " + " ".join(pip_packages) + "\n# COMMAND ----------\n\n"
    return pip_cell + content


def _upload_notebooks(w: WorkspaceClient, notebook_dir: str,
                      tasks: list[dict]) -> dict[str, str]:
    w.workspace.mkdirs(path=notebook_dir)
    paths = {}
    for task in tasks:
        task_key      = task["task_key"]
        local_path    = PIPELINE_DIR / task["script"]
        pip_packages  = task.get("pip", [])
        notebook_path = f"{notebook_dir}/{task_key}"
        encoded = base64.standard_b64encode(
            _notebook_content(local_path, pip_packages).encode()
        ).decode()
        w.workspace.import_(
            path=notebook_path,
            format=ImportFormat.SOURCE,
            language=Language.PYTHON,
            content=encoded,
            overwrite=True,
        )
        print(f"  Uploaded {task['script']} → {notebook_path}")
        paths[task_key] = notebook_path
    return paths


def _build_job_settings(cfg: dict, notebook_paths: dict[str, str]) -> JobSettings:
    # Job-level parameters — Databricks pushes these to every task automatically
    job_params = [
        JobParameterDefinition(name=k, default=str(v))
        for k, v in cfg["parameters"].items()
    ]
    # Prompt content is too long for job.yml; pass as per-task base_parameters
    # only for evaluate_rag (the only task that needs them at widget level).
    def _base_params(task_key: str) -> dict | None:
        extra = _extra_params(task_key)
        return extra if extra else None

    tasks = [
        Task(
            task_key=task["task_key"],
            notebook_task=NotebookTask(
                notebook_path=notebook_paths[task["task_key"]],
                source=Source.WORKSPACE,
                base_parameters=_base_params(task["task_key"]),
            ),
            depends_on=[TaskDependency(task_key=k) for k in task["depends_on"]] or None,
            timeout_seconds=cfg["timeout_seconds"],
        )
        for task in cfg["tasks"]
    ]

    return JobSettings(
        name=cfg["name"],
        parameters=job_params,
        tasks=tasks,
        schedule=CronSchedule(
            quartz_cron_expression=cfg["schedule"],
            timezone_id=cfg["timezone"],
            pause_status=PauseStatus.UNPAUSED,
        ),
        timeout_seconds=cfg["timeout_seconds"],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def deploy() -> None:
    if not JOB_YML.exists():
        print("ERROR: job.yml not found. Run from the project root.", file=sys.stderr)
        sys.exit(1)

    cfg = _load_config()
    print(f"Loaded job.yml — '{cfg['name']}', {len(cfg['parameters'])} parameter(s), "
          f"{len(cfg['tasks'])} task(s).")

    w        = WorkspaceClient()
    username = w.current_user.me().user_name
    notebook_dir = WORKSPACE_NOTEBOOK_DIR.format(username=username)

    print(f"\nUploading pipeline notebooks to {notebook_dir}/ ...")
    notebook_paths = _upload_notebooks(w, notebook_dir, cfg["tasks"])

    settings      = _build_job_settings(cfg, notebook_paths)
    existing_jobs = {j.settings.name: j.job_id for j in w.jobs.list() if j.settings}

    if cfg["name"] in existing_jobs:
        job_id = existing_jobs[cfg["name"]]
        print(f"\nJob '{cfg['name']}' (id={job_id}) exists — updating...")
        w.jobs.reset(job_id=job_id, new_settings=settings)
        print("Job updated.")
    else:
        print(f"\nCreating job '{cfg['name']}'...")
        result = w.jobs.create(
            name=settings.name,
            parameters=settings.parameters,
            tasks=settings.tasks,
            schedule=settings.schedule,
            timeout_seconds=settings.timeout_seconds,
        )
        job_id = result.job_id
        print(f"Job created with id={job_id}.")

    host = w.config.host.rstrip("/")
    print(f"\nJob URL: {host}/#job/{job_id}")
    print("Scheduled: daily at 01:00 UTC (completes within 01:00–04:00 window).")
    print(f"To trigger manually: .venv/bin/python3 -c \"from databricks.sdk import WorkspaceClient; WorkspaceClient().jobs.run_now(job_id={job_id})\"")


if __name__ == "__main__":
    deploy()
