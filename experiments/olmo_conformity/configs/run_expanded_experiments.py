#!/usr/bin/env python3
"""
Automated Pipeline for Running Expanded Conformity Experiments

This script orchestrates the complete experiment workflow:
1. Runs all temperature variants (0.0 to 1.0 in 0.2 increments)
2. Captures and saves run metadata (run_id, run_dir, temperature)
3. Generates per-run analysis reports
4. Generates combined cross-temperature comparison analysis

Output Structure:
- Individual runs saved to: {runs_dir}/{timestamp}_{run_id}/
- Combined analysis saved to: {repo_root}/Comparing_Experiments/

Path Configuration:
- By default (local mode): uses {repo_root}/models and {repo_root}/runs
- With --hpc flag: uses paths from experiments/olmo_conformity/configs/paths.json

Usage:
    # Full pipeline (run experiments + all analysis) - LOCAL mode
    python run_expanded_experiments.py
    
    # Keep Mac awake during experiments (recommended for long runs)
    python run_expanded_experiments.py --no-sleep
    
    # HPC mode - uses paths from paths.json
    python run_expanded_experiments.py --hpc
    
    # With custom runs directory (overrides both local and HPC defaults)
    python run_expanded_experiments.py --runs-dir /scratch/runs
    
    # Using Ollama for inference
    python run_expanded_experiments.py --api-base http://localhost:11434/v1
    
    # Skip running, only generate analysis from existing runs
    python run_expanded_experiments.py --skip-runs

    # Run experiments only (useful for splitting temperatures across HPC jobs)
    python run_expanded_experiments.py --runs-only --temps 0.0
    
    # Only regenerate combined analysis (requires runs_metadata.json)
    python run_expanded_experiments.py --only-analysis
    
    # Dry run (show what would be done without executing)
    python run_expanded_experiments.py --dry-run
    
Sleep Prevention (macOS):
    The --no-sleep flag uses macOS's built-in 'caffeinate' utility to prevent
    the system from sleeping while experiments run. This allows you to close
    the laptop lid (if on AC power) or let the screen turn off without
    interrupting the experiment pipeline.
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Global reference to caffeinate process for cleanup
_caffeinate_process: Optional[subprocess.Popen] = None


def get_memory_usage_mb() -> float:
    """
    Get current process memory usage in MB.
    
    Uses psutil if available, falls back to resource module on Unix.
    Returns 0.0 if unable to determine memory usage.
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    
    try:
        import resource
        # getrusage returns memory in KB on Linux, bytes on macOS
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            return usage / (1024 * 1024)  # macOS: bytes to MB
        else:
            return usage / 1024  # Linux: KB to MB
    except (ImportError, AttributeError):
        pass
    
    return 0.0


def log_memory_usage(logger: logging.Logger, context: str = "") -> None:
    """Log current memory usage with optional context."""
    mem_mb = get_memory_usage_mb()
    if mem_mb > 0:
        logger.info(f"Memory usage{' (' + context + ')' if context else ''}: {mem_mb:.1f} MB")


def start_caffeinate(logger: Optional[logging.Logger] = None) -> bool:
    """
    Start caffeinate to prevent macOS from sleeping.
    
    Uses caffeinate with flags:
    - -s: Prevent system sleep (keeps CPU running)
    - -i: Prevent idle sleep
    - -w: Wait for process with given PID (our script's PID)
    
    This ensures the Mac stays awake even with the lid closed (if on AC power)
    or screen off while the experiments run.
    
    Returns:
        True if caffeinate was started successfully, False otherwise
    """
    global _caffeinate_process
    log = logger or logging.getLogger("expanded_experiments")
    
    # Only works on macOS
    if platform.system() != "Darwin":
        log.info("Not on macOS, skipping caffeinate (sleep prevention)")
        return False
    
    # Check if caffeinate is available
    try:
        subprocess.run(["which", "caffeinate"], capture_output=True, check=True)
    except subprocess.CalledProcessError:
        log.warning("caffeinate not found, system may sleep during experiments")
        return False
    
    try:
        # Start caffeinate that waits for our process
        # -s: prevent system sleep
        # -i: prevent idle sleep  
        # -w: wait for specified PID to finish
        current_pid = os.getpid()
        _caffeinate_process = subprocess.Popen(
            ["caffeinate", "-s", "-i", "-w", str(current_pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.info(f"Started caffeinate (PID {_caffeinate_process.pid}) to prevent sleep during experiments")
        log.info("  Your Mac will stay awake even with screen off or lid closed (on AC power)")
        
        # Register cleanup handler
        atexit.register(stop_caffeinate, logger)
        
        return True
    except Exception as e:
        log.warning(f"Failed to start caffeinate: {e}")
        return False


def stop_caffeinate(logger: Optional[logging.Logger] = None) -> None:
    """Stop the caffeinate process if running."""
    global _caffeinate_process
    log = logger or logging.getLogger("expanded_experiments")
    
    if _caffeinate_process is not None:
        try:
            _caffeinate_process.terminate()
            _caffeinate_process.wait(timeout=5)
            log.info("Stopped caffeinate - system can sleep normally again")
        except Exception as e:
            log.debug(f"Error stopping caffeinate: {e}")
        finally:
            _caffeinate_process = None

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]  # experiments/olmo_conformity/configs -> repo root
CONFIGS_DIR = SCRIPT_DIR
PATHS_CONFIG_FILE = CONFIGS_DIR / "paths.json"
RUNS_SUMMARY_DIR = CONFIGS_DIR / "runs_summary"

# Default paths (local mode - repo root)
DEFAULT_MODELS_DIR = REPO_ROOT / "models"
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"
COMPARING_EXPERIMENTS_DIR = REPO_ROOT / "Comparing_Experiments"

# Temperature configurations
TEMPERATURES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CONFIG_TEMPLATE = "suite_expanded_temp{temp}.json"
DEFAULT_SUITE = CONFIGS_DIR / "test_scripts" / "suite_expanded_localtest_by_model.json"


def load_suite(suite_path: Path) -> Dict[str, Any]:
    """Load suite config. Suite defines models with temperatures; execution is model-first."""
    if not suite_path.exists():
        raise FileNotFoundError(f"Suite not found: {suite_path}")
    with open(suite_path, "r") as f:
        return json.load(f)


def build_run_config(suite: Dict[str, Any], variant: str, model_id: str, temperature: float) -> Dict[str, Any]:
    """Build a single-run config from suite: one model, one temperature."""
    import copy
    config = copy.deepcopy(suite)
    # Preserve full model spec (max_new_tokens, has_think_tokens, etc.)
    full_spec = next(
        (m for m in suite.get("models", []) if m.get("variant") == variant and m.get("model_id") == model_id),
        {"variant": variant, "model_id": model_id},
    )
    config["models"] = [copy.deepcopy(full_spec)]
    config.setdefault("run", {})["temperature"] = temperature
    return config


def load_hpc_paths() -> Dict[str, str]:
    """Load paths from paths.json for HPC mode."""
    if PATHS_CONFIG_FILE.exists():
        with open(PATHS_CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}

# Logging setup
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    temperature: float
    run_id: str
    run_dir: str
    db_path: str
    config_file: str
    status: str  # "completed", "failed", "skipped"
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    model_variant: Optional[str] = None  # For model-first mode


def setup_logging(log_file: Optional[Path] = None, max_bytes: int = 10_000_000, backup_count: int = 3) -> logging.Logger:
    """
    Configure logging to console and optionally to file with rotation.
    
    Args:
        log_file: Path to log file (optional)
        max_bytes: Maximum log file size before rotation (default 10MB)
        backup_count: Number of backup files to keep (default 3)
    """
    from logging.handlers import RotatingFileHandler
    
    logger = logging.getLogger("expanded_experiments")
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers on re-initialization
    if logger.handlers:
        logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(console_handler)
    
    # File handler with rotation (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, 
            mode='a', 
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(file_handler)
    
    return logger


def load_metadata(metadata_path: Path) -> Dict[str, Any]:
    """Load existing metadata or return empty structure."""
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception:
            # If the file is corrupted/partial (e.g., interrupted write), fall back
            # to a new metadata structure rather than crashing the whole pipeline.
            pass
    return {
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "experiments": {}
    }


def save_metadata(metadata: Dict[str, Any], metadata_path: Path) -> None:
    """
    Save metadata to JSON file (atomic write).

    Atomic replace prevents partial writes if the process is interrupted.
    """
    metadata["updated_at"] = datetime.now().isoformat()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    with open(tmp_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    os.replace(tmp_path, metadata_path)


@contextmanager
def _metadata_lock(metadata_path: Path, *, timeout_s: float = 600.0, poll_s: float = 0.25):
    """
    Cross-process lock for metadata updates.

    We lock a sibling lockfile so that multiple HPC jobs can safely append/update
    `runs_metadata.json` without clobbering each other's updates.
    """
    lock_path = metadata_path.with_suffix(metadata_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # Use fcntl when available (auto-released on process exit).
    try:
        import fcntl  # type: ignore
    except Exception:  # pragma: no cover
        fcntl = None  # type: ignore

    start = time.time()
    with open(lock_path, "a+") as f:
        if fcntl is not None:
            while True:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if (time.time() - start) > float(timeout_s):
                        raise TimeoutError(f"Timed out waiting for metadata lock: {lock_path}")
                    time.sleep(float(poll_s))
        yield
        if fcntl is not None:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


def save_runs_summary_csv(metadata: Dict[str, Any], logger: Optional[logging.Logger] = None) -> Path:
    """
    Save a CSV summary of all runs to runs_summary/ folder.
    
    Creates a CSV with columns:
    - run_id: The unique run identifier
    - temperature: The temperature setting for this run
    - status: completed, failed, or skipped
    - config_file: The config JSON used
    - started_at: Timestamp when run started
    - completed_at: Timestamp when run completed
    - run_dir: Full path to run directory
    - error_message: Error details if failed (empty if successful)
    
    Returns:
        Path to the saved CSV file
    """
    log = logger or logging.getLogger("expanded_experiments")
    
    # Ensure directory exists
    RUNS_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamped filename.
    # Include microseconds so concurrent HPC jobs don't collide on the same second.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    csv_path = RUNS_SUMMARY_DIR / f"runs_summary_{timestamp}.csv"
    
    # Also maintain a "latest" symlink/copy
    latest_path = RUNS_SUMMARY_DIR / "runs_summary_latest.csv"
    
    # Build rows
    rows = []
    experiments = metadata.get("experiments", {})

    def _sort_key(k: str):
        try:
            return (0, float(k))
        except ValueError:
            pass
        parts = k.split("_", 1)
        if len(parts) == 2:
            try:
                return (1, parts[0], float(parts[1]))
            except ValueError:
                pass
        return (2, k)

    for exp_key in sorted(experiments.keys(), key=_sort_key):
        info = experiments[exp_key]
        temp = info.get("temperature")
        if temp is None:
            try:
                temp = float(exp_key.split("_")[-1]) if "_" in exp_key else float(exp_key)
            except (ValueError, IndexError):
                temp = 0.0
        rows.append({
            "run_id": info.get("run_id", ""),
            "experiment_key": exp_key,
            "model_variant": info.get("model_variant", ""),
            "temperature": temp,
            "status": info.get("status", "unknown"),
            "config_file": info.get("config_file", ""),
            "started_at": info.get("started_at", ""),
            "completed_at": info.get("completed_at", ""),
            "run_dir": info.get("run_dir", ""),
            "db_path": info.get("db_path", ""),
            "error_message": info.get("error_message", "") or "",
        })

    # Write CSV
    if rows:
        import csv
        fieldnames = ["run_id", "experiment_key", "model_variant", "temperature", "status", "config_file",
                     "started_at", "completed_at", "run_dir", "db_path", "error_message"]
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        # Copy to latest
        with open(latest_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        log.info(f"Saved runs summary CSV: {csv_path}")
        log.info(f"Updated latest summary: {latest_path}")
    else:
        log.warning("No experiments to save in CSV summary")
    
    return csv_path


def run_single_experiment(
    config_path: Path,
    runs_dir: Path,
    temperature: float,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    dry_run: bool = False,
    model_variant: Optional[str] = None,
    config_label: Optional[str] = None,
    resume_run_id: Optional[str] = None,
    resume_auto: bool = False,
) -> ExperimentResult:
    """
    Run a single experiment and return the result.
    
    Args:
        config_path: Path to the suite config JSON
        runs_dir: Base directory for run outputs
        temperature: Temperature value for this experiment
        api_base: Optional API base URL (e.g., Ollama)
        api_key: Optional API key
        logger: Logger instance
        dry_run: If True, don't actually run the experiment
        
    Returns:
        ExperimentResult with run details
    """
    log = logger or logging.getLogger("expanded_experiments")
    config_file = config_label or config_path.name
    started_at = datetime.now().isoformat()
    
    if dry_run:
        label = f"{model_variant} T={temperature}" if model_variant else f"T={temperature}"
        log.info(f"[DRY RUN] Would run experiment for {label}")
        log.info(f"  Config: {config_path}")
        log.info(f"  Runs dir: {runs_dir}")
        if resume_auto and not resume_run_id:
            log.info("  [DRY RUN] Would pass --resume-auto to vivarium olmo-conformity")
        return ExperimentResult(
            temperature=temperature,
            run_id="dry-run-placeholder",
            run_dir=str(runs_dir / "dry-run-placeholder"),
            db_path=str(runs_dir / "dry-run-placeholder" / "simulation.db"),
            config_file=config_file,
            status="skipped",
            started_at=started_at,
            completed_at=datetime.now().isoformat(),
            model_variant=model_variant,
        )
    
    def _sanitize_cmd_for_log(cmd_parts: List[str]) -> str:
        """Redact known secret-bearing CLI args before logging command."""
        sanitized: List[str] = []
        skip_next = False
        secret_flags = {"--api-key"}
        for i, part in enumerate(cmd_parts):
            if skip_next:
                skip_next = False
                continue
            if part in secret_flags:
                sanitized.extend([part, "***REDACTED***"])
                if i + 1 < len(cmd_parts):
                    skip_next = True
                continue
            sanitized.append(part)
        return " ".join(sanitized)

    def _discover_partial_run(runs_base: Path, started_iso: str) -> Tuple[str, str, str]:
        """
        Best-effort discovery of a run_id/run_dir/db_path if subprocess failed after
        creating the run database.
        """
        try:
            started_ts = datetime.fromisoformat(started_iso).timestamp()
        except Exception:
            started_ts = time.time() - 6 * 3600

        candidates: List[Path] = []
        for p in runs_base.iterdir():
            if not p.is_dir():
                continue
            db = p / "simulation.db"
            if not db.exists():
                continue
            try:
                mt = db.stat().st_mtime
            except Exception:
                continue
            # Include dirs touched around this run window.
            if mt >= (started_ts - 300):
                candidates.append(p)

        candidates.sort(key=lambda p: (p / "simulation.db").stat().st_mtime, reverse=True)
        for d in candidates:
            db = d / "simulation.db"
            try:
                import sqlite3
                conn = sqlite3.connect(str(db))
                cur = conn.cursor()
                cur.execute("SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1")
                row = cur.fetchone()
                conn.close()
                if row and row[0]:
                    return str(row[0]), str(d), str(db)
            except Exception:
                continue
        return "", "", ""

    # Build command
    cmd = [
        sys.executable, "-m", "vivarium", "olmo-conformity",
        "--suite-config", str(config_path),
        "--runs-dir", str(runs_dir),
    ]
    if resume_run_id:
        cmd.extend(["--run-id", resume_run_id])
    elif resume_auto:
        cmd.append("--resume-auto")
    if api_base:
        cmd.extend(["--api-base", api_base])
    if api_key:
        cmd.extend(["--api-key", api_key])
    
    label = f"{model_variant} T={temperature}" if model_variant else f"T={temperature}"
    log.info(f"Running experiment for {label}...")
    log.info(f"  Command: {_sanitize_cmd_for_log(cmd)}")
    
    try:
        # Ensure models directory exists before running
        models_cache_dir = REPO_ROOT / "models" / "huggingface_cache"
        models_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure subprocess can import vivarium: add repo src to PYTHONPATH (needed on HPC when not pip-installed)
        env = os.environ.copy()
        src_dir = str(REPO_ROOT / "src")
        env["PYTHONPATH"] = src_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        
        # Run the experiment with explicit environment to ensure VIVARIUM_* vars are passed
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=env,
        )
        
        if result.returncode != 0:
            log.error(f"Experiment failed with return code {result.returncode}")
            log.error(f"STDERR: {result.stderr}")
            recovered_run_id, recovered_run_dir, recovered_db_path = _discover_partial_run(runs_dir, started_at)
            if recovered_run_id:
                log.warning(
                    "Recovered partial run after failure: "
                    f"run_id={recovered_run_id}, run_dir={recovered_run_dir}"
                )
            return ExperimentResult(
                temperature=temperature,
                run_id=recovered_run_id,
                run_dir=recovered_run_dir,
                db_path=recovered_db_path,
                config_file=config_file,
                status="failed",
                error_message=result.stderr[:1000] if result.stderr else "Unknown error",
                started_at=started_at,
                completed_at=datetime.now().isoformat(),
                model_variant=model_variant,
            )
        
        # Parse output for run_dir and db
        run_dir = ""
        db_path = ""
        for line in result.stdout.splitlines():
            if line.startswith("run_dir="):
                run_dir = line.split("=", 1)[1].strip()
            elif line.startswith("db="):
                db_path = line.split("=", 1)[1].strip()
        
        if not run_dir:
            log.error("Could not parse run_dir from output")
            log.error(f"STDOUT: {result.stdout}")
            return ExperimentResult(
                temperature=temperature,
                run_id="",
                run_dir="",
                db_path="",
                config_file=config_file,
                status="failed",
                error_message="Could not parse run_dir from output",
                started_at=started_at,
                completed_at=datetime.now().isoformat(),
                model_variant=model_variant,
            )
        
        # Extract run_id from run_dir (format: YYYYMMDD_HHMMSS_run-id)
        run_id = run_dir.split("_")[-1] if "_" in run_dir else os.path.basename(run_dir)
        
        log.info(f"Experiment completed successfully")
        log.info(f"  Run ID: {run_id}")
        log.info(f"  Run dir: {run_dir}")
        
        return ExperimentResult(
            temperature=temperature,
            run_id=run_id,
            run_dir=run_dir,
            db_path=db_path,
            config_file=config_file,
            status="completed",
            started_at=started_at,
            completed_at=datetime.now().isoformat(),
            model_variant=model_variant,
        )

    except Exception as e:
        log.exception(f"Exception running experiment: {e}")
        return ExperimentResult(
            temperature=temperature,
            run_id="",
            run_dir="",
            db_path="",
            config_file=config_file,
            status="failed",
            error_message=str(e),
            started_at=started_at,
            completed_at=datetime.now().isoformat(),
            model_variant=model_variant,
        )


def generate_per_run_report(
    run_id: str,
    db_path: str,
    run_dir: str,
    runs_dir: Path,
    logger: Optional[logging.Logger] = None,
    dry_run: bool = False,
) -> bool:
    """
    Generate analysis report for a single run.
    
    db_path and run_dir from metadata may be from another machine; we resolve
    paths from the current runs_dir so the report finds the DB on this host.
    
    Returns:
        True if successful, False otherwise
    """
    log = logger or logging.getLogger("expanded_experiments")
    
    if dry_run:
        log.info(f"[DRY RUN] Would generate report for run {run_id[:8]}...")
        return True
    
    # Resolve paths from current runs_dir so report works when metadata paths
    # were recorded on another machine (e.g. after git pull) or are stale
    run_dir_basename = os.path.basename(run_dir.rstrip("/")) or run_dir
    resolved_run_dir = runs_dir / run_dir_basename
    resolved_db_path = resolved_run_dir / "simulation.db"
    
    if not resolved_db_path.exists():
        log.warning(f"DB not found at {resolved_db_path}, skipping report (run may be on another node or path)")
        return False
    
    log.info(f"Generating report for run {run_id[:8]}...")
    
    cmd = [
        sys.executable, "-m", "vivarium", "olmo-conformity-report",
        "--run-id", run_id,
        "--db", str(resolved_db_path),
        "--run-dir", str(resolved_run_dir),
    ]
    
    env = os.environ.copy()
    src_dir = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=env,
        )
        
        if result.returncode != 0:
            log.error(f"Report generation failed: {result.stderr}")
            return False
        
        log.info(f"Report generated successfully for {run_id[:8]}")
        return True
        
    except Exception as e:
        log.exception(f"Exception generating report: {e}")
        return False


def generate_combined_analysis(
    metadata: Dict[str, Any],
    output_dir: Path,
    runs_dir: Path,
    logger: Optional[logging.Logger] = None,
    dry_run: bool = False,
) -> bool:
    """
    Generate combined cross-temperature analysis.
    
    This function creates analysis comparing all temperature runs.
    
    MEMORY OPTIMIZATION: 
    - Data is loaded incrementally and cleaned up after use
    - gc.collect() is called after major processing steps
    - Only rates_dict (small summary data) is kept in memory for plotting
    
    Returns:
        True if successful, False otherwise
    """
    import gc
    
    log = logger or logging.getLogger("expanded_experiments")
    
    if dry_run:
        log.info("[DRY RUN] Would generate combined analysis...")
        return True
    
    log.info("Generating combined cross-temperature analysis...")
    
    # Create output directories
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Build runs dictionary for analysis
    experiments = metadata.get("experiments", {})
    runs_dict: Dict[Any, Dict[str, str]] = {}
    completed_temps: List[float] = []
    model_first = False

    for exp_key, info in experiments.items():
        if info.get("status") != "completed":
            continue
        if "_" in exp_key:
            parts = exp_key.rsplit("_", 1)
            if len(parts) == 2:
                try:
                    variant, temp_str = parts[0], parts[1]
                    temp = float(temp_str)
                    model_first = True
                    completed_temps.append(temp)
                    runs_dict[(variant, temp)] = {
                        "dir": os.path.basename(info["run_dir"]),
                        "id": info["run_id"],
                    }
                except ValueError:
                    pass
        else:
            try:
                temp = float(exp_key)
                completed_temps.append(temp)
                runs_dict[temp] = {
                    "dir": os.path.basename(info["run_dir"]),
                    "id": info["run_id"],
                }
            except ValueError:
                pass

    if len(completed_temps) < 2 and len(runs_dict) < 2:
        log.warning("Need at least 2 completed runs for comparison analysis")
        return False

    completed_temps = sorted(set(completed_temps))
    log.info(f"Analyzing {len(completed_temps)} temperature levels: {completed_temps} (model_first={model_first})")
    
    try:
        # Import analysis functions
        sys.path.insert(0, str(REPO_ROOT / "Analysis Scripts"))
        
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from scipy import stats
        import sqlite3
        
        # Set plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # MEMORY OPTIMIZATION: Only keep rates_dict (small summaries) in memory
        # Process behavioral data incrementally and release it
        rates_dict = {}
        combined_rates = []  # For combined CSV

        query = """
        WITH first_outputs AS (
            SELECT trial_id, MIN(created_at) AS min_created_at
            FROM conformity_outputs
            GROUP BY trial_id
        ),
        first_output_ids AS (
            SELECT MIN(o.output_id) AS output_id, o.trial_id
            FROM conformity_outputs o
            JOIN first_outputs fo
              ON fo.trial_id = o.trial_id AND fo.min_created_at = o.created_at
            GROUP BY o.trial_id
        )
        SELECT 
            t.trial_id,
            t.model_id,
            t.variant,
            c.name AS condition_name,
            i.question,
            i.ground_truth_text,
            i.domain,
            d.name AS dataset_name,
            o.raw_text,
            o.parsed_answer_text,
            o.is_correct,
            o.refusal_flag,
            o.latency_ms
        FROM conformity_trials t
        JOIN conformity_conditions c ON t.condition_id = c.condition_id
        JOIN conformity_items i ON t.item_id = i.item_id
        JOIN conformity_datasets d ON d.dataset_id = i.dataset_id
        JOIN first_output_ids foi ON foi.trial_id = t.trial_id
        JOIN conformity_outputs o ON o.output_id = foi.output_id
        WHERE t.run_id = ?
          AND c.name IN ('control', 'asch_history_5', 'authoritative_bias')
          AND o.is_correct IS NOT NULL;
        """

        if model_first:
            for temp in completed_temps:
                temp_rates = []
                for key in list(runs_dict.keys()):
                    if not isinstance(key, tuple) or key[1] != temp:
                        continue
                    variant, _ = key
                    run_info = runs_dict[key]
                    db_path = runs_dir / run_info["dir"] / "simulation.db"
                    if not db_path.exists():
                        log.warning(f"Database not found for {variant} T={temp}: {db_path}")
                        continue
                    log.info(f"Loading data for {variant} T={temp}...")
                    conn = sqlite3.connect(str(db_path))
                    conn.row_factory = sqlite3.Row
                    df = pd.read_sql_query(query, conn, params=[run_info["id"]])
                    conn.close()
                    df["is_empty"] = df["raw_text"].isna() | (df["raw_text"] == "")
                    rates = df.groupby(["condition_name", "variant"]).agg(
                        n_trials=("trial_id", "count"),
                        n_correct=("is_correct", "sum"),
                        n_refusals=("refusal_flag", "sum"),
                    ).reset_index()
                    rates["accuracy"] = rates["n_correct"] / rates["n_trials"]
                    rates["error_rate"] = 1 - rates["accuracy"]
                    rates["refusal_rate"] = rates["n_refusals"] / rates["n_trials"]
                    rates["n_incorrect"] = rates["n_trials"] - rates["n_correct"]
                    temp_rates.append(rates)
                    del df
                    gc.collect()
                if temp_rates:
                    rates_dict[temp] = pd.concat(temp_rates, ignore_index=True)
                    rates_dict[temp].to_csv(tables_dir / f"rates_t{temp}.csv", index=False)
                    rates_with_temp = rates_dict[temp].copy()
                    rates_with_temp["temperature"] = temp
                    combined_rates.append(rates_with_temp)
                    log.info(f"  Processed T={temp} ({len(temp_rates)} variants), memory cleaned")
        else:
            for temp in completed_temps:
                run_info = runs_dict[temp]
                db_path = runs_dir / run_info["dir"] / "simulation.db"
                if not db_path.exists():
                    log.warning(f"Database not found for T={temp}: {db_path}")
                    continue
                log.info(f"Loading data for T={temp}...")
                conn = sqlite3.connect(str(db_path))
                conn.row_factory = sqlite3.Row
            
                df = pd.read_sql_query(query, conn, params=[run_info["id"]])
                conn.close()
                df["is_empty"] = df["raw_text"].isna() | (df["raw_text"] == "")

                rates = df.groupby(["condition_name", "variant"]).agg(
                    n_trials=("trial_id", "count"),
                    n_correct=("is_correct", "sum"),
                    n_refusals=("refusal_flag", "sum"),
                ).reset_index()
                rates["accuracy"] = rates["n_correct"] / rates["n_trials"]
                rates["error_rate"] = 1 - rates["accuracy"]
                rates["refusal_rate"] = rates["n_refusals"] / rates["n_trials"]
                rates["n_incorrect"] = rates["n_trials"] - rates["n_correct"]
                rates_dict[temp] = rates
                rates.to_csv(tables_dir / f"rates_t{temp}.csv", index=False)
                rates_with_temp = rates.copy()
                rates_with_temp["temperature"] = temp
                combined_rates.append(rates_with_temp)
                del df
                gc.collect()
                log.info(f"  Processed T={temp}, memory cleaned")
        
        if not rates_dict:
            log.error("No data loaded from any runs")
            return False
        
        # Save combined CSV
        log.info("Saving combined data tables...")
        combined_df = pd.concat(combined_rates, ignore_index=True)
        combined_df.to_csv(tables_dir / "rates_combined.csv", index=False)
        del combined_df, combined_rates
        gc.collect()
        
        # Generate figures (rates_dict is small - OK to keep in memory)
        log.info("Generating comparison figures...")
        
        # Figure 1: Error rates by condition and temperature
        _plot_error_rates_comparison(rates_dict, completed_temps, figures_dir, log)
        gc.collect()
        
        # Figure 2: Temperature curves
        _plot_temperature_curves(rates_dict, completed_temps, figures_dir, log)
        gc.collect()
        
        # Figure 3: Social pressure effect
        _plot_social_pressure_effect(rates_dict, completed_temps, figures_dir, log)
        gc.collect()
        
        # Figure 4: Heatmap
        _plot_heatmap(rates_dict, completed_temps, figures_dir, log)
        gc.collect()
        
        # Generate summary report
        _generate_summary_report(metadata, rates_dict, completed_temps, output_dir, log)
        
        # Final cleanup
        del rates_dict
        gc.collect()
        
        log.info(f"Combined analysis saved to {output_dir}")
        return True
        
    except Exception as e:
        log.exception(f"Exception generating combined analysis: {e}")
        return False


def _plot_error_rates_comparison(
    rates_dict: Dict[float, Any],
    temps: List[float],
    output_dir: Path,
    log: logging.Logger,
) -> None:
    """Plot error rates comparison across temperatures."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    conditions = ['control', 'asch_history_5', 'authoritative_bias']
    variants = sorted(rates_dict[temps[0]]['variant'].unique())
    
    # Create color map for temperatures
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(temps)))
    temp_colors = dict(zip(temps, colors))
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 11), sharey=True)
    axes = axes.flatten()
    
    for idx, variant in enumerate(variants):
        if idx >= len(axes):
            break
        ax = axes[idx]
        
        x = np.arange(len(conditions))
        width = 0.8 / len(temps)
        
        for i, temp in enumerate(temps):
            data = rates_dict[temp][rates_dict[temp]['variant'] == variant].set_index('condition_name')
            rates = [data.loc[c, 'error_rate'] if c in data.index else 0 for c in conditions]
            
            offset = (i - len(temps)/2 + 0.5) * width
            ax.bar(x + offset, rates, width, label=f"T={temp}", color=temp_colors[temp], alpha=0.85)
        
        ax.set_ylabel('Error Rate' if idx % 3 == 0 else '')
        ax.set_title(f'{variant}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Control', 'Asch', 'Authority'], rotation=15, ha='right')
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    for idx in range(len(variants), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Error Rates by Condition and Temperature', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'error_rates_by_temperature.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'error_rates_by_temperature.pdf', bbox_inches='tight')
    plt.close()
    log.info("  Saved error_rates_by_temperature")


def _plot_temperature_curves(
    rates_dict: Dict[float, Any],
    temps: List[float],
    output_dir: Path,
    log: logging.Logger,
) -> None:
    """Plot temperature-error rate curves."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    conditions = ['control', 'asch_history_5', 'authoritative_bias']
    condition_titles = {
        'control': 'Control (No Pressure)',
        'asch_history_5': 'Asch (5 Confederates)',
        'authoritative_bias': 'Authoritative Bias'
    }
    
    variants = sorted(rates_dict[temps[0]]['variant'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(variants)))
    variant_colors = dict(zip(variants, colors))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    for idx, condition in enumerate(conditions):
        ax = axes[idx]
        
        for variant in variants:
            rates = []
            for temp in temps:
                data = rates_dict[temp]
                row = data[(data['condition_name'] == condition) & (data['variant'] == variant)]
                if not row.empty:
                    rates.append(row['error_rate'].iloc[0])
                else:
                    rates.append(np.nan)
            
            ax.plot(temps, rates, 'o-', label=variant, color=variant_colors[variant], 
                   linewidth=2, markersize=8)
        
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Error Rate' if idx == 0 else '')
        ax.set_title(condition_titles.get(condition, condition))
        ax.set_xticks(temps)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        
        if idx == 2:
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
    
    plt.suptitle('Error Rate vs Temperature by Condition', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'temperature_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'temperature_curves.pdf', bbox_inches='tight')
    plt.close()
    log.info("  Saved temperature_curves")


def _plot_social_pressure_effect(
    rates_dict: Dict[float, Any],
    temps: List[float],
    output_dir: Path,
    log: logging.Logger,
) -> None:
    """Plot social pressure effect by temperature."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    variants = sorted(rates_dict[temps[0]]['variant'].unique())
    x = np.arange(len(variants))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(temps)))
    temp_colors = dict(zip(temps, colors))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.8 / len(temps)
    
    for i, temp in enumerate(temps):
        data = rates_dict[temp]
        
        effects = []
        for variant in variants:
            control_row = data[(data['condition_name'] == 'control') & (data['variant'] == variant)]
            asch_row = data[(data['condition_name'] == 'asch_history_5') & (data['variant'] == variant)]
            
            if not control_row.empty and not asch_row.empty:
                control_rate = control_row['error_rate'].iloc[0]
                asch_rate = asch_row['error_rate'].iloc[0]
                effect = asch_rate - control_rate
            else:
                effect = 0
            effects.append(effect)
        
        offset = (i - len(temps)/2 + 0.5) * width
        ax.bar(x + offset, effects, width, label=f"T={temp}", color=temp_colors[temp], alpha=0.85)
    
    ax.set_ylabel('Social Pressure Effect\n(Asch Error Rate - Control Error Rate)')
    ax.set_xlabel('Model Variant')
    ax.set_title('Social Pressure Effect by Temperature\n(Positive = Pressure Increases Errors)')
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=30, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'social_pressure_effect.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'social_pressure_effect.pdf', bbox_inches='tight')
    plt.close()
    log.info("  Saved social_pressure_effect")


def _plot_heatmap(
    rates_dict: Dict[float, Any],
    temps: List[float],
    output_dir: Path,
    log: logging.Logger,
) -> None:
    """Plot error rate heatmap."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    conditions = ['control', 'asch_history_5', 'authoritative_bias']
    condition_labels = {'control': 'Control', 'asch_history_5': 'Asch', 'authoritative_bias': 'Authority'}
    
    variants = sorted(rates_dict[temps[0]]['variant'].unique())
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    for idx, condition in enumerate(conditions):
        ax = axes[idx]
        
        # Build matrix: rows = variants, cols = temperatures
        matrix = np.zeros((len(variants), len(temps)))
        
        for i, variant in enumerate(variants):
            for j, temp in enumerate(temps):
                data = rates_dict[temp]
                row = data[(data['condition_name'] == condition) & (data['variant'] == variant)]
                if not row.empty:
                    matrix[i, j] = row['error_rate'].iloc[0]
        
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0.3, vmax=1.0)
        
        ax.set_xticks(np.arange(len(temps)))
        ax.set_xticklabels([f"T={t}" for t in temps])
        ax.set_yticks(np.arange(len(variants)))
        ax.set_yticklabels(variants if idx == 0 else [])
        ax.set_title(condition_labels.get(condition, condition))
        ax.set_xlabel('Temperature')
        if idx == 0:
            ax.set_ylabel('Model Variant')
        
        # Add text annotations
        for i in range(len(variants)):
            for j in range(len(temps)):
                ax.text(j, i, f'{matrix[i, j]:.1%}',
                       ha='center', va='center', fontsize=8,
                       color='white' if matrix[i, j] > 0.65 else 'black')
    
    plt.colorbar(im, ax=axes, label='Error Rate', shrink=0.8)
    plt.suptitle('Error Rate Heatmap (Model x Temperature)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'error_rate_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'error_rate_heatmap.pdf', bbox_inches='tight')
    plt.close()
    log.info("  Saved error_rate_heatmap")


def _generate_summary_report(
    metadata: Dict[str, Any],
    rates_dict: Dict[float, Any],
    temps: List[float],
    output_dir: Path,
    log: logging.Logger,
) -> None:
    """Generate markdown summary report."""
    lines = [
        "# Cross-Temperature Conformity Analysis Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Experiment Summary",
        "",
        f"- **Temperature Levels Analyzed**: {temps}",
        f"- **Number of Experiments**: {len(temps)}",
        "",
        "## Run Information",
        "",
        "| Temperature | Run ID | Status | Config |",
        "|-------------|--------|--------|--------|",
    ]
    
    for temp_str, info in sorted(metadata.get("experiments", {}).items()):
        run_id = info.get("run_id", "N/A")[:8] + "..." if info.get("run_id") else "N/A"
        status = info.get("status", "unknown")
        config = info.get("config_file", "N/A")
        lines.append(f"| {temp_str} | {run_id} | {status} | {config} |")
    
    lines.extend([
        "",
        "## Key Findings",
        "",
        "### Error Rates by Model",
        "",
    ])
    
    # Add error rate summary
    for temp in temps:
        rates = rates_dict[temp]
        avg_error = rates['error_rate'].mean()
        lines.append(f"- **T={temp}**: Average error rate = {avg_error:.1%}")
    
    lines.extend([
        "",
        "## Output Files",
        "",
        "### Figures",
        "- `figures/error_rates_by_temperature.png` - Error rates by condition and temperature",
        "- `figures/temperature_curves.png` - Error rate vs temperature curves",
        "- `figures/social_pressure_effect.png` - Social pressure effect by temperature",
        "- `figures/error_rate_heatmap.png` - Error rate heatmap",
        "",
        "### Tables",
        "- `tables/rates_combined.csv` - Combined error rates across all temperatures",
        "- `tables/rates_t{X}.csv` - Per-temperature error rates",
        "",
    ])
    
    report_path = output_dir / "analysis_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    
    log.info(f"  Saved analysis_report.md")


def _run_judge_phase(
    metadata: Dict[str, Any],
    runs_dir: Path,
    logger: logging.Logger,
    dry_run: bool,
) -> None:
    """Run olmo-conformity-judgeval on each completed run."""
    experiments = metadata.get("experiments", {})
    for exp_key, info in experiments.items():
        if info.get("status") != "completed":
            logger.info(f"Skipping {exp_key} (status={info.get('status')})")
            continue
        run_id = info.get("run_id", "")
        run_dir = info.get("run_dir", "")
        db_path = info.get("db_path", "")
        run_dir_basename = os.path.basename(run_dir.rstrip("/")) or run_dir
        resolved_run_dir = runs_dir / run_dir_basename
        resolved_db = resolved_run_dir / "simulation.db"
        if not resolved_db.exists():
            logger.warning(f"DB not found for {exp_key}: {resolved_db}, skipping judge")
            continue
        if dry_run:
            logger.info(f"[DRY RUN] Would run judgeval for {exp_key} (run_id={run_id[:8]}...)")
            continue
        cmd = [
            sys.executable, "-m", "vivarium", "olmo-conformity-judgeval",
            "--run-id", run_id,
            "--db", str(resolved_db),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT / "src") + (os.pathsep + env.get("PYTHONPATH", ""))
        logger.info(f"Running judgeval for {exp_key}...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), env=env)
        if result.returncode != 0:
            logger.error(f"Judgeval failed for {exp_key}: {result.stderr[:500]}")
        else:
            logger.info(f"Judgeval completed for {exp_key}")


def _run_posthoc_phase(
    metadata: Dict[str, Any],
    runs_dir: Path,
    logger: logging.Logger,
    dry_run: bool,
) -> None:
    """Run olmo-conformity-posthoc on each completed run."""
    experiments = metadata.get("experiments", {})
    for exp_key, info in experiments.items():
        if info.get("status") != "completed":
            logger.info(f"Skipping {exp_key} (status={info.get('status')})")
            continue
        run_id = info.get("run_id", "")
        run_dir = info.get("run_dir", "")
        run_dir_basename = os.path.basename(run_dir.rstrip("/")) or run_dir
        resolved_run_dir = runs_dir / run_dir_basename
        resolved_db = resolved_run_dir / "simulation.db"
        if not resolved_db.exists():
            logger.warning(f"DB not found for {exp_key}: {resolved_db}, skipping posthoc")
            continue
        if dry_run:
            logger.info(f"[DRY RUN] Would run posthoc for {exp_key} (run_id={run_id[:8]}...)")
            continue
        cmd = [
            sys.executable, "-m", "vivarium", "olmo-conformity-posthoc",
            "--run-dir", str(resolved_run_dir),
            "--db", str(resolved_db),
            "--run-id", run_id,
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT / "src") + (os.pathsep + env.get("PYTHONPATH", ""))
        logger.info(f"Running posthoc for {exp_key}...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), env=env)
        if result.returncode != 0:
            logger.error(f"Posthoc failed for {exp_key}: {result.stderr[:500]}")
        else:
            logger.info(f"Posthoc completed for {exp_key}")


def main():
    parser = argparse.ArgumentParser(
        description="Automated pipeline for expanded conformity experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--hpc",
        action="store_true",
        help="Use HPC paths from paths.json (default: use local repo paths)",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=None,
        help=f"Base directory for experiment outputs (default: {DEFAULT_RUNS_DIR} for local, paths.json for HPC)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help=f"Base directory for model files (default: {DEFAULT_MODELS_DIR} for local, paths.json for HPC)",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="API base URL for remote inference (e.g., http://localhost:11434/v1 for Ollama)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for remote inference (optional)",
    )
    parser.add_argument(
        "--skip-runs",
        action="store_true",
        help="Skip running experiments, use existing runs_metadata.json",
    )
    parser.add_argument(
        "--only-analysis",
        action="store_true",
        help="Only regenerate combined analysis (requires existing metadata)",
    )
    parser.add_argument(
        "--runs-only",
        action="store_true",
        help="Run experiments (phase 1) and update metadata only (skip per-run reports and combined analysis)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Run experiments even if metadata marks them completed (e.g. when runs were on another machine)",
    )
    parser.add_argument(
        "--temps",
        type=str,
        default=None,
        help="Comma-separated list of temperatures to run (default: all)",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        help="Path to suite config (default: test_scripts/suite_expanded_localtest_by_model.json). Suite defines models and temperatures.",
    )
    parser.add_argument(
        "--by-temp",
        action="store_true",
        help="Use legacy temperature-first behavior (suite_expanded_temp{T}.json per temp)",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["trials", "judge", "posthoc", "all"],
        default="all",
        help="Pipeline phase: trials (run experiments), judge (run judgeval on completed runs), posthoc (run posthoc on completed runs), all (trials then judge then posthoc)",
    )
    parser.add_argument(
        "--no-sleep",
        action="store_true",
        help="Prevent macOS from sleeping during experiments (uses caffeinate)",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Path to runs metadata JSON (default: runs_metadata.json, or runs_metadata_32b.json when suite is a 32B suite). Use to scope the 32B limited study to a dedicated file.",
    )
    parser.add_argument(
        "--resume-run-id",
        type=str,
        default=None,
        help="Resume an existing run by its UUID. Passes --run-id to the runner so it reuses the DB and skips completed trials.",
    )
    parser.add_argument(
        "--resume-auto",
        action="store_true",
        help="Forward --resume-auto to vivarium olmo-conformity to resume the best matching incomplete run under runs_dir (ignored if --resume-run-id is set).",
    )

    args = parser.parse_args()

    if args.runs_only and (args.skip_runs or args.only_analysis):
        print("Error: --runs-only cannot be combined with --skip-runs or --only-analysis", file=sys.stderr)
        return 2

    if args.phase == "judge" and (args.skip_runs or args.only_analysis):
        print("Error: --phase judge requires runs metadata; omit --skip-runs and --only-analysis", file=sys.stderr)
        return 2

    if args.phase == "posthoc" and (args.skip_runs or args.only_analysis):
        print("Error: --phase posthoc requires runs metadata; omit --skip-runs and --only-analysis", file=sys.stderr)
        return 2

    # Parse temperatures (used for legacy --by-temp mode)
    if args.temps:
        temps_to_run = [float(t.strip()) for t in args.temps.split(",")]
    else:
        temps_to_run = TEMPERATURES

    # Resolve suite for model-first mode (suite defines models + temperatures)
    suite_path: Optional[Path] = None
    suite: Optional[Dict[str, Any]] = None
    if not args.by_temp:
        suite_path = Path(args.suite) if args.suite else DEFAULT_SUITE
        if suite_path.exists():
            suite = load_suite(suite_path)
    
    # Determine paths based on mode (HPC vs local)
    if args.hpc:
        # HPC mode: load paths from paths.json
        hpc_paths = load_hpc_paths()
        models_dir = Path(args.models_dir) if args.models_dir else Path(hpc_paths.get("models_dir", str(DEFAULT_MODELS_DIR)))
        runs_dir = Path(args.runs_dir) if args.runs_dir else Path(hpc_paths.get("runs_dir", str(DEFAULT_RUNS_DIR)))
    else:
        # Local mode: use repo-relative defaults
        models_dir = Path(args.models_dir) if args.models_dir else DEFAULT_MODELS_DIR
        runs_dir = Path(args.runs_dir) if args.runs_dir else DEFAULT_RUNS_DIR
    # Metadata file: explicit --metadata, or 32B-scoped file for 32B suites, or default
    if args.metadata:
        metadata_path = Path(args.metadata)
        if not metadata_path.is_absolute():
            metadata_path = REPO_ROOT / metadata_path
    elif suite and (
        "32b" in str(suite_path).lower() or "32b" in suite.get("suite_name", "").lower()
    ):
        metadata_path = COMPARING_EXPERIMENTS_DIR / "runs_metadata_32b.json"
    else:
        metadata_path = COMPARING_EXPERIMENTS_DIR / "runs_metadata.json"
    # If we split temperatures into separate HPC jobs, multiple processes would write
    # to the same rotating log file (not multiprocess-safe). Use per-temp log files
    # for single-temp runs-only jobs.
    if args.runs_only and len(temps_to_run) == 1:
        temp_slug = str(temps_to_run[0]).replace(".", "p")
        log_path = COMPARING_EXPERIMENTS_DIR / f"analysis_log_t{temp_slug}.txt"
    else:
        log_path = COMPARING_EXPERIMENTS_DIR / "analysis_log.txt"
    
    # Setup logging
    COMPARING_EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_path if not args.dry_run else None)
    
    logger.info("=" * 60)
    logger.info("EXPANDED CONFORMITY EXPERIMENTS PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Repo root: {REPO_ROOT}")
    logger.info(f"Mode: {'HPC' if args.hpc else 'LOCAL'}")
    logger.info(f"Metadata file: {metadata_path}")
    logger.info(f"Models directory: {models_dir}")
    logger.info(f"Runs directory: {runs_dir}")
    logger.info(f"Output directory: {COMPARING_EXPERIMENTS_DIR}")
    use_model_first = suite is not None
    logger.info(f"Mode: {'model-first' if use_model_first else 'temperature-first (legacy)'}")
    logger.info(f"Phase: {args.phase}")
    if use_model_first:
        logger.info(f"Suite: {suite_path}")
    else:
        logger.info(f"Temperatures: {temps_to_run}")
    log_memory_usage(logger, "pipeline start")
    if args.dry_run:
        logger.info("MODE: DRY RUN (no changes will be made)")
    
    # Prevent macOS from sleeping if requested
    if args.no_sleep:
        start_caffeinate(logger)
    
    # Set environment variables for the runner to use
    # These override paths.json settings when NOT in HPC mode
    if not args.hpc:
        # For local mode, set env vars to override any HPC paths in paths.json
        models_cache_path = models_dir / "huggingface_cache"
        os.environ["VIVARIUM_HF_CACHE"] = str(models_cache_path)
        os.environ["VIVARIUM_ARTIFACTS_DIR"] = str(runs_dir)
        # Back-compat for scripts that still check AAM_*
        os.environ["AAM_MODELS_DIR"] = str(models_cache_path)
        os.environ["AAM_RUNS_DIR"] = str(runs_dir)
        
        # Create directories to ensure they exist
        models_cache_path.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Set VIVARIUM_HF_CACHE={os.environ['VIVARIUM_HF_CACHE']}")
        logger.info(f"Set VIVARIUM_ARTIFACTS_DIR={os.environ['VIVARIUM_ARTIFACTS_DIR']}")
        logger.info(f"Created/verified models cache directory: {models_cache_path}")
        logger.info(f"Created/verified runs directory: {runs_dir}")
    
    # Load existing metadata
    metadata = load_metadata(metadata_path)
    phase1_results: List[ExperimentResult] = []
    
    # Phase 1: Run experiments (unless skipped)
    run_trials = (args.phase in ("trials", "all")) and not args.skip_runs and not args.only_analysis
    if run_trials:
        logger.info("")
        logger.info("=" * 60)
        logger.info("PHASE 1: RUNNING EXPERIMENTS")
        logger.info("=" * 60)

        if use_model_first:
            # Prefer suite default_temperatures; else if suite specifies run.temperature (single temp), use only that
            if "default_temperatures" in suite:
                default_temps = suite["default_temperatures"]
            else:
                run_block = suite.get("run", {})
                if isinstance(run_block.get("temperature"), (int, float)):
                    default_temps = [float(run_block["temperature"])]
                else:
                    default_temps = TEMPERATURES
            models_specs = suite.get("models", [])

            for model_spec in models_specs:
                variant = model_spec["variant"]
                model_id = model_spec["model_id"]
                temps = model_spec.get("temperatures") or default_temps
                if args.temps:
                    temps = [t for t in temps if t in temps_to_run]

                for temp in temps:
                    exp_key = f"{variant}_{temp}"
                    existing = metadata.get("experiments", {}).get(exp_key, {})
                    if existing.get("status") == "completed" and not args.dry_run and not args.force_rerun:
                        logger.info(f"{exp_key} already completed, skipping (run_id={existing.get('run_id', 'N/A')[:8]}...)")
                        phase1_results.append(
                            ExperimentResult(
                                temperature=temp,
                                run_id=str(existing.get("run_id") or ""),
                                run_dir=str(existing.get("run_dir") or ""),
                                db_path=str(existing.get("db_path") or ""),
                                config_file=str(existing.get("config_file") or ""),
                                status="skipped",
                                error_message=None,
                                started_at=None,
                                completed_at=None,
                                model_variant=variant,
                            )
                        )
                        continue

                    config = build_run_config(suite, variant, model_id, temp)
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=True) as f:
                        json.dump(config, f, indent=2)
                        f.flush()
                        config_path = Path(f.name)

                        logger.info("")
                        logger.info(f"--- {variant} T={temp} ---")

                        result = run_single_experiment(
                            config_path=config_path,
                            runs_dir=runs_dir,
                            temperature=temp,
                            api_base=args.api_base,
                            api_key=args.api_key,
                            logger=logger,
                            dry_run=args.dry_run,
                            model_variant=variant,
                            config_label=f"{suite_path.name} ({variant}, T={temp})",
                            resume_run_id=getattr(args, "resume_run_id", None),
                            resume_auto=bool(getattr(args, "resume_auto", False)),
                        )
                    phase1_results.append(result)

                    if not args.dry_run:
                        with _metadata_lock(metadata_path):
                            latest = load_metadata(metadata_path)
                            latest.setdefault("experiments", {})
                            latest["experiments"][exp_key] = {
                                "temperature": result.temperature,
                                "run_id": result.run_id,
                                "run_dir": result.run_dir,
                                "db_path": result.db_path,
                                "config_file": result.config_file,
                                "status": result.status,
                                "error_message": result.error_message,
                                "started_at": result.started_at,
                                "completed_at": result.completed_at,
                                "model_variant": variant,
                            }
                            save_metadata(latest, metadata_path)
                            logger.info(f"Metadata saved to {metadata_path}")
                            save_runs_summary_csv(latest, logger)
                            metadata = latest

                    import gc
                    gc.collect()
                    log_memory_usage(logger, f"after {exp_key} experiment")

        else:
            # Legacy: temperature-first loop
            for temp in temps_to_run:
                temp_str = str(temp)

                existing = metadata.get("experiments", {}).get(temp_str, {})
                if existing.get("status") == "completed" and not args.dry_run and not args.force_rerun:
                    logger.info(f"T={temp} already completed, skipping (run_id={existing.get('run_id', 'N/A')[:8]}...)")
                    phase1_results.append(
                        ExperimentResult(
                            temperature=temp,
                            run_id=str(existing.get("run_id") or ""),
                            run_dir=str(existing.get("run_dir") or ""),
                            db_path=str(existing.get("db_path") or ""),
                            config_file=str(existing.get("config_file") or ""),
                            status="skipped",
                            error_message=None,
                            started_at=None,
                            completed_at=None,
                        )
                    )
                    continue

                config_file = CONFIG_TEMPLATE.format(temp=temp)
                config_path = CONFIGS_DIR / config_file

                if not config_path.exists():
                    logger.error(f"Config not found: {config_path}")
                    continue

                logger.info("")
                logger.info(f"--- Temperature {temp} ---")

                result = run_single_experiment(
                    config_path=config_path,
                    runs_dir=runs_dir,
                    temperature=temp,
                    api_base=args.api_base,
                    api_key=args.api_key,
                    logger=logger,
                    dry_run=args.dry_run,
                    resume_run_id=getattr(args, "resume_run_id", None),
                    resume_auto=bool(getattr(args, "resume_auto", False)),
                )
                phase1_results.append(result)

                if not args.dry_run:
                    with _metadata_lock(metadata_path):
                        latest = load_metadata(metadata_path)
                        latest.setdefault("experiments", {})
                        latest["experiments"][temp_str] = {
                            "temperature": result.temperature,
                            "run_id": result.run_id,
                            "run_dir": result.run_dir,
                            "db_path": result.db_path,
                            "config_file": result.config_file,
                            "status": result.status,
                            "error_message": result.error_message,
                            "started_at": result.started_at,
                            "completed_at": result.completed_at,
                        }
                        save_metadata(latest, metadata_path)
                        logger.info(f"Metadata saved to {metadata_path}")
                        save_runs_summary_csv(latest, logger)
                        metadata = latest

                import gc
                gc.collect()
                log_memory_usage(logger, f"after T={temp} experiment")

        # If runs-only or phase=trials, stop after experiments.
        if args.runs_only or args.phase == "trials":
            logger.info("")
            logger.info("=" * 60)
            logger.info("STOPPING AFTER TRIALS (runs-only or phase=trials)")
            logger.info("=" * 60)

            failed = [r for r in phase1_results if r.status == "failed"]
            completed = [r for r in phase1_results if r.status == "completed"]
            skipped = [r for r in phase1_results if r.status == "skipped"]

            requested = f"{len(phase1_results)} runs" if use_model_first else f"Temperatures: {temps_to_run}"
            logger.info(f"Requested: {requested}")
            logger.info(f"Completed: {len(completed)}; Failed: {len(failed)}; Skipped: {len(skipped)}")
            for r in failed:
                label = f"{r.model_variant} T={r.temperature}" if r.model_variant else f"T={r.temperature}"
                logger.info(f"  FAILED {label}: {r.error_message or 'Unknown error'}")
            return 1 if failed else 0

    # Phase 1b: Judge (run judgeval on completed runs)
    if args.phase in ("judge", "all"):
        logger.info("")
        logger.info("=" * 60)
        logger.info("PHASE: JUDGE (olmo-conformity-judgeval)")
        logger.info("=" * 60)
        _run_judge_phase(metadata, runs_dir, logger, args.dry_run)

    # Phase 1c: Posthoc (run posthoc on completed runs)
    if args.phase in ("posthoc", "all"):
        logger.info("")
        logger.info("=" * 60)
        logger.info("PHASE: POSTHOC (olmo-conformity-posthoc)")
        logger.info("=" * 60)
        _run_posthoc_phase(metadata, runs_dir, logger, args.dry_run)

    if args.phase in ("judge", "posthoc"):
        logger.info("")
        logger.info("Phase complete (judge/posthoc only)")
        return 0

    # Phase 2: Generate per-run reports
    if not args.only_analysis:
        logger.info("")
        logger.info("=" * 60)
        logger.info("PHASE 2: GENERATING PER-RUN REPORTS")
        logger.info("=" * 60)
        
        for temp_str, info in metadata.get("experiments", {}).items():
            if info.get("status") != "completed":
                logger.info(f"Skipping T={temp_str} (status={info.get('status')})")
                continue
            
            generate_per_run_report(
                run_id=info["run_id"],
                db_path=info["db_path"],
                run_dir=info["run_dir"],
                runs_dir=runs_dir,
                logger=logger,
                dry_run=args.dry_run,
            )
    
    # Phase 3: Generate combined analysis
    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 3: GENERATING COMBINED ANALYSIS")
    logger.info("=" * 60)
    log_memory_usage(logger, "before combined analysis")
    
    success = generate_combined_analysis(
        metadata=metadata,
        output_dir=COMPARING_EXPERIMENTS_DIR,
        runs_dir=runs_dir,
        logger=logger,
        dry_run=args.dry_run,
    )
    
    # Final garbage collection after analysis
    import gc
    gc.collect()
    log_memory_usage(logger, "after combined analysis")
    
    # Save final CSV summary
    if not args.dry_run:
        csv_path = save_runs_summary_csv(metadata, logger)
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    
    completed = sum(1 for info in metadata.get("experiments", {}).values() 
                   if info.get("status") == "completed")
    failed = sum(1 for info in metadata.get("experiments", {}).values() 
                if info.get("status") == "failed")
    
    logger.info(f"Experiments completed: {completed}")
    logger.info(f"Experiments failed: {failed}")
    logger.info(f"Combined analysis: {'SUCCESS' if success else 'FAILED'}")
    logger.info(f"Output directory: {COMPARING_EXPERIMENTS_DIR}")
    logger.info(f"Metadata file: {metadata_path}")
    if not args.dry_run:
        logger.info(f"Runs summary CSV: {csv_path}")
        logger.info(f"Runs summary dir: {RUNS_SUMMARY_DIR}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
