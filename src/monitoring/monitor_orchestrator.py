"""
Monitoring Orchestration Engine for Smart Home Energy Advisor (Phase 3.5).

Coordinates periodic or on-demand execution of:
1. Data Drift Detection (src/monitoring/drift_detector.py)
2. Model Performance Monitoring (src/monitoring/performance_monitor.py)
3. Alert Lifecycle Evaluation & Dispatch (src/monitoring/alert_manager.py)

Operates out-of-band from FastAPI inference requests. Can be invoked via CLI,
cron, Windows Task Scheduler, or a configurable Python interval loop.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import pandas as pd

# Ensure project root is in path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.monitoring.alert_manager import AlertManager, AlertType, NotificationMessage
from src.monitoring.database import get_db_path, init_db
from src.monitoring.drift_detector import (
    DEFAULT_REF_PATH,
    run_drift_analysis,
)
from src.monitoring.performance_monitor import (
    evaluate_model_performance,
    generate_performance_report,
)

logger = logging.getLogger("monitoring.orchestrator")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def run_monitoring_cycle(
    ref_data: Optional[Union[str, pd.DataFrame]] = None,
    prod_data: Optional[Union[str, pd.DataFrame]] = None,
    db_path: Optional[str] = None,
    model_version: Optional[str] = None,
    alert_manager: Optional[AlertManager] = None,
    cooldown_seconds: Optional[int] = None,
    save_reports: bool = False,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a single monitoring cycle combining drift, performance, and alerting.

    Args:
        ref_data: Path or DataFrame for historical baseline data (defaults to data/train.csv).
        prod_data: Path or DataFrame for recent production observations.
        db_path: Path to SQLite monitoring database.
        model_version: Optional model version filter for performance monitoring.
        alert_manager: Optional AlertManager instance (creates one if None).
        cooldown_seconds: Optional cooldown period for duplicate alert suppression.
        save_reports: Whether to persist JSON/HTML reports to disk.
        output_dir: Directory for persisting monitoring reports.

    Returns:
        Structured summary dictionary of the monitoring cycle.
    """
    cycle_id = f"cycle_{uuid.uuid4().hex[:8]}"
    start_time = datetime.now(timezone.utc).isoformat()
    logger.info("Starting monitoring cycle '%s' at %s", cycle_id, start_time)

    active_db = db_path or get_db_path()
    init_db(active_db)
    mgr = alert_manager or AlertManager(db_path=active_db)

    run_summary: Dict[str, Any] = {
        "cycle_id": cycle_id,
        "timestamp": start_time,
        "database_path": active_db,
        "model_version": model_version or "all_models",
        "drift_status": "skipped",
        "drift_result": None,
        "performance_status": "skipped",
        "performance_result": None,
        "alerts_triggered": [],
        "alerts_suppressed": [],
        "alerts_resolved": [],
        "execution_failures": [],
        "notifications": [],
    }

    # ─── 1. Run Data Drift Detection ──────────────────────────────────
    ref_path = ref_data or DEFAULT_REF_PATH
    if prod_data is not None:
        try:
            logger.info("Evaluating data drift against reference: %s", ref_path)
            drift_summary = run_drift_analysis(
                reference_data=ref_path,
                production_data=prod_data,
            )
            run_summary["drift_status"] = "evaluated"
            run_summary["drift_result"] = drift_summary

            # Auto-resolve any prior drift_detection execution failure
            fail_resolved = mgr.resolve_execution_failure("drift_detection")
            if fail_resolved:
                res_alert, res_notif = fail_resolved
                run_summary["alerts_resolved"].append(res_alert.to_dict())
                run_summary["notifications"].append(res_notif.to_dict())

            # Evaluate drift alert rules
            alert, notif = mgr.evaluate_drift_result(
                drift_result=drift_summary,
                cooldown_seconds=cooldown_seconds,
            )
            if alert:
                if notif and notif.event_type == "suppressed":
                    run_summary["alerts_suppressed"].append(alert.to_dict())
                elif notif and notif.event_type == "resolved":
                    run_summary["alerts_resolved"].append(alert.to_dict())
                else:
                    run_summary["alerts_triggered"].append(alert.to_dict())
                if notif:
                    run_summary["notifications"].append(notif.to_dict())

        except Exception as e:
            logger.error("Drift detection execution failed: %s", e, exc_info=True)
            run_summary["drift_status"] = "failed"
            fail_alert, fail_notif = mgr.record_execution_failure(
                operation_name="drift_detection",
                error=e,
                model_version=model_version,
            )
            run_summary["execution_failures"].append(fail_alert.to_dict())
            run_summary["notifications"].append(fail_notif.to_dict())
    else:
        logger.info("No production dataset provided; skipping drift detection in this cycle.")

    # ─── 2. Run Model Performance Monitoring ──────────────────────────
    try:
        logger.info("Evaluating model performance for version: %s", model_version or "all")
        perf_summary = evaluate_model_performance(
            db_path=active_db,
            model_version=model_version,
        )
        run_summary["performance_status"] = "evaluated"
        run_summary["performance_result"] = perf_summary

        # Auto-resolve any prior performance_monitoring execution failure
        fail_resolved = mgr.resolve_execution_failure("performance_monitoring")
        if fail_resolved:
            res_alert, res_notif = fail_resolved
            run_summary["alerts_resolved"].append(res_alert.to_dict())
            run_summary["notifications"].append(res_notif.to_dict())

        # Evaluate performance alert rules
        alert, notif = mgr.evaluate_performance_result(
            perf_result=perf_summary,
            model_version=model_version,
            cooldown_seconds=cooldown_seconds,
        )
        if alert:
            if notif and notif.event_type == "suppressed":
                run_summary["alerts_suppressed"].append(alert.to_dict())
            elif notif and notif.event_type == "resolved":
                run_summary["alerts_resolved"].append(alert.to_dict())
            else:
                run_summary["alerts_triggered"].append(alert.to_dict())
            if notif:
                run_summary["notifications"].append(notif.to_dict())

        if save_reports:
            out_dir = output_dir or os.path.join(_PROJECT_ROOT, "reports", "performance")
            generate_performance_report(
                db_path=active_db,
                output_dir=out_dir,
                model_version=model_version,
            )

    except Exception as e:
        logger.error("Performance monitoring execution failed: %s", e, exc_info=True)
        run_summary["performance_status"] = "failed"
        fail_alert, fail_notif = mgr.record_execution_failure(
            operation_name="performance_monitoring",
            error=e,
            model_version=model_version,
        )
        run_summary["execution_failures"].append(fail_alert.to_dict())
        run_summary["notifications"].append(fail_notif.to_dict())

    # Count currently active alerts in the database
    run_summary["active_alerts_count"] = len(mgr.get_active_alerts())
    run_summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    logger.info(
        "Monitoring cycle '%s' complete. Triggered: %d, Suppressed: %d, Resolved: %d, Active: %d",
        cycle_id,
        len(run_summary["alerts_triggered"]),
        len(run_summary["alerts_suppressed"]),
        len(run_summary["alerts_resolved"]),
        run_summary["active_alerts_count"],
    )
    return run_summary


# ─── CLI Entrypoint ───────────────────────────────────────────────────

def build_cli_parser() -> argparse.ArgumentParser:
    """Build command-line parser for monitoring orchestrator."""
    parser = argparse.ArgumentParser(
        description="Automated Monitoring and Alerting Orchestration Engine.",
    )
    parser.add_argument(
        "--ref",
        dest="ref_data",
        default=DEFAULT_REF_PATH,
        help="Path to historical reference dataset (defaults to data/train.csv).",
    )
    parser.add_argument(
        "--prod",
        dest="prod_data",
        default=None,
        help="Path to production observation dataset (e.g. data/simulated/normal_scenario.csv).",
    )
    parser.add_argument(
        "--db",
        dest="db_path",
        default=None,
        help="Path to SQLite monitoring database.",
    )
    parser.add_argument(
        "--model-version",
        dest="model_version",
        default=None,
        help="Model version identifier to evaluate (e.g. GradientBoostingRegressor).",
    )
    parser.add_argument(
        "--cooldown",
        dest="cooldown_seconds",
        type=int,
        default=DEFAULT_COOLDOWN_SECONDS,
        help="Cooldown period in seconds for duplicate alert suppression.",
    )
    parser.add_argument(
        "--save-reports",
        action="store_true",
        help="Persist performance and drift reports to reports/ directory.",
    )
    parser.add_argument(
        "--interval-seconds",
        dest="interval_seconds",
        type=int,
        default=0,
        help="If > 0, runs continuously as a scheduler polling every N seconds. If 0, runs once and exits.",
    )
    return parser


def main() -> None:
    """CLI execution entrypoint."""
    parser = build_cli_parser()
    args = parser.parse_args()

    if args.interval_seconds > 0:
        logger.info("Starting scheduler loop with interval: %d seconds", args.interval_seconds)
        try:
            while True:
                summary = run_monitoring_cycle(
                    ref_data=args.ref_data,
                    prod_data=args.prod_data,
                    db_path=args.db_path,
                    model_version=args.model_version,
                    cooldown_seconds=args.cooldown_seconds,
                    save_reports=args.save_reports,
                )
                print(json.dumps({
                    "cycle_id": summary["cycle_id"],
                    "timestamp": summary["timestamp"],
                    "drift_status": summary["drift_status"],
                    "performance_status": summary["performance_status"],
                    "active_alerts": summary["active_alerts_count"],
                }, indent=2))
                time.sleep(args.interval_seconds)
        except KeyboardInterrupt:
            logger.info("Scheduler loop terminated by user.")
    else:
        summary = run_monitoring_cycle(
            ref_data=args.ref_data,
            prod_data=args.prod_data,
            db_path=args.db_path,
            model_version=args.model_version,
            cooldown_seconds=args.cooldown_seconds,
            save_reports=args.save_reports,
        )
        print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
