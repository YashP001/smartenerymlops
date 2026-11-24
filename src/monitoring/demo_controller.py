"""Fresh-batch monitoring demonstration; never trains or registers a model.

Input drift uses the existing simulator. Accuracy uses separately replayed held-out
targets so a covariate shift is never presented as proof of performance degradation.
"""

import hashlib
from http.client import HTTPConnection, HTTPSConnection
import json
import math
import os
from pathlib import Path
import signal
import threading
import time
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.parse import urlsplit
import uuid

import pandas as pd

from src.features import FEATURE_COLUMNS, TARGET_COLUMN, validate_inference_input
from src.predict import load_model, predict
from src.simulation import generate_simulated_data, validate_simulated_dataset
from src.monitoring.alert_manager import AlertManager
from src.monitoring.database import (
    DEFAULT_DB_PATH, get_connection, get_store_id, init_db,
    log_feedback, log_monitoring_execution_failure, log_monitoring_run, log_prediction,
)
from src.monitoring.monitor_orchestrator import run_monitoring_cycle
from src.monitoring.performance_monitor import DEFAULT_MIN_SAMPLES

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEMO_DIR = ROOT / "reports" / "demo" / "continuous"
SCENARIOS = ("normal", "hvac_shift", "sensor_drift", "model_degradation", "recovery")
AUTO_SEQUENCE = ("normal", "hvac_shift", "sensor_drift", "model_degradation", "recovery")


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def set_scenario(path, scenario):
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    write_json_atomic(path, {"scenario": scenario, "updated_at": datetime.now(timezone.utc).isoformat()})


def read_scenario(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))["scenario"]
    if value not in SCENARIOS:
        raise ValueError(f"Unknown scenario in state file: {value}")
    return value


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DemoController:
    def __init__(
        self, *, reference=ROOT / "data/train.csv", test=ROOT / "data/test.csv",
        db_path=DEFAULT_DEMO_DIR / "predictions.db", output_dir=DEFAULT_DEMO_DIR,
        samples=500, seed=42, severity=1.2, model_path=ROOT / "models/gradient_boosting.pkl",
        api_url=None, cooldown=3600, reports=False, degradation_input="normal",
    ):
        if samples < DEFAULT_MIN_SAMPLES:
            raise ValueError(f"At least {DEFAULT_MIN_SAMPLES} samples are required")
        if not math.isfinite(severity) or severity < 0:
            raise ValueError("Severity must be finite and nonnegative")
        if degradation_input not in ("normal", "hvac_shift", "sensor_drift"):
            raise ValueError("Degradation input must be normal, hvac_shift, or sensor_drift")
        self.degradation_input = degradation_input
        self.reference, self.test = Path(reference).resolve(), Path(test).resolve()
        self.db_path, self.output_dir = Path(db_path).resolve(), Path(output_dir).resolve()
        self.model_path = Path(model_path).resolve()
        if self.db_path == Path(DEFAULT_DB_PATH).resolve():
            raise ValueError("Use a dedicated demonstration database, not the development database")
        if self.db_path in (self.reference, self.test, self.model_path):
            raise ValueError("Database cannot overwrite a historical dataset or model")
        self.ref_df = pd.read_csv(self.reference)
        self.test_df = pd.read_csv(self.test)
        for frame in (self.ref_df, self.test_df):
            if not set(FEATURE_COLUMNS + [TARGET_COLUMN]).issubset(frame.columns):
                raise ValueError("Reference and test data must contain all six features and the target")
        if samples > len(self.test_df):
            raise ValueError("Sample count exceeds held-out rows; do not duplicate evaluation observations")
        self.samples, self.seed, self.severity = samples, seed, severity
        self.api_url = api_url.rstrip("/") if api_url else None
        self._connection = None
        self.reports, self.cooldown = reports, cooldown
        self.cycle_number = 0
        self.session_id = uuid.uuid4().hex[:12]
        init_db(str(self.db_path))
        with get_connection(str(self.db_path)) as conn:
            purpose = conn.execute("SELECT value FROM monitoring_metadata WHERE key='purpose'").fetchone()
            count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            if (purpose and purpose[0] != "continuous_monitoring_demo") or (count and not purpose):
                raise ValueError("Refusing to append demo data to a database not owned by this controller")
            conn.execute("INSERT OR IGNORE INTO monitoring_metadata VALUES ('purpose', 'continuous_monitoring_demo')")
            conn.commit()
        self.manager = AlertManager(db_path=str(self.db_path), default_cooldown_seconds=cooldown)
        self.model = None
        self.model_version = None
        self.active_scenario = None
        if self.api_url:
            # Check before POST: a remote/development API must never receive demo writes.
            self.verify_api_store()
        else:
            self.model = load_model(model_path=str(self.model_path), use_registry=False)
            self.model_version = type(self.model).__name__
        self.checkpoint = {
            "source": "api" if self.api_url else str(self.model_path),
            "local_artifact_sha256": file_sha256(self.model_path) if not self.api_url else None,
            "baseline": "Phase 1 historical checkpoint: MAE 0.0739, RMSE 0.1512, R2 0.3723",
            "note": "The local pickle is the Phase 1 train+validation checkpoint; registry v1 is a distinct train-only checkpoint. API checkpoint identity is not exposed by the existing API.",
        }

    def request(self, endpoint, payload=None):
        body = json.dumps(payload).encode() if payload is not None else None
        url = urlsplit(self.api_url)
        if url.scheme not in ("http", "https") or not url.hostname:
            raise ValueError("API URL must use http or https and include a hostname")
        if self._connection is None:
            connection_type = HTTPSConnection if url.scheme == "https" else HTTPConnection
            self._connection = connection_type(url.hostname, url.port, timeout=30)
        try:
            self._connection.request("POST" if body is not None else "GET", url.path + endpoint,
                                     body=body, headers={"Content-Type": "application/json"})
            response = self._connection.getresponse()
            data, headers = response.read(), response.headers
            if response.status >= 400:
                raise HTTPError(self.api_url + endpoint, response.status, response.reason, headers, None)
            return data, headers
        except Exception:
            # Never automatically retry a POST: it might already have been persisted.
            self.close()
            raise

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def verify_api_store(self):
        _, headers = self.request("/metrics")
        if headers.get("X-Monitoring-Store-ID") != get_store_id(str(self.db_path)):
            raise ValueError("API and controller do not share the same demonstration database; no predictions sent")

    def ingest(self, rows, actuals):
        ids, versions = [], set()
        if self.api_url:
            # Uvicorn may close idle keep-alives during the monitoring interval.
            # Reuse within a batch, then establish a fresh connection next batch.
            self.close()
            self.verify_api_store()
        for features, actual in zip(rows, actuals):
            validate_inference_input(features)
            if self.api_url:
                raw, _ = self.request("/predict", features)
                result = json.loads(raw)
                prediction_id = result["prediction_id"]
                # Background logging may finish after the inference HTTP response.
                deadline = time.monotonic() + 10
                while True:
                    with get_connection(str(self.db_path)) as conn:
                        record = conn.execute(
                            "SELECT model_version FROM predictions WHERE prediction_id=?", (prediction_id,)
                        ).fetchone()
                    if record:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Prediction was not persisted in the shared demonstration store")
                    time.sleep(0.02)
                versions.add(record[0])
                self.request("/feedback", {"prediction_id": prediction_id, "actual_energy_kwh": float(actual)})
            else:
                prediction_id = str(uuid.uuid4())
                start = time.perf_counter()
                value = predict(self.model, features)
                version = type(self.model).__name__
                log_prediction(
                    prediction_id, features, value, model_version=version,
                    latency_ms=(time.perf_counter() - start) * 1000, db_path=str(self.db_path),
                )
                log_feedback(prediction_id, float(actual), db_path=str(self.db_path))
                versions.add(version)
            ids.append(prediction_id)
        if len(versions) != 1:
            raise RuntimeError("Model changed within a demonstration batch; evaluate versions separately")
        return ids, versions.pop()

    def run_cycle(self, scenario):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        self.cycle_number += 1
        self.active_scenario = scenario
        cycle_seed = self.seed + self.cycle_number - 1
        batch_id = f"{self.session_id}_{self.cycle_number:05d}"
        batch_dir = self.output_dir / "batches" / batch_id
        batch_dir.mkdir(parents=True, exist_ok=False)
        base_scenario = scenario if scenario in ("hvac_shift", "sensor_drift") else "normal"
        if scenario == "model_degradation":
            base_scenario = self.degradation_input
        production = generate_simulated_data(
            scenario=base_scenario, n_samples=self.samples, random_state=cycle_seed,
            severity=self.severity, ref_df=self.ref_df,
        )
        validation = validate_simulated_dataset(production)
        if not validation["passed"]:
            raise ValueError(validation["errors"])
        heldout = self.test_df.sample(n=self.samples, random_state=cycle_seed).copy()
        actuals = heldout[TARGET_COLUMN].to_numpy().copy()
        feedback_source = "genuine held-out test targets (historical replay)"
        offset = 0.0
        if scenario == "model_degradation":
            # Controlled change in P(y|X); no edits to model predictions or model artifact.
            offset = 0.4 * self.severity
            actuals += offset
            production[TARGET_COLUMN] += offset
            feedback_source = "SIMULATED: held-out target + 0.4 * severity kWh; unchanged inputs/model"
        csv_path = batch_dir / "production.csv"
        production.to_csv(csv_path, index=False)
        ids, version = self.ingest(heldout[FEATURE_COLUMNS].to_dict("records"), actuals)
        self.model_version = version
        recovery = self.manager.resolve_execution_failure("demo_ingestion")
        provenance = {
            "batch_id": batch_id, "scenario": scenario, "seed": cycle_seed,
            "input_scenario": base_scenario,
            "samples": self.samples, "severity": self.severity,
            "production_source": str(csv_path), "reference_source": str(self.reference),
            "feedback_source": feedback_source, "test_source": str(self.test),
            "simulated_target_offset_kwh": offset, "prediction_ids": ids,
            "checkpoint": self.checkpoint,
            "evaluation_strategy": "current batch prediction IDs; previous batches retained and excluded",
        }
        write_json_atomic(batch_dir / "provenance.json", provenance)
        result = run_monitoring_cycle(
            ref_data=str(self.reference), prod_data=str(csv_path), db_path=str(self.db_path),
            model_version=version, alert_manager=self.manager, cooldown_seconds=self.cooldown,
            prediction_ids=ids, scenario=scenario, data_source=str(csv_path), batch_id=batch_id,
            output_dir=str(batch_dir), save_reports=self.reports, generate_drift_reports=self.reports,
        )
        result.update({
            "cycle_number": self.cycle_number, "observations": len(production),
            "feedback_source": feedback_source, "checkpoint": self.checkpoint,
            "active_alerts": self.manager.get_active_alerts(),
            "ingestion_failure_recovered": recovery is not None,
        })
        write_json_atomic(batch_dir / "summary.json", result)
        write_json_atomic(self.output_dir / "latest.json", result)
        with (self.output_dir / "cycles.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result) + "\n")
        return result

    def record_failure(self, error):
        alert, notification = self.manager.record_execution_failure("demo_ingestion", error)
        log_monitoring_execution_failure("demo_ingestion", str(error), db_path=str(self.db_path))
        timestamp = datetime.now(timezone.utc).isoformat()
        log_monitoring_run(
            run_id=f"failed_{uuid.uuid4().hex}", timestamp=timestamp,
            drift_status="failed", performance_status="failed", alerts_triggered=1,
            alerts_resolved=0, active_alerts_count=len(self.manager.get_active_alerts()),
            db_path=str(self.db_path), model_version=self.model_version,
            scenario=self.active_scenario, details={"error": str(error)},
        )
        return {"error": str(error), "alert": alert.to_dict(), "notification": notification.to_dict()}


def concise_summary(result):
    drift = result.get("drift_result") or {}
    performance = result.get("performance_result") or {}
    metrics = performance.get("metrics") or {}
    return {
        "scenario": result.get("scenario"), "cycle": result.get("cycle_number"),
        "observations": result.get("observations"), "paired_feedback": performance.get("evaluated_predictions"),
        "drift_share": drift.get("drift_share"), "drifted_features": drift.get("drifted_features"),
        "mae": metrics.get("mae"), "rmse": metrics.get("rmse"), "r2": metrics.get("r2"),
        "bias": metrics.get("bias"), "performance_status": performance.get("status"),
        "active_alerts": result.get("active_alerts_count"),
        "new_or_renotified_alerts": len(result.get("alerts_triggered", [])),
        "new_alerts": sum(a.get("trigger_count", 1) == 1 for a in result.get("alerts_triggered", [])),
        "reminder_notifications": sum(a.get("trigger_count", 1) > 1 for a in result.get("alerts_triggered", [])),
        "suppressed": len(result.get("alerts_suppressed", [])),
        "resolved": len(result.get("alerts_resolved", [])),
    }


def run_loop(controller, state_file, *, interval=10, stop_event=None, cycles=None, automatic=False, hold_cycles=2):
    if not math.isfinite(interval) or interval < 0 or hold_cycles < 1 or (cycles is not None and cycles < 1):
        raise ValueError("Use a finite nonnegative interval and positive cycle counts")
    stop = stop_event if stop_event is not None else threading.Event()
    completed = 0
    while not stop.is_set() and (cycles is None or completed < cycles):
        try:
            scenario = AUTO_SEQUENCE[(completed // hold_cycles) % len(AUTO_SEQUENCE)] if automatic else read_scenario(state_file)
            result = controller.run_cycle(scenario)
            print(json.dumps(concise_summary(result)), flush=True)
        except Exception as error:
            print(json.dumps(controller.record_failure(error)), flush=True)
        completed += 1
        if cycles is None or completed < cycles:
            stop.wait(interval)
    return completed


def install_shutdown_handlers(stop_event):
    previous = {}
    for name in (signal.SIGINT, signal.SIGTERM):
        previous[name] = signal.signal(name, lambda *_: stop_event.set())
    return previous
