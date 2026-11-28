"""Portable fresh-batch lifecycle tests, with temporary artifacts and no MLflow calls."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import signal
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.features import FEATURE_COLUMNS, TARGET_COLUMN
from src.monitoring import alert_manager as alert_module
from src.monitoring.database import (
    get_alert_history, get_connection, get_monitoring_runs, get_monitoring_stats,
    get_store_id, init_db, log_feedback, log_monitoring_run, log_prediction,
)
from src.monitoring.demo_controller import (
    DemoController, install_shutdown_handlers, read_scenario, run_loop, set_scenario,
)
from src.monitoring.metrics import MetricsContainer
from src.monitoring.monitor_orchestrator import run_monitoring_cycle
from src.monitoring.performance_monitor import evaluate_model_performance


class ReplayModel:
    """Fixed analytic predictor; no fitting, credentials, or external artifacts."""
    n_features_in_ = 6

    def predict(self, frame):
        return frame["energy_current_hour"].to_numpy()


@pytest.fixture
def demo_files(tmp_path):
    rng = np.random.default_rng(7)
    n = 1008  # Six complete weeks; avoid calendar composition shifts in the baseline.
    energy = rng.gamma(1.5, 0.08, n)
    hour = np.arange(n) % 24
    day = (np.arange(n) // 24) % 7
    data = pd.DataFrame({
        "energy_current_hour": energy, "energy_previous_hour": energy * 0.95,
        "energy_two_hours_ago": energy * 0.9, "target_hour": hour,
        "target_day_of_week": day, "target_is_weekend": (day >= 5).astype(int),
        TARGET_COLUMN: energy + 0.01,
    })
    reference, test, model = [tmp_path / name for name in ("train.csv", "test.csv", "model.pkl")]
    data.to_csv(reference, index=False)
    data.to_csv(test, index=False)
    joblib.dump(ReplayModel(), model)
    return dict(reference=reference, test=test, model_path=model,
                db_path=tmp_path / "demo.db", output_dir=tmp_path / "output", samples=168)


def test_repeated_lifecycle_retains_history_and_excludes_degraded_batches(demo_files):
    controller = DemoController(**demo_files)
    before = {name: Path(demo_files[name]).read_bytes() for name in ("reference", "test", "model_path")}
    batches = []
    for _ in range(2):
        normal = controller.run_cycle("normal")
        assert normal["active_alerts_count"] == 0
        assert normal["performance_result"]["status"] == "healthy"
        hvac = controller.run_cycle("hvac_shift")
        assert hvac["drift_result"]["dataset_drift_detected"]
        assert hvac["performance_result"]["status"] == "healthy"
        sensor = controller.run_cycle("sensor_drift")
        assert sensor["drift_result"]["dataset_drift_detected"]
        degraded = controller.run_cycle("model_degradation")
        assert degraded["performance_result"]["status"] == "degraded"
        assert not degraded["drift_result"]["dataset_drift_detected"]
        repeated = controller.run_cycle("model_degradation")
        assert repeated["alerts_suppressed"]
        assert repeated["active_alerts_count"] == 1
        assert repeated["active_alerts"][0]["trigger_count"] == 2
        recovery = controller.run_cycle("recovery")
        assert recovery["performance_result"]["status"] == "healthy"
        assert recovery["performance_result"]["evaluated_predictions"] == 168
        assert recovery["active_alerts_count"] == 0
        assert any(a["alert_type"] == "model_degradation" for a in recovery["alerts_resolved"])
        batches.extend([normal, hvac, sensor, degraded, repeated, recovery])
    assert all(a["status"] == "resolved" and a["resolved_at"] for a in get_alert_history(db_path=str(controller.db_path)))
    assert len(get_alert_history(db_path=str(controller.db_path))) == 4
    assert len(get_monitoring_runs(db_path=str(controller.db_path))) == 12
    paths = [b["data_source"] for b in batches]
    assert len(set(paths)) == 12
    # Fresh seeds and observations; not repeated evaluation of a stale CSV.
    assert Path(paths[0]).read_bytes() != Path(paths[1]).read_bytes()
    with get_connection(str(controller.db_path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 12 * 168
        assert conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 12 * 168
        row = conn.execute("SELECT scenario, details_json FROM monitoring_runs ORDER BY rowid DESC LIMIT 1").fetchone()
        assert row[0] == "recovery"
        assert json.loads(row[1])["evaluation_strategy"] == "explicit_batch"
    for name, content in before.items():
        assert Path(demo_files[name]).read_bytes() == content


def test_batch_filter_model_isolation_and_insufficient_feedback_protection(demo_files):
    controller = DemoController(**demo_files)
    degraded = controller.run_cycle("model_degradation")
    assert degraded["active_alerts_count"] == 1
    empty = run_monitoring_cycle(db_path=str(controller.db_path), model_version="ReplayModel",
                                 prediction_ids=[], generate_drift_reports=False)
    assert empty["performance_result"]["status"] == "no_feedback_data"
    assert empty["active_alerts_count"] == 1
    degraded_provenance = json.loads((Path(degraded["data_source"]).parent / "provenance.json").read_text())
    insufficient = run_monitoring_cycle(
        db_path=str(controller.db_path), model_version="ReplayModel",
        prediction_ids=degraded_provenance["prediction_ids"][:10], generate_drift_reports=False,
    )
    assert insufficient["performance_result"]["status"] == "insufficient_data"
    assert insufficient["active_alerts_count"] == 1
    healthy = controller.run_cycle("recovery")
    provenance = json.loads((Path(healthy["data_source"]).parent / "provenance.json").read_text())
    ids = provenance["prediction_ids"]
    wrong_model = evaluate_model_performance(db_path=str(controller.db_path), model_version="another-version", prediction_ids=ids)
    assert wrong_model["evaluated_predictions"] == 0
    small = evaluate_model_performance(db_path=str(controller.db_path), model_version="ReplayModel", prediction_ids=ids[:10])
    assert small["status"] == "insufficient_data"
    assert small["metrics"]["sample_count"] == 10


def test_api_and_collector_share_database_and_scenario_updates(demo_files, monkeypatch):
    import api.main as api
    monkeypatch.setenv("MONITORING_DB_PATH", str(demo_files["db_path"]))
    monkeypatch.setattr(api, "load_model", lambda **kwargs: ReplayModel())
    monkeypatch.setattr(api, "default_metrics", MetricsContainer(db_path=str(demo_files["db_path"])))
    with TestClient(api.app) as client:
        def request(self, endpoint, payload=None):
            response = client.get(endpoint) if payload is None else client.post(endpoint, json=payload)
            response.raise_for_status()
            return response.content, response.headers
        monkeypatch.setattr(DemoController, "request", request)
        options = dict(demo_files, samples=40)
        controller = DemoController(**options, api_url="http://testserver")
        degraded = controller.run_cycle("model_degradation")
        assert degraded["performance_result"]["status"] == "degraded"
        payload = client.get("/metrics")
        assert payload.headers["X-Monitoring-Store-ID"] == get_store_id(str(demo_files["db_path"]))
        assert 'energy_advisor_model_bias_kwh{model_version="local"}' in payload.text
        assert 'energy_advisor_monitoring_scenario_info{scenario="model_degradation"} 1.0' in payload.text
        assert 'energy_advisor_active_alerts_total 1.0' in payload.text
        controller.run_cycle("recovery")
        payload = client.get("/metrics").text
        assert 'energy_advisor_monitoring_scenario_info{scenario="recovery"} 1.0' in payload
        assert 'energy_advisor_model_r2{model_version="local"}' in payload
        assert 'energy_advisor_active_alerts_total 0.0' in payload


def test_mismatched_api_store_rejected_before_any_post(demo_files, monkeypatch):
    calls = []
    def request(self, endpoint, payload=None):
        calls.append((endpoint, payload))
        return b"", {"X-Monitoring-Store-ID": "different-store"}
    monkeypatch.setattr(DemoController, "request", request)
    with pytest.raises(ValueError, match="do not share"):
        DemoController(**demo_files, api_url="http://testserver")
    assert calls == [("/metrics", None)]


def test_existing_non_demo_database_rejected(demo_files):
    db = str(demo_files["db_path"])
    init_db(db)
    features = pd.read_csv(demo_files["test"])[FEATURE_COLUMNS].iloc[0].to_dict()
    log_prediction("real-prediction", features, 0.1, db_path=db)
    with pytest.raises(ValueError, match="not owned"):
        DemoController(**demo_files)
    with get_connection(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 1


def test_metrics_do_not_reuse_old_accuracy_after_no_feedback(demo_files):
    controller = DemoController(**demo_files)
    controller.run_cycle("normal")
    assert get_monitoring_stats(str(controller.db_path))["latest_mae"]
    run_monitoring_cycle(db_path=str(controller.db_path), model_version="ReplayModel", prediction_ids=[])
    assert not get_monitoring_stats(str(controller.db_path))["latest_mae"]
    payload = MetricsContainer(db_path=str(controller.db_path)).generate_payload().decode()
    assert "energy_advisor_model_mae" not in payload


def test_cooldown_is_measured_since_last_notification_not_last_detection(demo_files, monkeypatch):
    current = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return current[0]
    monkeypatch.setattr(alert_module, "datetime", Clock)
    controller = DemoController(**demo_files, cooldown=10)
    drift = dict(dataset_drift_detected=True, drift_share=0.5, drifted_features=FEATURE_COLUMNS[:3], total_features_count=6)
    first, notification = controller.manager.evaluate_drift_result(drift)
    assert notification.event_type == "triggered"
    current[0] += timedelta(seconds=6)
    second, notification = controller.manager.evaluate_drift_result(drift)
    assert notification.event_type == "suppressed"
    assert second.last_notified_at == first.updated_at
    current[0] += timedelta(seconds=5)
    third, notification = controller.manager.evaluate_drift_result(drift)
    assert notification.event_type == "triggered"
    assert third.trigger_count == 3
    assert third.alert_id == first.alert_id


def test_manual_loop_rereads_scenario_and_shutdown_interrupts_interval(tmp_path):
    path = tmp_path / "state.json"
    set_scenario(path, "normal")
    stop = threading.Event()
    observed = []
    class Controller:
        def run_cycle(self, scenario):
            observed.append(scenario)
            if len(observed) == 1:
                set_scenario(path, "hvac_shift")
            else:
                stop.set()
            return {"scenario": scenario}
        def record_failure(self, error):
            raise AssertionError(error)
    assert run_loop(Controller(), path, interval=0, stop_event=stop) == 2
    assert observed == ["normal", "hvac_shift"]
    # A stop requested during a cycle skips even a one-hour wait immediately.
    stop.clear()
    class Stopping(Controller):
        def run_cycle(self, scenario):
            stop.set()
            return {}
    assert run_loop(Stopping(), path, interval=3600, stop_event=stop) == 1


def test_signal_handlers_request_graceful_shutdown(monkeypatch):
    handlers = {}
    monkeypatch.setattr(signal, "signal", lambda name, handler: handlers.setdefault(name, handler))
    stop = threading.Event()
    install_shutdown_handlers(stop)
    handlers[signal.SIGTERM](signal.SIGTERM, None)
    assert stop.is_set()


def test_ingestion_failure_is_not_reported_as_statistical_drift(demo_files):
    controller = DemoController(**demo_files)
    result = controller.record_failure(RuntimeError("input feed unavailable"))
    assert result["alert"]["alert_type"] == "monitoring_execution_failure"
    stats = get_monitoring_stats(str(controller.db_path))
    assert stats["latest_drift_share"] is None
    assert stats["execution_failures"]["demo_ingestion"] == 1
    recovered = controller.run_cycle("recovery")
    assert recovered["ingestion_failure_recovered"]
    assert recovered["active_alerts_count"] == 0


def test_large_batch_filter_has_no_sql_parameter_limit(demo_files):
    controller = DemoController(**demo_files)
    result = evaluate_model_performance(
        db_path=str(controller.db_path), prediction_ids=[f"missing-{i}" for i in range(40000)]
    )
    assert result["evaluated_predictions"] == 0


def test_scenario_generation_is_reproducible_without_reusing_batches(demo_files, tmp_path):
    first = DemoController(**demo_files).run_cycle("normal")
    options = dict(demo_files, db_path=tmp_path / "second.db", output_dir=tmp_path / "second")
    second = DemoController(**options).run_cycle("normal")
    columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    pd.testing.assert_frame_equal(pd.read_csv(first["data_source"])[columns], pd.read_csv(second["data_source"])[columns])
    assert first["performance_result"]["metrics"] == second["performance_result"]["metrics"]
    assert first["batch_id"] != second["batch_id"]


def test_demo_sqlite_tuning_is_explicitly_opt_in(tmp_path, monkeypatch):
    db = str(tmp_path / "durability.db")
    monkeypatch.delenv("MONITORING_SQLITE_SYNCHRONOUS", raising=False)
    init_db(db)
    with get_connection(db) as conn:
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL
    monkeypatch.setenv("MONITORING_SQLITE_SYNCHRONOUS", "NORMAL")
    with get_connection(db) as conn:
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1


def test_both_alerts_resolve_on_final_recovery_when_input_shift_is_retained(demo_files):
    controller = DemoController(**demo_files, degradation_input="sensor_drift")
    degraded = controller.run_cycle("model_degradation")
    assert degraded["active_alerts_count"] == 2
    assert {a["alert_type"] for a in degraded["active_alerts"]} == {"data_drift", "model_degradation"}
    repeated = controller.run_cycle("model_degradation")
    assert len(repeated["alerts_suppressed"]) == 2
    recovery = controller.run_cycle("recovery")
    assert recovery["active_alerts_count"] == 0
    assert {a["alert_type"] for a in recovery["alerts_resolved"]} == {"data_drift", "model_degradation"}


def test_api_transport_reuses_connection_and_does_not_retry_failed_posts(demo_files):
    ports, posts = [], []
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def do_GET(self):
            ports.append(self.client_address[1])
            body = b"{}"
            self.send_response(200)
            self.send_header("X-Monitoring-Store-ID", get_store_id(str(demo_files["db_path"])))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_POST(self):
            posts.append(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.end_headers()
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    controller = None
    try:
        controller = DemoController(**demo_files, api_url=f"http://127.0.0.1:{server.server_port}")
        controller.request("/metrics")
        assert len(ports) == 2 and len(set(ports)) == 1
        # Simulate a keep-alive closed during the interval between batches.
        controller._connection.sock.shutdown(socket.SHUT_RDWR)
        with pytest.raises(HTTPError):
            features = pd.read_csv(demo_files["test"])[FEATURE_COLUMNS].iloc[0].to_dict()
            controller.ingest([features], [0.1])
        assert len(ports) == 3 and ports[2] != ports[0]
        assert len(posts) == 1
        assert controller._connection is None
    finally:
        if controller:
            controller.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
