"""Manual scenario switching or automatic, independent fresh-batch monitoring."""

import argparse
import json
from pathlib import Path
import signal
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.monitoring.demo_controller import (
    DEFAULT_DEMO_DIR, ROOT, SCENARIOS, DemoController, install_shutdown_handlers,
    read_scenario, run_loop, set_scenario,
)
from src.monitoring.database import get_alert_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    switch = sub.add_parser("set-scenario", help="Switch the next monitoring cycle without stopping the process")
    switch.add_argument("scenario", choices=SCENARIOS)
    switch.add_argument("--state-file", type=Path, default=DEFAULT_DEMO_DIR / "state.json")
    status = sub.add_parser("status")
    status.add_argument("--output-dir", type=Path, default=DEFAULT_DEMO_DIR)
    history = sub.add_parser("history", help="Inspect retained alert IDs, trigger counts, and resolution timestamps")
    history.add_argument("--db", type=Path, default=DEFAULT_DEMO_DIR / "predictions.db")
    run = sub.add_parser("run")
    run.add_argument("--reference", type=Path, default=ROOT / "data/train.csv")
    run.add_argument("--test", type=Path, default=ROOT / "data/test.csv")
    run.add_argument("--db", type=Path, default=DEFAULT_DEMO_DIR / "predictions.db")
    run.add_argument("--output-dir", type=Path, default=DEFAULT_DEMO_DIR)
    run.add_argument("--state-file", type=Path, default=DEFAULT_DEMO_DIR / "state.json")
    run.add_argument("--samples", type=int, default=500)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--severity", type=float, default=1.2)
    run.add_argument("--degradation-input", choices=("normal", "hvac_shift", "sensor_drift"), default="normal",
                     help="Keep a known input shift during degradation to show both alerts resolving at recovery")
    run.add_argument("--model-path", type=Path, default=ROOT / "models/gradient_boosting.pkl")
    run.add_argument("--api-url", help="API must expose the same demonstration store identity before any POST")
    run.add_argument("--interval", type=float, default=10)
    run.add_argument("--cooldown", type=int, default=3600)
    run.add_argument("--scenario", choices=SCENARIOS, default="normal", help="Initial scenario if state file does not exist")
    run.add_argument("--automatic", action="store_true")
    run.add_argument("--hold-cycles", type=int, default=2)
    run.add_argument("--cycles", type=int, default=None, help="Stop after N cycles; otherwise run until Ctrl+C")
    run.add_argument("--reports", action="store_true", help="Generate Evidently HTML when installed, plus performance JSON")
    args = parser.parse_args()
    if args.command == "set-scenario":
        set_scenario(args.state_file, args.scenario)
        print(f"Next cycle: {read_scenario(args.state_file)}")
        return
    if args.command == "status":
        print((args.output_dir / "latest.json").read_text(encoding="utf-8"))
        return
    if args.command == "history":
        print(json.dumps(get_alert_history(db_path=str(args.db)), indent=2))
        return
    controller = DemoController(
        reference=args.reference, test=args.test, db_path=args.db, output_dir=args.output_dir,
        samples=args.samples, seed=args.seed, severity=args.severity, model_path=args.model_path,
        api_url=args.api_url, cooldown=args.cooldown, reports=args.reports,
        degradation_input=args.degradation_input,
    )
    if not args.state_file.exists():
        set_scenario(args.state_file, args.scenario)
    stop = threading.Event()
    previous = install_shutdown_handlers(stop)
    try:
        run_loop(controller, args.state_file, interval=args.interval, stop_event=stop,
                 cycles=args.cycles, automatic=args.automatic, hold_cycles=args.hold_cycles)
    except KeyboardInterrupt:
        stop.set()
    finally:
        controller.close()
        for name, handler in previous.items():
            signal.signal(name, handler)
        print("Monitoring stopped; database, batches, and alert history preserved.", flush=True)


if __name__ == "__main__":
    main()
