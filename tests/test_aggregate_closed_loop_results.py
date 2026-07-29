import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.aggregate_closed_loop_results import build_report, load_rows, summarize


def write_summary(
    root: Path,
    name: str,
    task: int,
    seed: int,
    success: float,
    events: int,
    steps: int,
    stuck_events: int = 0,
    stuck_steps: int = 0,
) -> None:
    out = root / name
    out.mkdir(parents=True)
    payload = {
        "task_id": task,
        "seed": seed,
        "summary": {
            "success_rate": success,
            "mean_steps": 100,
            "mean_effective_control_steps": 110,
            "hazard_events": {"body": 0, "object": events, "stuck": stuck_events, "any": events},
            "hazard_steps": {"body": 0, "object": steps, "stuck": stuck_steps, "any": steps},
            "events_per_1k": events * 10.0,
            "rollback_planned": 1,
            "rollback_failed": 0,
            "rollback_rendered_frames": 5,
        },
    }
    (out / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


class AggregateClosedLoopResultsTests(unittest.TestCase):
    maxDiff = None

    def test_load_rows_reads_nested_hazard_metrics(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_summary(root, "candidate_task5_seed60", 5, 60, 1.0, events=2, steps=12)
            rows = load_rows(root, "candidate")
            self.assertEqual(rows[0]["hazard_events_any"], 2)
            self.assertEqual(rows[0]["hazard_steps_any"], 12)
            self.assertEqual(rows[0]["hazard_events_stuck"], 0)
            self.assertEqual(rows[0]["hazard_steps_stuck"], 0)
            self.assertEqual(rows[0]["rollback_rendered_frames"], 5)

    def test_load_rows_reads_stuck_hazard_metrics(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_summary(
                root,
                "candidate_task6_seed61",
                6,
                61,
                1.0,
                events=4,
                steps=18,
                stuck_events=2,
                stuck_steps=7,
            )
            rows = load_rows(root, "candidate")
            summary = summarize(rows)
            self.assertEqual(rows[0]["hazard_events_stuck"], 2)
            self.assertEqual(rows[0]["hazard_steps_stuck"], 7)
            self.assertEqual(summary["hazard_events_stuck_sum"], 2)
            self.assertEqual(summary["hazard_steps_stuck_sum"], 7)

    def test_build_report_compares_matched_task_seed_pairs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            candidate = root / "candidate"
            write_summary(base, "base_task5_seed60", 5, 60, 1.0, events=3, steps=30)
            write_summary(candidate, "candidate_task5_seed60", 5, 60, 1.0, events=1, steps=10)
            report = build_report([("base", base), ("candidate", candidate)], base_label="base")
            self.assertEqual(summarize(load_rows(candidate, "candidate"))["hazard_steps_any_sum"], 10)
            comparison = report["comparisons"]["candidate"]
            self.assertEqual(comparison["deltas"]["hazard_steps_any_delta"], -20)
            self.assertEqual(comparison["pairs"][0]["task"], 5)

    def test_load_rows_expands_multi_task_runner_summary(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "libero_10_tasks_0_3_seed7_rl"
            out.mkdir()
            payload = {
                "benchmark": "libero_10",
                "task_id": None,
                "task_ids": [0, 3],
                "seed": 7,
                "summary": {"success_rate": 0.5},
                "tasks": [
                    {
                        "task_id": 0,
                        "summary": {
                            "success_rate": 1.0,
                            "hazard_events": {"any": 1},
                            "hazard_steps": {"any": 2},
                        },
                    },
                    {
                        "task_id": 3,
                        "summary": {
                            "success_rate": 0.0,
                            "hazard_events": {"any": 2},
                            "hazard_steps": {"any": 4},
                        },
                    },
                ],
            }
            (out / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
            rows = load_rows(root, "candidate")
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                [(row["benchmark"], row["task"], row["seed"]) for row in rows],
                [("libero_10", 0, 7), ("libero_10", 3, 7)],
            )
            self.assertEqual(summarize(rows)["hazard_events_any_sum"], 3)

    def test_manual_placeholder_hazards_are_not_aggregated(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_summary(root, "candidate_task5_seed60", 5, 60, 1.0, events=0, steps=0)
            path = root / "candidate_task5_seed60" / "summary.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["summary"]["manual_hazard_review_required"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")

            rows = load_rows(root, "candidate")
            self.assertIsNone(rows[0]["hazard_events_any"])
            summary = summarize(rows)
            self.assertFalse(summary["hazard_metrics_available"])
            self.assertIsNone(summary["hazard_events_any_sum"])


if __name__ == "__main__":
    unittest.main()
