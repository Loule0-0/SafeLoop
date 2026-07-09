import unittest

from scripts.build_qwen_hardcase_dataset import merge_and_filter_samples


def sample(source, current_object, image="run/images/00010_00.jpg", sample_index=10):
    return {
        "messages": [{"role": "user", "content": f"tau=50 current_global_step={sample_index}"}],
        "images": [image],
        "labels": {
            "future_body": 0.0,
            "future_body_tth": 1.0,
            "future_object": float(current_object),
            "future_object_tth": 0.0 if current_object else 1.0,
            "current_body": 0.0,
            "current_object": float(current_object),
        },
        "metadata": {
            "benchmark": "libero_10",
            "task_id": 8,
            "episode": 0,
            "sample_index": sample_index,
            "source": source,
        },
    }


class QwenHardcaseDatasetTests(unittest.TestCase):
    def test_drops_same_step_rollback_image_collisions_with_conflicting_labels(self):
        samples = [
            sample("online_stride", 1.0),
            sample("rollback_pre", 1.0),
            sample("rollback_post", 0.0),
            sample("online_stride", 0.0, image="run/images/00020_00.jpg", sample_index=20),
        ]

        kept, report = merge_and_filter_samples(samples)

        self.assertEqual(len(kept), 1)
        self.assertEqual(report["collision_groups_removed"], 1)
        self.assertEqual(report["dropped_samples"], 3)

    def test_keeps_fixed_same_step_rollback_samples_with_distinct_image_paths(self):
        samples = [
            sample("rollback_pre", 1.0, image="run/images/00010_rollback_pre_00.jpg"),
            sample("rollback_post", 0.0, image="run/images/00010_rollback_post_00.jpg"),
        ]

        kept, report = merge_and_filter_samples(samples)

        self.assertEqual(len(kept), 2)
        self.assertEqual(report["collision_groups_removed"], 0)

    def test_dedupes_exact_repeated_samples(self):
        samples = [sample("online_stride", 0.0), sample("online_stride", 0.0)]

        kept, report = merge_and_filter_samples(samples)

        self.assertEqual(len(kept), 1)
        self.assertEqual(report["exact_duplicates_removed"], 1)

    def test_same_step_samples_from_different_run_dirs_do_not_collide(self):
        samples = [
            sample("rollback_pre", 1.0, image="run_a/images/00010_rollback_pre_00.jpg"),
            sample("rollback_post", 0.0, image="run_a/images/00010_rollback_post_00.jpg"),
            sample("online_stride", 1.0, image="run_b/images/00010_00.jpg"),
            sample("rollback_pre", 1.0, image="run_b/images/00010_00.jpg"),
            sample("rollback_post", 0.0, image="run_b/images/00010_00.jpg"),
        ]

        kept, report = merge_and_filter_samples(samples)

        self.assertEqual(len(kept), 2)
        self.assertEqual(report["collision_groups_removed"], 1)
        self.assertTrue(all("run_a" in item["images"][0] for item in kept))


if __name__ == "__main__":
    unittest.main()
