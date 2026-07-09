import tempfile
import unittest
from pathlib import Path

import numpy as np

from safety_guard.video import VideoWriteResult, write_frames


class FakeImageIO:
    def __init__(self):
        self.calls = []

    def mimwrite(self, output_path, frames, fps):
        self.calls.append((output_path, frames, fps))


class VideoUtilsTests(unittest.TestCase):
    def test_write_frames_uses_imageio_writer(self):
        fake = FakeImageIO()
        frames = [np.zeros((4, 4, 3), dtype=np.uint8), np.ones((4, 4, 3), dtype=np.uint8)]

        result = write_frames(frames, Path("rollback.mp4"), fps=12, imageio_module=fake)

        self.assertEqual(result, VideoWriteResult(path=Path("rollback.mp4"), frame_count=2, mode="video"))
        self.assertEqual(fake.calls[0][0], "rollback.mp4")
        self.assertEqual(fake.calls[0][2], 12)

    def test_write_frames_falls_back_to_png_sequence_without_imageio(self):
        frames = [np.zeros((3, 3, 3), dtype=np.uint8)]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = write_frames(frames, Path(tmpdir) / "rollback.mp4", imageio_module=None)

            self.assertEqual(result.mode, "png_sequence")
            self.assertEqual(result.frame_count, 1)
            self.assertTrue((Path(tmpdir) / "rollback_frames" / "frame_000000.png").exists())


if __name__ == "__main__":
    unittest.main()
