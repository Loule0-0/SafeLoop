from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


_AUTO_IMAGEIO = object()


@dataclass(frozen=True)
class VideoWriteResult:
    path: Path
    frame_count: int
    mode: str


def write_frames(
    frames: Iterable[np.ndarray],
    output_path: Path,
    fps: int = 10,
    imageio_module: object = _AUTO_IMAGEIO,
) -> VideoWriteResult:
    output_path = Path(output_path)
    frame_list = [_as_rgb_uint8(frame) for frame in frames]
    if not frame_list:
        raise ValueError("at least one frame is required")

    if imageio_module is _AUTO_IMAGEIO:
        try:
            import imageio.v2 as imageio_module  # type: ignore[no-redef]
        except ModuleNotFoundError:
            imageio_module = None

    if imageio_module is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio_module.mimwrite(str(output_path), frame_list, fps=fps)
        return VideoWriteResult(path=output_path, frame_count=len(frame_list), mode="video")

    frames_dir = output_path.with_name(f"{output_path.stem}_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frame_list):
        _write_png(frames_dir / f"frame_{index:06d}.png", frame)
    return VideoWriteResult(path=frames_dir, frame_count=len(frame_list), mode="png_sequence")


def _as_rgb_uint8(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("frames must have shape HxWx3")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _write_png(path: Path, frame: np.ndarray) -> None:
    height, width, _ = frame.shape
    raw_rows = b"".join(b"\x00" + frame[row].tobytes() for row in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw_rows))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)
