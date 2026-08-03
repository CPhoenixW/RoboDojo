import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
from typing import Any

import numpy as np


def format_video_saved_message(
    path: str,
    n_frames: int,
    width: int,
    height: int,
    fps: float,
) -> str:
    return (
        f"🎬 Video is saved to `{path}`, containing "
        f"\033[94m{n_frames}\033[0m frames at {width}×{height} "
        f"resolution and {fps} FPS."
    )


_VIDEO_CODEC: str | None = None


def _resolve_video_codec() -> str:
    """Choose the process-wide video codec, preferring NVENC when available."""
    global _VIDEO_CODEC
    if _VIDEO_CODEC is not None:
        return _VIDEO_CODEC

    requested = os.environ.get("ROBODOJO_VIDEO_CODEC", "auto").strip().lower()
    if requested in {"cpu", "x264", "libx264"}:
        _VIDEO_CODEC = "libx264"
        return _VIDEO_CODEC
    if requested not in {"auto", "nvenc", "h264_nvenc"}:
        raise ValueError(
            "ROBODOJO_VIDEO_CODEC must be one of auto, h264_nvenc, or libx264; "
            f"got {requested!r}"
        )

    nvenc_available = False
    probe_error = ""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        try:
            # An encoder can be listed by ffmpeg even when the driver, GPU
            # visibility, or NVENC build is unusable. Run one tiny real encode
            # so auto mode cannot make the evaluation fail on first frame.
            probe = subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=16x16:d=0.1",
                    "-frames:v",
                    "1",
                    "-an",
                    "-c:v",
                    "h264_nvenc",
                    "-f",
                    "null",
                    "-",
                ],
                check=False,
                capture_output=True,
                timeout=10,
            )
            nvenc_available = probe.returncode == 0
            if not nvenc_available:
                probe_error = probe.stderr.decode(errors="replace").strip().splitlines()[-1] if probe.stderr else "probe failed"
        except (OSError, subprocess.SubprocessError):
            probe_error = "probe failed"

    if requested in {"nvenc", "h264_nvenc"} and not nvenc_available:
        print(
            "[VideoStreamWriter] h264_nvenc is unavailable; falling back to libx264"
            f" ({probe_error}).",
            flush=True,
        )
    _VIDEO_CODEC = "h264_nvenc" if nvenc_available else "libx264"
    print(f"[VideoStreamWriter] selected codec={_VIDEO_CODEC}", flush=True)
    return _VIDEO_CODEC


class VideoStreamWriter:
    """Stream frames to an mp4 through a bounded asynchronous ffmpeg pipe.

    The simulation thread copies a frame into a bounded queue. A worker thread
    feeds ffmpeg, so video encoding does not block normal simulation steps.
    Backpressure is intentional: official videos must keep every frame.
    """

    _SENTINEL = object()

    def __init__(
        self,
        out_path: str,
        height: int,
        width: int,
        channels: int,
        fps: float = 30.0,
        is_rgb: bool = True,
    ) -> None:
        if channels == 3:
            pixel_format = "rgb24" if is_rgb else "bgr24"
        elif channels == 4:
            pixel_format = "rgba"
        else:
            raise ValueError(f"Unsupported channel count for video: {channels}")
        self.out_path = out_path
        self.height = height
        self.width = width
        self.channels = channels
        self.fps = fps
        self.n_frames = 0
        self._queue_size = max(1, int(os.environ.get("ROBODOJO_VIDEO_QUEUE_SIZE", "8")))
        self._queue_timeout = max(1.0, float(os.environ.get("ROBODOJO_VIDEO_QUEUE_TIMEOUT_S", "30")))
        self._close_timeout = max(1.0, float(os.environ.get("ROBODOJO_VIDEO_CLOSE_TIMEOUT_S", "60")))
        self._frames: queue.Queue[np.ndarray | object] = queue.Queue(maxsize=self._queue_size)
        self._closed = False
        self._state_lock = threading.Lock()
        self._done = threading.Event()
        self._error: BaseException | None = None
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        codec = _resolve_video_codec()
        if codec == "h264_nvenc":
            codec_args = [
                "-c:v",
                "h264_nvenc",
                "-preset",
                os.environ.get("ROBODOJO_NVENC_PRESET", "p4"),
                "-rc",
                "constqp",
                "-qp",
                os.environ.get("ROBODOJO_NVENC_QP", "23"),
            ]
        else:
            codec_args = [
                "-c:v",
                "libx264",
                "-preset",
                os.environ.get("ROBODOJO_X264_PRESET", "fast"),
                "-crf",
                os.environ.get("ROBODOJO_X264_CRF", "23"),
            ]

        self.proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                pixel_format,
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(fps),
                "-i",
                "-",
                "-pix_fmt",
                "yuv420p",
                *codec_args,
                out_path,
            ],
            stdin=subprocess.PIPE,
        )
        self._worker = threading.Thread(
            target=self._write_loop,
            name=f"robodojo-video-{Path(out_path).stem}",
            daemon=True,
        )
        self._worker.start()

    def _write_loop(self) -> None:
        try:
            while True:
                item = self._frames.get()
                try:
                    if item is self._SENTINEL:
                        return
                    if self.proc is None or self.proc.stdin is None:
                        raise RuntimeError("ffmpeg stdin is unavailable")
                    self.proc.stdin.write(item.tobytes())
                finally:
                    self._frames.task_done()
        except BaseException as exc:
            self._error = exc
        finally:
            # Make queue accounting complete after an encoder failure so close
            # and abort cannot hang on frames that will never be written.
            while True:
                try:
                    self._frames.get_nowait()
                except queue.Empty:
                    break
                else:
                    self._frames.task_done()

            proc = self.proc
            if proc is not None:
                try:
                    if proc.stdin is not None:
                        proc.stdin.close()
                except Exception:
                    pass
                try:
                    return_code = proc.wait(timeout=self._close_timeout)
                    if return_code != 0 and self._error is None:
                        self._error = OSError(f"ffmpeg exited with status {return_code}")
                except subprocess.TimeoutExpired as exc:
                    self._error = self._error or exc
                    try:
                        proc.kill()
                        proc.wait(timeout=5)
                    except Exception:
                        pass
            self._done.set()

    def append(self, frame: np.ndarray) -> None:
        with self._state_lock:
            if self._closed or self.proc is None:
                raise RuntimeError("Cannot append to a closed VideoStreamWriter.")
            if self._error is not None:
                raise RuntimeError("Video encoder failed") from self._error
        if frame.ndim != 3 or frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise ValueError(
                f"Frame shape {tuple(frame.shape)} does not match writer ({self.height}x{self.width}x{self.channels})."
            )
        if frame.shape[2] != self.channels:
            raise ValueError(f"Frame has {frame.shape[2]} channels; expected {self.channels}.")
        # Isaac/warp may reuse the source buffer immediately after this call.
        payload = np.ascontiguousarray(frame, dtype=np.uint8).copy()
        try:
            self._frames.put(payload, timeout=self._queue_timeout)
        except queue.Full as exc:
            raise TimeoutError(
                f"Video encoder queue stayed full for {self._queue_timeout:.1f}s: {self.out_path}"
            ) from exc
        self.n_frames += 1

    def close(self, *, announce: bool = True) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._frames.put(self._SENTINEL, timeout=self._queue_timeout)
        except queue.Full as exc:
            self.abort()
            raise TimeoutError(f"Video encoder queue could not be finalized: {self.out_path}") from exc
        if not self._done.wait(self._close_timeout + self._queue_timeout):
            self.abort()
            raise TimeoutError(f"Video encoder did not finish within timeout: {self.out_path}")
        self._worker.join(timeout=1)
        self.proc = None
        if self._error is not None:
            raise RuntimeError(f"Video encoder failed for `{self.out_path}`") from self._error
        if announce:
            print(
                format_video_saved_message(
                    self.out_path,
                    self.n_frames,
                    self.width,
                    self.height,
                    self.fps,
                )
            )

    def abort(self) -> None:
        """Stop encoding and remove a partial output file."""
        with self._state_lock:
            if self._closed and self._done.is_set():
                return
            self._closed = True
        if self.proc is not None:
            try:
                self.proc.kill()
            except Exception:
                pass
        try:
            self._frames.put_nowait(self._SENTINEL)
        except queue.Full:
            while True:
                try:
                    self._frames.get_nowait()
                except queue.Empty:
                    break
                else:
                    self._frames.task_done()
            try:
                self._frames.put_nowait(self._SENTINEL)
            except queue.Full:
                pass
        self._done.wait(timeout=5)
        self._worker.join(timeout=5)
        self.proc = None
        try:
            if os.path.exists(self.out_path):
                os.remove(self.out_path)
        except Exception:
            pass


def save_json(
    data: Any,
    path: str | os.PathLike,
    overwrite: bool = True,
    make_dirs: bool = True,
    sort_keys: bool = False,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    p = Path(path)

    if make_dirs:
        p.parent.mkdir(parents=True, exist_ok=True)

    if p.exists() and not overwrite:
        raise FileExistsError(f"{p} already exists and overwrite=False")

    tmp = p.with_suffix(p.suffix + ".tmp")

    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(
                data,
                f,
                ensure_ascii=ensure_ascii,
                sort_keys=sort_keys,
                indent=indent,
            )
            f.write("\n")
        os.replace(tmp, p)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        finally:
            raise
