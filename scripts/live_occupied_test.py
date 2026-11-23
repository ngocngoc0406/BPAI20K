"""Camera occupied by another application: a separate process holds camera 0, then the live assistant tries to start.
Records whether the start is refused with a clear message (or, if the driver allows sharing, that it works).
Output: experiments/live/cam_occupied.json
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.live.audio import AudioConfig  # noqa: E402
from src.live.controller import LiveController  # noqa: E402
from src.live.sources import CameraSource  # noqa: E402
from src.utils.config import resolve  # noqa: E402
from src.utils.io import save_json  # noqa: E402

HOLDER = ("import cv2, time\ncap = cv2.VideoCapture(0, cv2.CAP_DSHOW)\nok, _ = cap.read()\nprint('HOLDING', ok, flush=True)\n"
          "t = time.time()\nwhile time.time() - t < 25:\n    cap.read()\ncap.release()\n")


def main() -> int:
    holder = subprocess.Popen([sys.executable, "-c", HOLDER], stdout=subprocess.PIPE, text=True)
    line = holder.stdout.readline().strip()
    time.sleep(1.0)
    ctl = LiveController()
    out = {"holder": line}
    t0 = time.time()
    try:
        ctl.start(CameraSource(0), resolve("models/safedrive_v3"), audio_backend="none", record_events=False,
                  audio_cfg=AudioConfig(enabled=False))
        time.sleep(5)
        s, frame = ctl.snapshot()
        out.update(started=True, frames_received=frame is not None, state=s.get("camera"), error=s.get("error"), degraded=s.get("degraded"), input_interval_s=s.get("input_interval_s"), status=s.get("status"), stalled_s=s.get("stalled_s"),
                   capture_fps=s.get("capture_fps"))
        ctl.stop()
    except Exception as e:  # noqa: BLE001
        out.update(started=False, error_type=type(e).__name__, message=str(e))
    out["seconds_to_outcome"] = time.time() - t0
    holder.wait(40)
    # after the other app releases the camera, the assistant must be able to start
    try:
        ctl2 = LiveController()
        ctl2.start(CameraSource(0), resolve("models/safedrive_v3"), audio_backend="none", record_events=False, audio_cfg=AudioConfig(enabled=False))
        time.sleep(3)
        s, frame = ctl2.snapshot()
        out["after_release_started"] = frame is not None
        ctl2.stop()
    except Exception as e:  # noqa: BLE001
        out["after_release_started"] = False
        out["after_release_error"] = str(e)
    save_json(out, resolve("experiments/live/cam_occupied.json"))
    print("OCCUPIED", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
