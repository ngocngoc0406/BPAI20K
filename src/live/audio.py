"""Audio warnings (local, Windows winsound; no network).

- Three tone patterns: CAUTION = one soft low chime, WARNING = two medium beeps, HIGH RISK = three fast high beeps.
- Spoken Vietnamese text is NOT used: no vi-VN speech voice is installed on the test machine (only en-US SAPI voices),
  so the Vietnamese message is shown on screen together with the tone.
- Anti-repetition: per-level cooldown + a global minimum gap. A HIGHER level may interrupt a lower level's cooldown
  (after the global gap); the same or a lower level is suppressed while its cooldown runs.
- Messages describe what the system observed; they never assert that a collision will happen.
"""
from __future__ import annotations

import io
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.utils.config import resolve

LEVEL_RANK = {"NORMAL": 0, "UNCERTAIN": 0, "CAUTION": 1, "WARNING": 2, "HIGH RISK": 3}
MESSAGES = {
    "CAUTION": "Có tình huống giao thông cần quan sát.",
    "WARNING_LEAD": "Chú ý phương tiện phía trước.",
    "WARNING": "Có tình huống giao thông cần quan sát kỹ.",
    "HIGH RISK": "Nguy cơ va chạm, hãy kiểm tra tình hình phía trước.",
}
TONES = {"CAUTION": [(523, 0.18)], "WARNING": [(880, 0.15), (0, 0.08), (880, 0.15)],
         "HIGH RISK": [(1320, 0.10), (0, 0.05), (1320, 0.10), (0, 0.05), (1320, 0.10)]}


def tone_wav(pattern: list[tuple[float, float]], volume: float, sr: int = 22050) -> bytes:
    chunks = []
    for freq, dur in pattern:
        n = int(sr * dur)
        t = np.arange(n) / sr
        s = np.zeros(n) if freq == 0 else np.sin(2 * np.pi * freq * t)
        env = np.minimum(1.0, np.minimum(t / 0.01, (dur - t) / 0.02)) if n else s  # click-free attack / release
        chunks.append(s * env)
    x = np.concatenate(chunks) if chunks else np.zeros(1)
    pcm = (np.clip(x * max(0.0, min(volume, 1.0)), -1, 1) * 32767 * 0.8).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


@dataclass
class AudioConfig:
    enabled: bool = True
    volume: float = 0.6
    cooldown_s: dict = field(default_factory=lambda: {"CAUTION": 20.0, "WARNING": 8.0, "HIGH RISK": 5.0})
    global_gap_s: float = 2.0
    sound_caution: bool = True


class AudioAlerter:
    def __init__(self, cfg: AudioConfig | None = None, sound_dir: str | Path = resolve("data/live/sounds"), backend: str = "auto"):
        self.cfg = cfg or AudioConfig()
        self.dir = Path(sound_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.last_level_t: dict[str, float] = {}
        self.last_any_t = -1e9
        self.log: list[dict] = []
        self.available, self.reason = self._probe(backend)
        self._vol_written: float | None = None

    def _probe(self, backend: str) -> tuple[bool, str]:
        if backend == "none":
            return False, "audio disabled (no backend)"
        try:
            import winsound  # noqa: F401
        except ImportError:
            return False, "winsound not available (non-Windows system)"
        return True, "winsound"

    def _ensure_files(self) -> None:
        if self._vol_written == self.cfg.volume:
            return
        for lvl, pat in TONES.items():
            (self.dir / f"{lvl.replace(' ', '_')}.wav").write_bytes(tone_wav(pat, self.cfg.volume))
        self._vol_written = self.cfg.volume

    @staticmethod
    def message(level: str, lead_in_corridor: bool) -> str:
        if level == "WARNING":
            return MESSAGES["WARNING_LEAD"] if lead_in_corridor else MESSAGES["WARNING"]
        return MESSAGES.get(level, "")

    def decide(self, level: str, now: float | None = None) -> tuple[bool, str]:
        """Pure decision (testable): should this level sound now?"""
        now = time.time() if now is None else now
        r = LEVEL_RANK.get(level, 0)
        if r == 0:
            return False, "normal"
        if level == "CAUTION" and not self.cfg.sound_caution:
            return False, "caution sound off"
        if now - self.last_any_t < self.cfg.global_gap_s:
            return False, "global gap"
        if now - self.last_level_t.get(level, -1e9) < self.cfg.cooldown_s.get(level, 5.0):
            return False, "level cooldown"
        # a lower or equal level is suppressed while ANY higher level is still in its own cooldown
        for other, t_o in self.last_level_t.items():
            if LEVEL_RANK[other] > r and now - t_o < self.cfg.cooldown_s.get(other, 5.0):
                return False, f"higher level {other} recently"
        return True, "ok"

    def alert(self, level: str, lead_in_corridor: bool = False, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        ok, why = self.decide(level, now)
        rec = {"t": now, "level": level, "message": self.message(level, lead_in_corridor), "played": False, "reason": why, "api_ms": None}
        if ok:
            self.last_level_t[level] = now
            self.last_any_t = now
            if not self.cfg.enabled:
                rec["reason"] = "muted by user"
            elif not self.available:
                rec["reason"] = f"unavailable: {self.reason}"
            else:
                try:
                    import winsound
                    self._ensure_files()
                    t0 = time.perf_counter()
                    winsound.PlaySound(str(self.dir / f"{level.replace(' ', '_')}.wav"), winsound.SND_FILENAME | winsound.SND_ASYNC)
                    rec["api_ms"] = 1000 * (time.perf_counter() - t0)
                    rec["played"] = True
                except Exception as e:  # noqa: BLE001
                    self.available, self.reason = False, f"playback failed: {e!r}"
                    rec["reason"] = self.reason
        self.log.append(rec)
        self.log = self.log[-500:]
        return rec

