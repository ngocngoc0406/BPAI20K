"""Dataset loading, annotation parsing, validation, splitting, statistics."""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.annotation_parser import AnnotationError, parse_annotations
from src.data.dataset_loader import load_dataset, records_to_frame
from src.data.dataset_splitter import split_videos
from src.data.dataset_statistics import compute_statistics
from src.data.dataset_validator import validate


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_parse_canonical(tmp_path):
    p = _write_csv(tmp_path / "a.csv", [{"id": 12, "time_of_event": 20.1, "time_of_alert": 18.0, "target": 1},
                                        {"id": 7, "time_of_event": None, "time_of_alert": None, "target": 0}])
    df = parse_annotations(p)
    assert list(df.video_id) == ["00012", "00007"]
    assert df.label.tolist() == [1, 0]
    assert df.time_of_event.isna().tolist() == [False, True]


def test_parse_nexar_hf_layout(tmp_path):
    for cls, te in (("positive", 19.5), ("negative", None)):
        (tmp_path / cls).mkdir()
        _write_csv(tmp_path / cls / "metadata.csv", [{"file_name": "00822.mp4", "time_of_event": te,
                                                      "time_of_alert": te, "light_conditions": "Normal"}])
    assert parse_annotations(tmp_path / "positive" / "metadata.csv").label.tolist() == [1]
    neg = parse_annotations(tmp_path / "negative" / "metadata.csv")
    assert neg.label.tolist() == [0] and neg.video_id.tolist() == ["00822"]


def test_parse_missing_columns(tmp_path):
    p = _write_csv(tmp_path / "b.csv", [{"foo": 1}])
    with pytest.raises(AnnotationError):
        parse_annotations(p)


def test_parse_missing_file(tmp_path):
    with pytest.raises(AnnotationError):
        parse_annotations(tmp_path / "nope.csv")


def test_load_and_validate(tmp_path, zoom_video, blank_video):
    raw = tmp_path / "raw"
    (raw / "positive").mkdir(parents=True)
    (raw / "negative").mkdir(parents=True)
    (raw / "positive" / "00001.mp4").write_bytes(zoom_video.read_bytes())
    (raw / "negative" / "00002.mp4").write_bytes(blank_video.read_bytes())
    (raw / "negative" / "00003.mp4").write_bytes(b"not a video")  # corrupt file
    ann = _write_csv(tmp_path / "ann.csv", [
        {"id": 1, "time_of_event": 3.0, "time_of_alert": 2.0, "target": 1},
        {"id": 2, "time_of_event": None, "time_of_alert": None, "target": 0},
        {"id": 3, "time_of_event": None, "time_of_alert": None, "target": 0},
        {"id": 4, "time_of_event": 99.0, "time_of_alert": 98.0, "target": 1},  # no video
        {"id": 5, "time_of_event": None, "time_of_alert": None, "target": 1},  # missing event time
    ])
    recs = load_dataset(ann, raw, only_available=False)
    rep = validate(recs)
    assert "00001" in rep.ok and "00002" in rep.ok
    assert "video unreadable / corrupt" in rep.issues["00003"]
    assert "video file missing" in rep.issues["00004"]
    assert any("without time_of_event" in m for m in rep.issues["00005"])
    stats = compute_statistics(records_to_frame(recs), rep.video_info)
    assert stats["n_positive"] == 3 and stats["n_negative"] == 2


def test_split_no_leakage(toy_dataset):
    recs, _ = toy_dataset
    s = split_videos(recs, {"train": 0.5, "val": 0.25, "test": 0.25}, seed=1)
    allv = s["train"] + s["val"] + s["test"]
    assert len(allv) == len(set(allv)) == len(recs)
    assert split_videos(recs, {"train": 0.5, "val": 0.25, "test": 0.25}, seed=1) == s  # deterministic
