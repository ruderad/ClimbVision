"""Release safety nets: a replaced video is noticed, --version, folders as arguments."""

import json

import cv2
import numpy as np

from climbvision.cli import expand_results, expand_videos, version_text
from climbvision.engine.video import fingerprint, video_changed, warn_if_changed


def _video(path, n, fps=30.0, size=(64, 48)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n):
        w.write(np.full((size[1], size[0], 3), i % 255, np.uint8))
    w.release()
    return path


def test_replaced_video_is_noticed(tmp_path, capsys):
    v = _video(tmp_path / "clip.mp4", 60)
    summary = {"video": str(v), "frames": 60, "fps": 30.0, "video_fingerprint": fingerprint(v)}
    assert video_changed(summary, v) is None
    _video(v, 45)  # the file is replaced by a shorter cut
    assert "changed" in video_changed(summary, v)
    # results from before fingerprints: judged on fps and frame count
    old = {"video": str(v), "frames": 60, "fps": 30.0}
    assert "frames" in video_changed(old, v)
    assert video_changed({"frames": 44, "fps": 30.0}, v) is None  # a frame or two short: fine
    assert "fps" in video_changed({"frames": 45, "fps": 25.0}, v)
    (tmp_path / "res").mkdir()
    (tmp_path / "res" / "summary.json").write_text(json.dumps(summary))
    assert warn_if_changed(tmp_path / "res", v)
    assert "re-run" in capsys.readouterr().err


def test_version_and_folders(tmp_path):
    assert version_text().startswith("climbvision ") and "climbvision" in version_text()[12:]
    vids = tmp_path / "videos"
    vids.mkdir()
    for name in ("a.MOV", "b.mp4", "notes.txt", ".hidden.mov"):
        (vids / name).write_bytes(b"x")
    assert [p.name for p in expand_videos([vids])] == ["a.MOV", "b.mp4"]
    out = tmp_path / "outputs"
    for name in ("a", "b", "_inspect", "empty"):
        (out / name).mkdir(parents=True)
    for name in ("a", "b", "_inspect"):
        (out / name / "summary.json").write_text("{}")
    assert [p.name for p in expand_results([out])] == ["a", "b"]
    assert expand_results([out / "a"]) == [out / "a"]
