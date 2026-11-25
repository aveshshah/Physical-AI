from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np
import pytest
from PIL import Image

from vlm_assistant import AppConfig, MediaConfig, VisionAssistant
from vlm_assistant.backends.openai_backend import OpenAICompatBackend, to_openai_messages
from vlm_assistant.cli import Repl, build_parser, config_from_args
from vlm_assistant.config import ChatConfig, ModelConfig
from vlm_assistant.conversation import Conversation
from vlm_assistant.media import FrameBuffer, ImageItem, VideoItem, Webcam, load_image, sample_video


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture
def video_file(tmp_path):
    """10-second, 10 fps synthetic video: frame i is uniformly filled with gray level i."""
    path = tmp_path / "clip.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 240))
    for i in range(100):
        w.write(np.full((240, 320, 3), i * 2, dtype=np.uint8))
    w.release()
    return path


@pytest.fixture
def image_file(tmp_path):
    path = tmp_path / "big.png"
    Image.new("RGB", (4000, 2000), (10, 200, 30)).save(path)
    return path


def mock_assistant(**chat) -> VisionAssistant:
    cfg = AppConfig.from_dict({"model": {"backend": "mock"}, "chat": chat})
    return VisionAssistant(cfg)


# ----------------------------------------------------------------------------- config
def test_config_from_dict_and_unknown_key():
    cfg = AppConfig.from_dict({"model": {"backend": "mock", "max_new_tokens": 7}, "media": {"video_fps": 1.0}})
    assert cfg.model.backend == "mock" and cfg.model.max_new_tokens == 7 and cfg.media.video_fps == 1.0
    with pytest.raises(ValueError):
        AppConfig.from_dict({"model": {"nope": 1}})


def test_example_configs_load():
    from pathlib import Path

    for path in (Path(__file__).parents[1] / "configs").glob("*.yaml"):
        AppConfig.load(path)


def test_cli_overrides():
    args = build_parser().parse_args(["--backend", "openai", "--quant", "4bit", "--camera", "2", "--base-url", "http://x/v1"])
    cfg = config_from_args(args)
    assert cfg.model.backend == "openai" and cfg.model.quantization == "4bit"
    assert cfg.media.webcam_source == 2 and cfg.model.base_url == "http://x/v1"


# ----------------------------------------------------------------------------- media
def test_load_image_downscales(image_file):
    item = load_image(image_file, MediaConfig(image_max_side=1000))
    assert item.image.size == (1000, 500)


def test_sample_video_uniform(video_file):
    v = sample_video(video_file, MediaConfig(video_fps=1.0, video_max_frames=32))
    assert len(v.frames) == 10
    assert v.timestamps[0] == pytest.approx(0.0)
    assert v.timestamps[-1] == pytest.approx(9.9, abs=0.15)
    levels = [np.asarray(f)[0, 0, 0] for f in v.frames]
    assert levels == sorted(levels)  # chronological


def test_sample_video_caps_frames_and_window(video_file):
    v = sample_video(video_file, MediaConfig(video_fps=10.0, video_max_frames=8), start=2.0, end=6.0)
    assert len(v.frames) == 8
    assert v.timestamps[0] == pytest.approx(2.0, abs=0.11)
    assert v.timestamps[-1] == pytest.approx(5.9, abs=0.11)


def test_sample_video_even_frame_count(video_file):
    v = sample_video(video_file, MediaConfig(video_fps=0.5, video_max_frames=32))  # 5 frames -> padded
    assert len(v.frames) % 2 == 0


def test_frame_buffer_rate_and_clip():
    buf = FrameBuffer(MediaConfig(webcam_buffer_seconds=4, webcam_buffer_fps=2, video_max_frames=32))
    for i in range(40):  # 10 s at 4 fps
        buf.push(np.full((48, 64, 3), i, dtype=np.uint8), t=i * 0.25)
    assert buf.buffered_seconds() == pytest.approx(4.0)  # trimmed to window, 2 fps
    clip = buf.clip(2.0)
    assert clip.duration == pytest.approx(2.0)
    assert len(clip.frames) % 2 == 0
    assert np.asarray(buf.snapshot().image)[0, 0, 0] == 39


class FakeCapture:
    def __init__(self):
        self.i = 0

    def isOpened(self):
        return True

    def read(self):
        time.sleep(0.01)
        self.i += 1
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        frame[..., 0] = 255  # BGR blue
        return True, frame

    def release(self):
        pass


def test_webcam_thread_with_fake_capture():
    cam = Webcam(MediaConfig(webcam_buffer_fps=50, webcam_buffer_seconds=2), capture_factory=FakeCapture)
    with cam:
        time.sleep(0.3)
        snap = cam.snapshot()
        r, g, b = np.asarray(snap.image)[0, 0]
        assert (r, g, b) == (0, 0, 255)  # BGR -> RGB conversion
        clip = cam.clip(0.2)
        assert len(clip.frames) >= 2
    assert not cam.running


# ----------------------------------------------------------------------------- conversation
def test_conversation_trims_history_and_old_media():
    conv = Conversation(ChatConfig(system_prompt="sys", max_history_turns=2, keep_media_turns=1))
    img = ImageItem(Image.new("RGB", (8, 8)), "a.png")
    for i in range(4):
        conv.add_user(f"q{i}", [img])
        conv.add_assistant(f"a{i}")
    msgs = conv.to_messages()
    assert msgs[0]["role"] == "system"
    assert [m["role"] for m in msgs[1:]] == ["user", "assistant", "user", "assistant"]
    older, newest = msgs[1], msgs[3]
    assert older["content"][0]["type"] == "text" and "earlier image" in older["content"][0]["text"]
    assert newest["content"][0]["type"] == "image"


def test_conversation_save(tmp_path):
    conv = Conversation()
    conv.add_user("hi", [ImageItem(Image.new("RGB", (8, 8)), "x.png")])
    conv.add_assistant("hello")
    data = json.loads(conv.save(tmp_path / "t.json").read_text())
    assert data["turns"][0]["media"] == ["image from x.png (8x8)"]


# ----------------------------------------------------------------------------- assistant
def test_assistant_chat_records_history():
    a = mock_assistant()
    img = ImageItem(Image.new("RGB", (32, 16)), "x")
    out = a.ask("what is it?", [img])
    assert "an image (32x16)" in out and "what is it?" in out
    out2 = a.ask("and now?")
    assert "user turn 2" in out2
    assert [t.role for t in a.conversation.turns] == ["user", "assistant", "user", "assistant"]


def test_assistant_failure_rolls_back_user_turn():
    a = mock_assistant()

    def boom(messages, **kw):
        raise RuntimeError("gpu on fire")
        yield  # pragma: no cover

    a.backend.stream = boom
    with pytest.raises(RuntimeError):
        a.ask("hello")
    assert a.conversation.turns == []


def test_oneshot_does_not_touch_history():
    a = mock_assistant()
    clip = VideoItem([Image.new("RGB", (8, 8))] * 4, [0, 0.5, 1, 1.5], "webcam", 2.0)
    assert "a video (4 frames, 1.5s)" in "".join(a.oneshot("live?", [clip]))
    assert a.conversation.turns == []


def test_repl_commands(image_file, video_file, tmp_path, capsys):
    a = mock_assistant()
    r = Repl(a)
    assert r.handle(f"/image {image_file}")
    assert len(r.pending) == 1
    assert r.handle(f"/video {video_file} 1 3")
    assert r.handle("what do you see?")
    out = capsys.readouterr().out
    assert "an image" in out and "a video" in out and r.pending == []
    assert r.handle(f"/save {tmp_path / 's.json'}")
    assert r.handle("/image /does/not/exist.png")
    assert "error" in capsys.readouterr().out
    assert r.handle("/reset") and a.conversation.turns == []
    assert r.handle("/quit") is False


# ----------------------------------------------------------------------------- openai backend
def test_openai_message_conversion():
    clip = VideoItem([Image.new("RGB", (8, 8))] * 2, [0.0, 0.5], "cam", 2.0)
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": "sys"}]},
        {"role": "user", "content": [{"type": "video", "video": clip}, {"type": "text", "text": "q"}]},
    ]
    out = to_openai_messages(msgs)
    assert out[0] == {"role": "system", "content": "sys"}
    kinds = [p["type"] for p in out[1]["content"]]
    assert kinds.count("image_url") == 2
    assert out[1]["content"][1]["text"] == "<0.0 seconds>"
    assert out[1]["content"][2]["image_url"]["url"].startswith("data:image/jpeg;base64,")


class _SSEHandler(BaseHTTPRequestHandler):
    received: dict = {}

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _SSEHandler.received = body
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for piece in ["Hello", " robot", "!"]:
            evt = {"choices": [{"delta": {"content": piece}}]}
            self.wfile.write(f"data: {json.dumps(evt)}\n\n".encode())
        usage = {"choices": [], "usage": {"prompt_tokens": 42, "completion_tokens": 3}}
        self.wfile.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())

    def log_message(self, *a):
        pass


def test_openai_backend_streams_from_server():
    server = HTTPServer(("127.0.0.1", 0), _SSEHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        cfg = ModelConfig(backend="openai", base_url=f"http://127.0.0.1:{server.server_port}/v1", model_id="qwen3-vl")
        backend = OpenAICompatBackend(cfg)
        msgs = [{"role": "user", "content": [{"type": "image", "image": Image.new("RGB", (8, 8))},
                                             {"type": "text", "text": "hi"}]}]
        assert "".join(backend.stream(msgs, max_new_tokens=5)) == "Hello robot!"
        assert backend.last_stats.prompt_tokens == 42 and backend.last_stats.completion_tokens == 3
        assert _SSEHandler.received["max_tokens"] == 5 and _SSEHandler.received["model"] == "qwen3-vl"
    finally:
        server.shutdown()
