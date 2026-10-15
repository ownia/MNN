import argparse
import concurrent.futures
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import wave


class Qwen3TTSServerTest(unittest.TestCase):
    base_url = "http://127.0.0.1:8000"
    model = "qwen3-tts"
    voice = "default"
    timeout = 180
    binary = None
    recovery_ref_audio = None

    def payload(self):
        return {"model": self.model, "voice": self.voice, "input": "Hello.", "response_format": "wav"}

    def request(self, payload=None, path="/v1/audio/speech", raw=None, timeout=None):
        data = json.dumps(payload).encode("utf-8") if payload is not None else raw
        request = urllib.request.Request(
            self.base_url + path, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout if timeout is None else timeout) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as error:
            with error:
                return error.code, error.headers, error.read()

    def assert_error(self, result, status, parameter=None):
        actual_status, headers, data = result
        self.assertEqual(actual_status, status, data)
        self.assertEqual(headers.get_content_type(), "application/json")
        error = json.loads(data)["error"]
        self.assertTrue(error["message"])
        self.assertEqual(error["type"], "server_error" if status >= 500 else "invalid_request_error")
        self.assertEqual(error["param"], parameter)

    def assert_waveform(self, result):
        status, headers, data = result
        self.assertEqual(status, 200, data[:300])
        self.assertEqual(headers.get_content_type(), "audio/wav")
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(int(headers["Content-Length"]), len(data))
        self.assertNotIn("Transfer-Encoding", headers)
        first_audio_ms = float(headers["X-TTFA-Ms"])
        generation_ms = float(headers["X-Generation-Ms"])
        self.assertGreater(first_audio_ms, 0)
        self.assertGreaterEqual(generation_ms, first_audio_ms)
        with wave.open(io.BytesIO(data), "rb") as waveform:
            self.assertEqual(waveform.getnchannels(), 1)
            self.assertEqual(waveform.getsampwidth(), 2)
            self.assertEqual(waveform.getframerate(), 24000)
            self.assertGreater(waveform.getnframes(), 0)
            samples = waveform.readframes(waveform.getnframes())
            self.assertEqual(len(samples), waveform.getnframes() * 2)
            self.assertTrue(any(samples))

    def test_health_and_models(self):
        status, _, data = self.request(path="/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["status"], "ok")
        status, _, data = self.request(path="/v1/models")
        self.assertEqual(status, 200)
        models = json.loads(data)
        self.assertEqual(models["object"], "list")
        self.assertEqual([model["id"] for model in models["data"]], [self.model])

    def test_invalid_json(self):
        for raw in (b"{", b"[]", b"null", b"", b'{"input":"\xff"}'):
            with self.subTest(raw=raw):
                self.assert_error(self.request(raw=raw), 400)

    def test_required_fields(self):
        for field in ("model", "input", "voice"):
            for value in (None, "", 1, [], {}):
                with self.subTest(field=field, value=value):
                    payload = self.payload()
                    payload[field] = value
                    self.assert_error(self.request(payload), 400, field)
            payload = self.payload()
            del payload[field]
            self.assert_error(self.request(payload), 400, field)

    def test_unsupported_options(self):
        for field, value in (
            ("response_format", "mp3"), ("response_format", 1),
            ("speed", 1.5), ("speed", "1"), ("speed", None),
            ("stream", True), ("stream", "false"),
            ("voice", "unknown-voice"), ("ref_audio", "/client/path.wav"),
            ("instructions", "unsupported"), ("input", "   \n"), ("input", "x" * 4097),
        ):
            with self.subTest(field=field, value=value):
                payload = self.payload()
                payload[field] = value
                self.assert_error(self.request(payload), 400, field)
        payload = self.payload()
        payload["model"] = "unknown-model"
        self.assert_error(self.request(payload), 404, "model")

    def test_payload_limit_and_unknown_route(self):
        self.assert_error(self.request(raw=b"x" * 16385), 413)
        self.assert_error(self.request(path="/not-an-endpoint"), 404)

    def test_repeated_requests(self):
        self.assert_waveform(self.request(self.payload()))
        payload = self.payload()
        del payload["response_format"]
        payload.update(speed=1.0, stream=False)
        self.assert_waveform(self.request(payload))

    def test_concurrent_requests(self):
        clients = 8
        barrier = threading.Barrier(clients)

        def synthesize():
            barrier.wait(timeout=10)
            return self.request(self.payload())

        with concurrent.futures.ThreadPoolExecutor(max_workers=clients) as executor:
            futures = [executor.submit(synthesize) for _ in range(clients)]
            first = next(concurrent.futures.as_completed(futures))
            self.assert_error(first.result(), 503)
            for path in ("/health", "/v1/models"):
                status, _, _ = self.request(path=path, timeout=2)
                self.assertEqual(status, 200)
            completed = 0
            rejected = 0
            for future in futures:
                result = future.result()
                if result[0] == 503:
                    self.assert_error(result, 503)
                    self.assertGreater(int(result[1]["Retry-After"]), 0)
                    rejected += 1
                else:
                    self.assert_waveform(result)
                    completed += 1
            self.assertGreater(completed, 0)
            self.assertGreater(rejected, 0)

    def test_generation_error_recovery(self):
        if self.recovery_ref_audio is None:
            self.skipTest("Requires --recovery-ref-audio pointing to the server's disposable reference WAV.")
        reference = Path(self.recovery_ref_audio).resolve(strict=True)
        self.assert_waveform(self.request(self.payload()))
        with tempfile.TemporaryDirectory(prefix="mnn-tts-recovery-", dir=reference.parent) as directory:
            unavailable = Path(directory) / "reference.wav"
            reference.rename(unavailable)
            try:
                self.assert_error(self.request(self.payload()), 500)
            finally:
                unavailable.rename(reference)
        self.assert_waveform(self.request(self.payload()))

    def test_startup_rejects_non_tts_models(self):
        if self.binary is None:
            self.skipTest("Requires --binary pointing to qwen3_tts_demo.")
        binary = str(Path(self.binary).resolve(strict=True))
        with tempfile.TemporaryDirectory(prefix="mnn-tts-startup-") as directory:
            model_dir = Path(directory)
            reference = model_dir / "reference.wav"
            with wave.open(str(reference), "wb") as waveform:
                waveform.setnchannels(1)
                waveform.setsampwidth(2)
                waveform.setframerate(24000)
                waveform.writeframes(b"\0" * 4800)
            configuration = {
                "llm_config": "metadata.json",
                "model_type": "qwen3_tts",
                "talker_type": "qwen3_tts",
                "has_talker": True,
            }
            (model_dir / "config.json").write_text(json.dumps(configuration), encoding="utf-8")
            for metadata in (
                {"model_type": "qwen2", "talker_type": "", "has_talker": False},
                {"model_type": "qwen3_tts", "talker_type": "qwen3_tts", "has_talker": False},
                {"model_type": "qwen2_5_omni", "talker_type": "qwen2_5_omni", "has_talker": True},
            ):
                with self.subTest(metadata=metadata):
                    (model_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
                    result = subprocess.run(
                        [binary, directory, "--serve", "--ref_audio", str(reference)],
                        capture_output=True, text=True, timeout=30, cwd=directory,
                    )
                    output = result.stdout + result.stderr
                    self.assertEqual(result.returncode, 1, output)
                    self.assertIn("server requires a Qwen3-TTS model", output)
                    self.assertNotIn("Qwen3-TTS server:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke-test a running Qwen3-TTS server.")
    parser.add_argument("--url", default=Qwen3TTSServerTest.base_url)
    parser.add_argument("--model", default=Qwen3TTSServerTest.model)
    parser.add_argument("--voice", default=Qwen3TTSServerTest.voice)
    parser.add_argument("--timeout", type=float, default=Qwen3TTSServerTest.timeout)
    parser.add_argument("--binary", help="Local demo executable for startup-validation tests.")
    parser.add_argument(
        "--recovery-ref-audio", help="Server's disposable reference WAV; temporarily renamed to test engine recovery."
    )
    parser.add_argument("tests", nargs="*", help="Optional unittest test names.")
    arguments = parser.parse_args()
    Qwen3TTSServerTest.base_url = arguments.url.rstrip("/")
    Qwen3TTSServerTest.model = arguments.model
    Qwen3TTSServerTest.voice = arguments.voice
    Qwen3TTSServerTest.timeout = arguments.timeout
    Qwen3TTSServerTest.binary = arguments.binary
    Qwen3TTSServerTest.recovery_ref_audio = arguments.recovery_ref_audio
    unittest.main(argv=[sys.argv[0]] + arguments.tests, verbosity=2)