"""How a booting pod gets its code (2026-09-30): the gateway PUSHES it through RunPod's proxy to
`docker/pod_receiver.py`, because a pod in EUR-IS-1 cannot open a connection to the VPS at all
(the first real run of the old "pod fetches from the gateway" design hung in curl for 13 min,
stopped itself, and was started again). Everything here runs on localhost: the real receiver,
the real `runpod_pods.push_code`, and -- in the last test -- the real start command under bash."""

from __future__ import annotations

import base64
import importlib.util
import io
import os
import socket
import subprocess
import tarfile
import threading
import time
import zlib

import pytest

from apps.gateway import runpod_pods as rp

KEY = "k" * 40


def _load_receiver():
    spec = importlib.util.spec_from_file_location("pod_receiver", rp.POD_RECEIVER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _bundle(bootstrap: str) -> bytes:
    """A stand-in for the gateway's `git archive`: the bootstrap plus one other file."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, text in (("docker/runpod_bootstrap.sh", bootstrap), ("start_app.py", "print('app')\n")):
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _wait_until_awaiting(url: str, timeout_s: float = 10.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if rp.pod_awaits_code(url, timeout_s=2):
            return
        time.sleep(0.05)
    raise AssertionError("the receiver never came up")


def test_receiver_travels_in_the_pod_spec_intact_and_small():
    packed = rp.pod_receiver_env()
    assert zlib.decompress(base64.b64decode(packed)) == rp.POD_RECEIVER_SCRIPT.read_bytes()
    assert len(packed) < 4096  # an env var in RunPod's pod spec -- keep it modest


def test_receiver_takes_only_the_keyed_bundle_then_exits(tmp_path, monkeypatch):
    port = _free_port()
    tgz, boot = tmp_path / "code.tar.gz", tmp_path / "bootstrap.sh"
    monkeypatch.setenv("PORT", str(port))
    monkeypatch.setenv("PV_POD_KEY", KEY)
    monkeypatch.setenv("PV_CODE_TGZ", str(tgz))
    monkeypatch.setenv("PV_BOOTSTRAP", str(boot))
    monkeypatch.setenv("PV_CODE_WAIT_S", "30")
    receiver = _load_receiver()
    result: dict[str, int] = {}
    thread = threading.Thread(target=lambda: result.update(rc=receiver.main()), daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    _wait_until_awaiting(url)

    # the gateway's health probe keeps reading "not up yet" while the pod waits for its code
    assert rp.probe_health(url) is None
    bundle = _bundle("echo booted\n")
    with pytest.raises(rp.RunPodError) as wrong_key:
        rp.push_code(url, "not-the-key", bundle, "abc123")
    assert wrong_key.value.status == 403
    with pytest.raises(rp.RunPodError) as garbage:
        rp.push_code(url, KEY, b"not a tarball", "abc123")
    assert garbage.value.status == 400
    assert not tgz.exists() and thread.is_alive()  # neither was accepted; still waiting

    assert rp.push_code(url, KEY, bundle, "abc123def456") is True
    thread.join(10)
    assert result == {"rc": 0}
    assert tgz.read_bytes() == bundle
    assert (tmp_path / "code.tar.gz.commit").read_text() == "abc123def456"
    assert boot.read_text() == "echo booted\n" and os.access(boot, os.X_OK)
    assert rp.pod_awaits_code(url, timeout_s=2) is False  # gone: the bootstrap owns the port now


def test_receiver_gives_up_when_no_code_arrives(tmp_path, monkeypatch):
    monkeypatch.setenv("PORT", str(_free_port()))
    monkeypatch.setenv("PV_POD_KEY", KEY)
    monkeypatch.setenv("PV_CODE_TGZ", str(tmp_path / "code.tar.gz"))
    monkeypatch.setenv("PV_CODE_WAIT_S", "0.2")
    assert _load_receiver().main() == 1  # -> the start command's `|| stop-self`


def test_start_command_runs_the_bootstrap_it_was_pushed(tmp_path):
    """The exact `dockerStartCmd` the gateway gives RunPod, run under bash: it must exec the
    receiver from the env, take the push, and run the pushed bootstrap with PV_CODE_TGZ set."""
    port = _free_port()
    env = {"PV_POD_RECEIVER": rp.pod_receiver_env(), "PV_POD_KEY": KEY}
    cmd = rp.pod_create_body(name="p", gpu_type_ids=["x"], volume_id="v", env=env)["dockerStartCmd"]
    assert cmd[:2] == ["bash", "-c"]
    # Same command, with the pod's /tmp paths pointed into this test's own directory.
    script = cmd[2].replace(rp.POD_CODE_TGZ, str(tmp_path / "code.tar.gz"))
    script = script.replace(rp.POD_BOOTSTRAP, str(tmp_path / "bootstrap.sh"))
    proc_env = {"PATH": os.environ["PATH"], "PORT": str(port), "PV_CODE_WAIT_S": "30", **env}
    proc = subprocess.Popen(["bash", "-c", script], env=proc_env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    try:
        _wait_until_awaiting(f"http://127.0.0.1:{port}")
        bundle = _bundle('echo "BOOTSTRAP RAN with $PV_CODE_TGZ"; tar -tzf "$PV_CODE_TGZ" | sort\n')
        assert rp.push_code(f"http://127.0.0.1:{port}", KEY, bundle, "c0ffee") is True
        out, _ = proc.communicate(timeout=30)
    finally:
        proc.kill()
    assert proc.returncode == 0, out
    assert f"BOOTSTRAP RAN with {tmp_path / 'code.tar.gz'}" in out
    assert "docker/runpod_bootstrap.sh" in out and "start_app.py" in out
    assert "stopping this pod" not in out
