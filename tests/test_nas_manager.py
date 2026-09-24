import io
from pathlib import Path

import pytest

pytest.importorskip("paramiko")

from scripts.nas_manager import _filter_expected_stderr, sftp_sync_file_smart


class _Stat:
    def __init__(self, size):
        self.st_size = size


class _FakeSftp:
    def __init__(self, content: bytes):
        self.content = content
        self.put_count = 0

    def stat(self, _remote_path):
        return _Stat(len(self.content))

    def open(self, _remote_path, _mode):
        return io.BytesIO(self.content)

    def put(self, local_path, _remote_path):
        self.put_count += 1
        self.content = Path(local_path).read_bytes()


def test_sync_same_size_content_change_is_uploaded(tmp_path):
    local_path = tmp_path / "payload.bin"
    local_path.write_bytes(b"abcd")
    sftp = _FakeSftp(b"abce")

    sftp_sync_file_smart(sftp, local_path, "/remote/payload.bin")
    assert sftp.put_count == 1

    sftp_sync_file_smart(sftp, local_path, "/remote/payload.bin")
    assert sftp.put_count == 1


def test_stderr_filter_is_username_agnostic_and_preserves_errors():
    stderr = "[sudo] password for another-user:\nCould not chdir to home directory /home/another-user: No such file\nreal failure\n"
    assert _filter_expected_stderr(stderr, suppress_password_prompt=True) == "real failure"
    assert "password for another-user" in _filter_expected_stderr(stderr, suppress_password_prompt=False)
