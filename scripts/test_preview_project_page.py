"""Exercise real HTTP responses so media seeks cannot silently return full files."""

from functools import partial
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import unittest

from preview_project_page import MediaPreviewHandler


class QuietHandler(MediaPreviewHandler):
    def log_message(self, *args):
        pass


class MediaPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = TemporaryDirectory()
        cls.payload = bytes(range(256)) * 1024
        root = Path(cls.directory.name)
        (root / "sample.wav").write_bytes(cls.payload)
        (root / "empty.wav").touch()
        (root / "index.html").write_text("<h1>Project page</h1>")
        cls.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), partial(QuietHandler, directory=str(root))
        )
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.directory.cleanup()

    def request(self, path="/sample.wav?v=1", headers=None, method="GET"):
        connection = HTTPConnection(*self.server.server_address, timeout=5)
        try:
            connection.request(method, path, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_media_ranges(self):
        size = len(self.payload)
        for requested, start, end in [
            ("bytes=0-1", 0, 1),  # Browser capability probe.
            ("bytes=100000-170000", 100000, 170000),
            ("bytes=200000-", 200000, size - 1),
            ("bytes=-100", size - 100, size - 1),
            ("bytes=200000-999999", 200000, size - 1),
        ]:
            with self.subTest(requested=requested):
                status, headers, body = self.request(headers={"Range": requested})
                self.assertEqual(status, 206)
                self.assertEqual(headers["Accept-Ranges"], "bytes")
                self.assertEqual(headers["Content-Range"], f"bytes {start}-{end}/{size}")
                self.assertEqual(int(headers["Content-Length"]), end - start + 1)
                self.assertEqual(body, self.payload[start:end + 1])

    def test_unsatisfiable_ranges(self):
        for requested in ["bytes=999999-", "bytes=10-5", "bytes=-0"]:
            with self.subTest(requested=requested):
                status, headers, body = self.request(headers={"Range": requested})
                self.assertEqual(status, 416)
                self.assertEqual(headers["Content-Range"], f"bytes */{len(self.payload)}")
                self.assertEqual(body, b"")
        self.assertEqual(self.request("/empty.wav", {"Range": "bytes=0-1"})[0], 416)

    def test_full_file_and_unsupported_ranges(self):
        for headers in [
            {}, {"Range": "invalid"}, {"Range": "bytes=0-1,10-20"},
            {"Range": "bytes=0-1", "If-Range": '"old-version"'},
        ]:
            with self.subTest(headers=headers):
                status, _, body = self.request(headers=headers)
                self.assertEqual(status, 200)
                self.assertEqual(body, self.payload)

    def test_page_head_and_missing_file(self):
        self.assertEqual(self.request("/")[2], b"<h1>Project page</h1>")
        status, headers, body = self.request(method="HEAD")
        self.assertEqual(status, 200)
        self.assertEqual(int(headers["Content-Length"]), len(self.payload))
        self.assertEqual(body, b"")
        self.assertEqual(self.request("/missing.wav")[0], 404)


if __name__ == "__main__":
    unittest.main()
