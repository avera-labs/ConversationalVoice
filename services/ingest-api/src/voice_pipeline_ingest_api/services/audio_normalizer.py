from __future__ import annotations

import multiprocessing
import os
import signal
from collections.abc import Callable
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Literal

from .pydub_worker import normalize_with_pydub
from .wav_validation import WavMetadata, WavValidationError, validate_normalized_wav

PYDUB_TIMEOUT_SECONDS = 30.0
WORKER_START_TIMEOUT_SECONDS = 10.0
WORKER_TERMINATION_GRACE_SECONDS = 1.0

WorkerOperation = Callable[[str, str], None]
WorkerState = tuple[Literal["ready", "ok", "error"], str | None]


class AudioNormalizationError(RuntimeError):
    """Raised when audio normalization cannot produce a valid WAV."""


class AudioNormalizationTimeout(AudioNormalizationError):
    """Raised when pydub normalization exceeds its execution deadline."""


def _worker_entry(
    connection: Connection,
    operation: WorkerOperation,
    source_path: str,
    destination_path: str,
) -> None:
    try:
        os.setsid()
        connection.send(("ready", None))
        operation(source_path, destination_path)
        connection.send(("ok", None))
    except Exception as exc:  # noqa: BLE001
        try:
            connection.send(("error", type(exc).__name__))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def _remove_incomplete_output(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


class AudioNormalizer:
    """Run pydub in an isolated process and enforce a hard timeout."""

    def __init__(
        self,
        *,
        timeout_seconds: float = PYDUB_TIMEOUT_SECONDS,
        start_timeout_seconds: float = WORKER_START_TIMEOUT_SECONDS,
        termination_grace_seconds: float = WORKER_TERMINATION_GRACE_SECONDS,
        operation: WorkerOperation = normalize_with_pydub,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if start_timeout_seconds <= 0:
            raise ValueError("start_timeout_seconds must be greater than zero")
        if termination_grace_seconds <= 0:
            raise ValueError("termination_grace_seconds must be greater than zero")

        self._timeout_seconds = timeout_seconds
        self._start_timeout_seconds = start_timeout_seconds
        self._termination_grace_seconds = termination_grace_seconds
        self._operation = operation
        self._context = multiprocessing.get_context("spawn")

    def normalize(self, source_path: Path, destination_path: Path) -> WavMetadata:
        """Normalize one audio file and return validated WAV metadata."""
        receive_connection, send_connection = self._context.Pipe(duplex=False)
        process = self._context.Process(
            target=_worker_entry,
            args=(
                send_connection,
                self._operation,
                str(source_path),
                str(destination_path),
            ),
            name="pydub-normalizer",
        )
        process_group_ready = False

        try:
            process.start()
            send_connection.close()

            state = self._receive_state(
                receive_connection,
                timeout_seconds=self._start_timeout_seconds,
            )
            if state is None or state[0] != "ready":
                self._stop_worker(process, process_group_ready=False)
                raise AudioNormalizationError(
                    "Audio normalization worker failed to start."
                )
            process_group_ready = True

            state = self._receive_state(
                receive_connection,
                timeout_seconds=self._timeout_seconds,
            )
            if state is None:
                self._stop_worker(process, process_group_ready=True)
                raise AudioNormalizationTimeout("Audio normalization timed out.")

            process.join(self._termination_grace_seconds)
            if process.is_alive():
                self._stop_worker(process, process_group_ready=True)
                raise AudioNormalizationError(
                    "Audio normalization worker did not exit."
                )
            if state[0] != "ok" or process.exitcode != 0:
                raise AudioNormalizationError("Audio normalization failed.")

            try:
                return validate_normalized_wav(destination_path)
            except WavValidationError as exc:
                raise AudioNormalizationError(
                    "Audio normalization produced an invalid WAV."
                ) from exc
        except BaseException:
            if process.is_alive():
                self._stop_worker(
                    process,
                    process_group_ready=process_group_ready,
                )
            _remove_incomplete_output(destination_path)
            raise
        finally:
            receive_connection.close()
            send_connection.close()
            process.close()

    @staticmethod
    def _receive_state(
        connection: Connection,
        *,
        timeout_seconds: float,
    ) -> WorkerState | None:
        if not connection.poll(timeout_seconds):
            return None
        try:
            return connection.recv()
        except (EOFError, OSError):
            return ("error", "WorkerConnectionClosed")

    def _stop_worker(
        self,
        process: multiprocessing.Process,
        *,
        process_group_ready: bool,
    ) -> None:
        if not process.is_alive():
            process.join()
            return

        group_signalled = False
        if process_group_ready and process.pid is not None:
            try:
                if os.getpgid(process.pid) == process.pid:
                    os.killpg(process.pid, signal.SIGTERM)
                    group_signalled = True
            except OSError:
                pass

        if not group_signalled and process.is_alive():
            process.terminate()

        process.join(self._termination_grace_seconds)
        if not process.is_alive():
            return

        if group_signalled and process.pid is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
        else:
            process.kill()
        process.join()
