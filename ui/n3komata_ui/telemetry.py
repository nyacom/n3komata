"""Low-overhead, best-effort host and BC-250 telemetry for the local UI.

The GPU activity values are derived from GRBM_STATUS flags, in the same
general way as radeontop. They are system-wide activity indicators, not CU
occupancy, per-process utilization, or a measure of useful FLOP throughput.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any


_BC250_PCI_DEVICE = "0x13fe"
# libdrm expects the absolute register address after the GC IP base is applied,
# not the 0x0da4 IP-local address from gc_10_3_0_offset.h. GRBM_STATUS is
# exposed as 0x8010 by the amdgpu info-register interface (dword offset 0x2004).
_GRBM_STATUS_DWORD_OFFSET = 0x8010 // 4
_GUI_ACTIVE_MASK = 0x80000000
_SPI_BUSY_MASK = 0x00400000


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="ascii").strip(), 0)
    except (OSError, ValueError):
        return None


def _find_bc250() -> tuple[Path, str] | None:
    for entry in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        if not re.fullmatch(r"card\d+", entry.name):
            continue
        device_dir = entry / "device"
        vendor = _read_int(device_dir / "vendor")
        device = _read_int(device_dir / "device")
        if vendor == 0x1002 and device == int(_BC250_PCI_DEVICE, 16):
            return entry, f"1002:{device:04x}"
    return None


class _AmdgpuRegisterReader:
    """Read the BC-250 GRBM status through libdrm's public amdgpu API."""

    def __init__(self, card: Path):
        library_name = ctypes.util.find_library("drm_amdgpu")
        if not library_name:
            raise RuntimeError("libdrm_amdgpu is not installed")
        self._lib = ctypes.CDLL(library_name, use_errno=True)
        self._initialize = self._lib.amdgpu_device_initialize
        self._initialize.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._initialize.restype = ctypes.c_int
        self._deinitialize = self._lib.amdgpu_device_deinitialize
        self._deinitialize.argtypes = [ctypes.c_void_p]
        self._deinitialize.restype = ctypes.c_int
        self._read_registers = self._lib.amdgpu_read_mm_registers
        self._read_registers.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self._read_registers.restype = ctypes.c_int

        device_node = Path("/dev/dri") / card.name
        self._fd = os.open(device_node, os.O_RDWR | getattr(os, "O_CLOEXEC", 0))
        self._handle = ctypes.c_void_p()
        major, minor = ctypes.c_uint32(), ctypes.c_uint32()
        result = self._initialize(self._fd, ctypes.byref(major), ctypes.byref(minor),
                                  ctypes.byref(self._handle))
        if result != 0:
            os.close(self._fd)
            self._fd = -1
            raise OSError(-result, os.strerror(-result), str(device_node))

    def read_status(self) -> int:
        value = ctypes.c_uint32()
        result = self._read_registers(
            self._handle,
            _GRBM_STATUS_DWORD_OFFSET,
            1,
            0xFFFFFFFF,
            0,
            ctypes.byref(value),
        )
        if result != 0:
            raise OSError(-result, os.strerror(-result), "GRBM_STATUS")
        return value.value

    def close(self) -> None:
        if self._handle:
            self._deinitialize(self._handle)
            self._handle = ctypes.c_void_p()
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1


class SystemTelemetrySampler:
    """Poll host counters in the background and expose a one-second snapshot."""

    def __init__(self, interval_s: float = 0.1):
        self._interval_s = interval_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: deque[dict[str, Any]] = deque(maxlen=120)
        self._previous_cpu: tuple[int, int] | None = None
        self._previous_disk: dict[str, int] = {}
        self._reader: _AmdgpuRegisterReader | None = None
        self._gpu_error: str | None = None
        self._gpu: dict[str, str] | None = None
        self._gtt_used_path: Path | None = None
        self._gtt_total_path: Path | None = None
        self._gpu_temp_input_path: Path | None = None
        self._gpu_temp_label = "temperature"
        self._gpu_power_input_path: Path | None = None
        self._gpu_power_label = "power"
        self._nvme_names: list[str] = []
        self._system = {
            "hostname": platform.node(),
            "kernel": platform.release(),
            "architecture": platform.machine(),
        }

        card = _find_bc250()
        if card is None:
            self._gpu_error = "BC-250 DRM card not found"
        else:
            card_path, pci_id = card
            self._gpu = {"card": card_path.name, "pci_id": pci_id}
            device_dir = card_path / "device"
            used = device_dir / "mem_info_gtt_used"
            total = device_dir / "mem_info_gtt_total"
            self._gtt_used_path = used if used.is_file() else None
            self._gtt_total_path = total if total.is_file() else None
            self._find_gpu_hwmon_sensors(device_dir)
            try:
                self._reader = _AmdgpuRegisterReader(card_path)
            except (OSError, RuntimeError, AttributeError) as exc:
                self._gpu_error = f"GPU activity unavailable: {type(exc).__name__}: {exc}"

        self._nvme_names = sorted(
            path.name for path in Path("/sys/block").glob("nvme*n*")
            if (path / "stat").is_file()
        )

    @staticmethod
    def _read_label(path: Path) -> str | None:
        try:
            label = path.read_text(encoding="ascii").strip()
        except OSError:
            return None
        return label or None

    def _find_gpu_hwmon_sensors(self, device_dir: Path) -> None:
        """Find temperature and PPT power sensors without assuming hwmon numbering."""
        for hwmon_dir in sorted(device_dir.glob("hwmon/hwmon*")):
            if self._read_label(hwmon_dir / "name") != "amdgpu":
                continue

            temperatures: list[tuple[str, Path]] = []
            for input_path in sorted(hwmon_dir.glob("temp*_input")):
                label_path = input_path.with_name(input_path.name.removesuffix("_input") + "_label")
                label = self._read_label(label_path) or input_path.stem.removesuffix("_input")
                temperatures.append((label, input_path))
            if temperatures:
                label, path = next(
                    (item for item in temperatures if item[0].lower() == "edge"), temperatures[0]
                )
                self._gpu_temp_label = label
                self._gpu_temp_input_path = path

            power_paths = sorted(hwmon_dir.glob("power*_average"))
            if not power_paths:
                power_paths = sorted(hwmon_dir.glob("power*_input"))
            if power_paths:
                path = power_paths[0]
                label_path = path.with_name(path.name.removesuffix("_average").removesuffix("_input") + "_label")
                self._gpu_power_label = self._read_label(label_path) or "power"
                self._gpu_power_input_path = path
            return

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="n3k-telemetry", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._reader is not None:
            self._reader.close()

    @staticmethod
    def _cpu_counters() -> tuple[int, int] | None:
        try:
            fields = Path("/proc/stat").read_text(encoding="ascii").splitlines()[0].split()[1:]
            values = [int(value) for value in fields]
        except (OSError, ValueError, IndexError):
            return None
        if len(values) < 5:
            return None
        total = sum(values[:8])
        idle = values[3] + values[4]
        return total, idle

    @staticmethod
    def _memory() -> dict[str, int] | None:
        values: dict[str, int] = {}
        try:
            for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
                key, raw = line.split(":", 1)
                if key in {"MemTotal", "MemAvailable"}:
                    values[key] = int(raw.strip().split()[0]) * 1024
        except (OSError, ValueError, IndexError):
            return None
        if "MemTotal" not in values or "MemAvailable" not in values:
            return None
        return {
            "total_bytes": values["MemTotal"],
            "available_bytes": values["MemAvailable"],
            "used_bytes": values["MemTotal"] - values["MemAvailable"],
        }

    def _disk_reads(self, elapsed: float | None) -> tuple[int | None, dict[str, int]]:
        counters: dict[str, int] = {}
        for name in self._nvme_names:
            try:
                fields = (Path("/sys/block") / name / "stat").read_text(encoding="ascii").split()
                sectors_read = int(fields[2])
            except (OSError, ValueError, IndexError):
                continue
            counters[name] = sectors_read * 512
        if elapsed is None or not counters or not self._previous_disk:
            return None, counters
        delta = sum(max(0, value - self._previous_disk.get(name, value))
                    for name, value in counters.items())
        return int(delta / max(elapsed, 1e-6)), counters

    def _sample(self) -> dict[str, Any]:
        now = time.monotonic()
        cpu = self._cpu_counters()
        cpu_percent = None
        if cpu is not None and self._previous_cpu is not None:
            total_delta = cpu[0] - self._previous_cpu[0]
            idle_delta = cpu[1] - self._previous_cpu[1]
            if total_delta > 0:
                cpu_percent = round(100.0 * max(0, total_delta - idle_delta) / total_delta, 1)
        self._previous_cpu = cpu

        disk_rate, disk_counters = self._disk_reads(
            now - self._samples[-1]["time"] if self._samples else None
        )
        self._previous_disk = disk_counters

        gpu_flags: tuple[bool, bool] | None = None
        if self._reader is not None:
            try:
                status = self._reader.read_status()
                gpu_flags = (bool(status & _GUI_ACTIVE_MASK), bool(status & _SPI_BUSY_MASK))
                self._gpu_error = None
            except OSError as exc:
                self._gpu_error = f"GPU activity read failed: {type(exc).__name__}: {exc}"

        gtt_used = _read_int(self._gtt_used_path) if self._gtt_used_path else None
        gtt_total = _read_int(self._gtt_total_path) if self._gtt_total_path else None
        gpu_temp_raw = _read_int(self._gpu_temp_input_path) if self._gpu_temp_input_path else None
        gpu_power_raw = _read_int(self._gpu_power_input_path) if self._gpu_power_input_path else None
        return {
            "time": now,
            "cpu_percent": cpu_percent,
            "memory": self._memory(),
            "gtt": {"used_bytes": gtt_used, "total_bytes": gtt_total},
            "nvme_read_bytes_per_second": disk_rate,
            "nvme_devices": list(self._nvme_names),
            "gpu_flags": gpu_flags,
            "gpu_temperature_c": round(gpu_temp_raw / 1000.0, 1) if gpu_temp_raw is not None else None,
            "gpu_power_w": round(gpu_power_raw / 1_000_000.0, 1) if gpu_power_raw is not None else None,
        }

    def _run(self) -> None:
        next_sample = time.monotonic()
        while not self._stop.is_set():
            sample = self._sample()
            with self._lock:
                self._samples.append(sample)
            next_sample += self._interval_s
            self._stop.wait(max(0.0, next_sample - time.monotonic()))

    def snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            samples = [sample for sample in self._samples if now - sample["time"] <= 1.25]
        latest = samples[-1] if samples else None
        gpu_samples = [sample["gpu_flags"] for sample in samples if sample["gpu_flags"] is not None]
        gpu_data: dict[str, Any] = {
            **(self._gpu or {}),
            "source": "amdgpu GRBM_STATUS",
            "sample_hz": round(1 / self._interval_s, 1),
            "temperature_c": latest["gpu_temperature_c"] if latest else None,
            "temperature_label": self._gpu_temp_label,
            "power_w": latest["gpu_power_w"] if latest else None,
            "power_label": self._gpu_power_label,
            "gui_active_percent": (round(100 * sum(flags[0] for flags in gpu_samples) / len(gpu_samples), 1)
                                   if gpu_samples else None),
            "spi_busy_percent": (round(100 * sum(flags[1] for flags in gpu_samples) / len(gpu_samples), 1)
                                 if gpu_samples else None),
            "error": self._gpu_error,
        }
        return {
            "sample_time": time.time(),
            "system": self._system,
            "gpu": gpu_data,
            "cpu_percent": latest["cpu_percent"] if latest else None,
            "memory": latest["memory"] if latest else None,
            "gtt": latest["gtt"] if latest else {"used_bytes": None, "total_bytes": None},
            "nvme_read_bytes_per_second": latest["nvme_read_bytes_per_second"] if latest else None,
            "nvme_devices": self._nvme_names,
        }
