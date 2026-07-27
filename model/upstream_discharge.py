#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime
import os
import struct
import time

import numpy as np

DEFAULT_TIMEOUT = 600.0                          # seconds
POLL_INTERVAL = 0.001                            # seconds

MAGIC = b"RAWSQV01"
HEADER_FORMAT = "<8siiii"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)     # 24
N_WRITTEN_OFFSET = 16

LAT_DTYPE = np.dtype("<f8")
LON_DTYPE = np.dtype("<f8")
VALUE_DTYPE = np.dtype("<f4")


def _points_size(n_points):
    return n_points * (LAT_DTYPE.itemsize + LON_DTYPE.itemsize)


def _record_size(n_points):
    return n_points * VALUE_DTYPE.itemsize


def _to_ordinal(date):
    if isinstance(date, str):
        date = datetime.datetime.strptime(date, "%Y-%m-%d")
    return date.toordinal()


class DischargeWriter(object):

    def __init__(self, path, lats, lons, start_date):
        lats = np.asarray(lats, dtype=LAT_DTYPE)
        lons = np.asarray(lons, dtype=LON_DTYPE)
        if lats.shape != lons.shape or lats.ndim != 1:
            raise ValueError("lats and lons must be matching 1-D arrays")

        self.path = path
        self.n_points = int(lats.size)
        self.start_ordinal = _to_ordinal(start_date)
        self.n_written = 0

        directory = os.path.dirname(os.path.abspath(path))
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)

        self._handle = open(path, "w+b")
        self._handle.write(struct.pack(HEADER_FORMAT, MAGIC, self.n_points,
                                       self.start_ordinal, 0, 0))
        self._handle.write(lats.tobytes())
        self._handle.write(lons.tobytes())
        self._handle.flush()

    def append(self, values):
        values = np.asarray(values, dtype=VALUE_DTYPE)
        if values.size != self.n_points:
            raise ValueError("expected %d value(s), got %d" % (self.n_points, values.size))

        self._handle.seek(0, os.SEEK_END)
        self._handle.write(values.tobytes())

        self.n_written += 1
        self._handle.seek(N_WRITTEN_OFFSET)
        self._handle.write(struct.pack("<i", self.n_written))
        self._handle.flush()

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class DischargeReader(object):
    def __init__(self, path, timeout = DEFAULT_TIMEOUT, poll = POLL_INTERVAL):
        self.path = path
        self.timeout = timeout
        self.poll = poll

        self._wait_until(lambda: os.path.isfile(path), "to be created")
        self._fd = os.open(path, os.O_RDONLY)

        raw = self._read_exactly(0, HEADER_SIZE, "its header")
        magic, n_points, start_ordinal, _, _ = struct.unpack(HEADER_FORMAT, raw)
        if magic != MAGIC:
            raise ValueError("%s is not a RAWS discharge file (magic %r)" % (path, magic))

        self.n_points = n_points
        self.start_ordinal = start_ordinal
        self._record_bytes = _record_size(n_points)
        self._data_offset = HEADER_SIZE + _points_size(n_points)

        coordinates = self._read_exactly(HEADER_SIZE, _points_size(n_points),
                                         "its point coordinates")
        split = n_points * LAT_DTYPE.itemsize
        self.lats = np.frombuffer(coordinates[:split], LAT_DTYPE)
        self.lons = np.frombuffer(coordinates[split:], LON_DTYPE)


    def _wait_until(self, ready, what):
        if ready():
            return
        deadline = time.monotonic() + self.timeout
        while not ready():
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "waited %g s for %s %s. The sub-basin that writes it is not making "
                    "progress -- it has most likely failed."
                    % (self.timeout, self.path, what))
            time.sleep(self.poll)

    def _read_exactly(self, offset, size, what):
        if size == 0:
            return b""
        self._wait_until(lambda: len(os.pread(self._fd, size, offset)) == size, what)
        return os.pread(self._fd, size, offset)

    def n_written(self):
        return struct.unpack("<i", os.pread(self._fd, 4, N_WRITTEN_OFFSET))[0]

    def first_date(self):
        return datetime.date.fromordinal(self.start_ordinal)

    def values_for(self, date):
        index = _to_ordinal(date) - self.start_ordinal
        if index < 0:
            raise IndexError("%s starts at %s but %s was requested"
                             % (self.path, self.first_date(),
                                date if isinstance(date, str) else date.isoformat()))

        self._wait_until(lambda: self.n_written() > index,
                         "day %s (record %d)" % (date, index))
        raw = os.pread(self._fd, self._record_bytes,
                       self._data_offset + index * self._record_bytes)
        return np.frombuffer(raw, VALUE_DTYPE)

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


class InflowCollection(object):
    def __init__(self, paths, lats, lons, tolerance = 1e-6, timeout = DEFAULT_TIMEOUT):
        self.shape = (len(lats), len(lons))
        self.readers = []
        self.rows = []
        self.cols = []

        for path in paths:
            reader = DischargeReader(path, timeout = timeout)
            rows = np.empty(reader.n_points, dtype = np.intp)
            cols = np.empty(reader.n_points, dtype = np.intp)
            for i in range(reader.n_points):
                rows[i] = self._locate(lats, reader.lats[i], "latitude", path, tolerance)
                cols[i] = self._locate(lons, reader.lons[i], "longitude", path, tolerance)
            self.readers.append(reader)
            self.rows.append(rows)
            self.cols.append(cols)

    @staticmethod
    def _locate(axis, value, what, path, tolerance):
        index = int(np.argmin(np.abs(axis - value)))
        if abs(axis[index] - value) > tolerance:
            raise ValueError(
                "%s: outlet %s %.6f does not fall on this clone (nearest cell centre is "
                "%.6f). The upstream sub-basin drains outside the receiving window."
                % (path, what, value, axis[index]))
        return index

    def field_for(self, date):
        """A full-window array holding each upstream outlet's discharge, zero elsewhere."""
        field = np.zeros(self.shape, dtype = np.float64)
        for reader, rows, cols in zip(self.readers, self.rows, self.cols):
            np.add.at(field, (rows, cols), reader.values_for(date))
        return field
