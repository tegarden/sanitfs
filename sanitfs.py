#!/usr/bin/env python3
"""
sanitfs.py — read-only FUSE mirror that strips selected characters from path
components in the presented view.

Policy:
- Sanitization: remove any character in STRIP_CHARS from each path component.
- Collisions: if two (or more) backing names sanitize to the same presented name
  within the same directory, that presented name is hidden (not listed) and lookup
  for it fails with ENOENT.
- Read-only: all mutating operations are denied with EROFS.

Usage:
  sanitfs.py /path/to/original 'chars-to-filter' /path/to/remount

Notes:
- Requires: fusepy (pip install fusepy) and FUSE enabled on the system.
- This is a "live view": it reflects backing changes, with a small per-directory
  cache to reduce overhead.
"""

import errno
import os
import stat
import sys
import time
from collections import defaultdict
from typing import Dict, Optional, Tuple

from fuse import FUSE, FuseOSError, LoggingMixIn, Operations


# Cache TTL (seconds) for directory mappings
CACHE_TTL = 2.0


class SanitFS(LoggingMixIn, Operations):
    """
    A read-only FUSE filesystem that mirrors a backing directory and sanitizes
    filename components in the presented view.
    """

    def __init__(self, backing_root: str, strip_chars: str):
        self.root = os.path.realpath(backing_root)
        self.strip_chars = set(strip_chars)

        # dir_cache maps: backing_dir_realpath -> (timestamp, map_presented_to_realname)
        # The mapping stores ONLY non-colliding presented names.
        self._dir_cache: Dict[str, Tuple[float, Dict[str, str]]] = {}

    # ------------------------
    # Helpers
    # ------------------------

    def _sanitize_component(self, name: str) -> str:
        # Strip only; do not normalize whitespace, do not collapse duplicates.
        return "".join(ch for ch in name if ch not in self.strip_chars)

    def _full_path(self, presented_path: str) -> str:
        """
        Resolve a presented FUSE path to a backing full path, translating each
        component via directory mappings.
        """
        if presented_path == "/":
            return self.root

        parts = [p for p in presented_path.split("/") if p]
        cur_backing = self.root

        for comp in parts:
            mapping = self._get_dir_mapping(cur_backing)
            real_name = mapping.get(comp)
            if real_name is None:
                # Not found, or hidden due to collision
                raise FuseOSError(errno.ENOENT)
            cur_backing = os.path.join(cur_backing, real_name)

        return cur_backing

    def _get_dir_mapping(self, backing_dir: str) -> Dict[str, str]:
        """
        Return mapping for a backing directory:
          presented_name -> real_name
        excluding any names that collide after sanitization.
        """
        now = time.time()
        cached = self._dir_cache.get(backing_dir)
        if cached is not None:
            ts, mapping = cached
            if (now - ts) <= CACHE_TTL:
                return mapping

        try:
            entries = os.listdir(backing_dir)
        except FileNotFoundError:
            raise FuseOSError(errno.ENOENT)
        except NotADirectoryError:
            raise FuseOSError(errno.ENOTDIR)
        except PermissionError:
            raise FuseOSError(errno.EACCES)

        # Build: sanitized -> [real1, real2, ...]
        buckets: Dict[str, list] = defaultdict(list)
        for real in entries:
            # Keep "." and ".." out (os.listdir doesn't include them, but being explicit is fine)
            if real in (".", ".."):
                continue
            presented = self._sanitize_component(real)
            # If sanitization yields empty, it becomes effectively unaddressable; treat as collision/hidden.
            if presented == "":
                buckets[presented].append(real)
            else:
                buckets[presented].append(real)

        mapping: Dict[str, str] = {}
        for presented, reals in buckets.items():
            # Hidden on collision, including empty presented name.
            if presented == "" or len(reals) != 1:
                continue
            mapping[presented] = reals[0]

        self._dir_cache[backing_dir] = (now, mapping)
        return mapping

    def _deny_ro(self):
        raise FuseOSError(errno.EROFS)

    # ------------------------
    # Read-only ops
    # ------------------------

    def access(self, path, mode):
        full = self._full_path(path)
        if not os.access(full, mode):
            raise FuseOSError(errno.EACCES)

    def getattr(self, path, fh=None):
        full = self._full_path(path)
        try:
            st = os.lstat(full)
        except FileNotFoundError:
            raise FuseOSError(errno.ENOENT)

        return dict(
            (key, getattr(st, key))
            for key in (
                "st_atime",
                "st_ctime",
                "st_gid",
                "st_mode",
                "st_mtime",
                "st_nlink",
                "st_size",
                "st_uid",
            )
        )

    def readlink(self, path):
        full = self._full_path(path)
        target = os.readlink(full)
        # Do not attempt to "re-sanitize" symlink targets; present verbatim.
        return target

    def readdir(self, path, fh):
        full_dir = self._full_path(path)

        yield "."
        yield ".."

        # We must list only non-colliding presented names.
        mapping = self._get_dir_mapping(full_dir)
        # Stable-ish order for usability
        for presented in sorted(mapping.keys()):
            yield presented

    def open(self, path, flags):
        full = self._full_path(path)

        # Enforce read-only regardless of mount flags.
        accmode = flags & os.O_ACCMODE
        if accmode != os.O_RDONLY:
            raise FuseOSError(errno.EROFS)

        # Also disallow truncation
        if flags & os.O_TRUNC:
            raise FuseOSError(errno.EROFS)

        return os.open(full, flags)

    def read(self, path, size, offset, fh):
        os.lseek(fh, offset, os.SEEK_SET)
        return os.read(fh, size)

    def release(self, path, fh):
        return os.close(fh)

    def statfs(self, path):
        full = self._full_path(path)
        stv = os.statvfs(full)
        return dict(
            (key, getattr(stv, key))
            for key in (
                "f_bavail",
                "f_bfree",
                "f_blocks",
                "f_bsize",
                "f_favail",
                "f_ffree",
                "f_files",
                "f_flag",
                "f_frsize",
                "f_namemax",
            )
        )

    # Optional: expose xattrs if desired
    def getxattr(self, path, name, position=0):
        full = self._full_path(path)
        try:
            return os.getxattr(full, name)
        except OSError as e:
            if e.errno in (errno.ENODATA, errno.EOPNOTSUPP):
                return b""
            raise

    def listxattr(self, path):
        full = self._full_path(path)
        try:
            return os.listxattr(full)
        except OSError as e:
            if e.errno in (errno.EOPNOTSUPP,):
                return []
            raise

    # ------------------------
    # Deny mutating ops (RO)
    # ------------------------

    def chmod(self, path, mode): self._deny_ro()
    def chown(self, path, uid, gid): self._deny_ro()
    def create(self, path, mode, fi=None): self._deny_ro()
    def mkdir(self, path, mode): self._deny_ro()
    def mknod(self, path, mode, dev): self._deny_ro()
    def rename(self, old, new): self._deny_ro()
    def rmdir(self, path): self._deny_ro()
    def symlink(self, name, target): self._deny_ro()
    def link(self, target, name): self._deny_ro()
    def unlink(self, path): self._deny_ro()
    def write(self, path, data, offset, fh): self._deny_ro()
    def truncate(self, path, length, fh=None): self._deny_ro()
    def utimens(self, path, times=None): self._deny_ro()
    def setxattr(self, path, name, value, options, position=0): self._deny_ro()
    def removexattr(self, path, name): self._deny_ro()


def main(argv: list) -> int:
    if len(argv) != 4:
        print(f"Usage: {argv[0]} BACKING_DIR STRIP_CHARS MOUNTPOINT", file=sys.stderr)
        return 2

    backing = argv[1]
    strip_chars = argv[2]
    mountpoint = argv[3]

    if not os.path.isdir(backing):
        print(f"Error: backing dir not found: {backing}", file=sys.stderr)
        return 2

    if not os.path.isdir(mountpoint):
        print(f"Error: mountpoint not found: {mountpoint}", file=sys.stderr)
        return 2

    # foreground: easier to debug; remove foreground=True to daemonize
    # ro: signals intent, but we also enforce RO in operations.
    # allow_other: optional; requires user_allow_other in /etc/fuse.conf
    fuse_options = {
        "foreground": False,
        "ro": True,
        "nothreads": True,
    }

    FUSE(SanitFS(backing, strip_chars), mountpoint, **fuse_options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

