from __future__ import annotations

import io
import logging
import os
import stat
from functools import cached_property, lru_cache
from typing import TYPE_CHECKING, BinaryIO
from uuid import UUID

from dissect.util import ts
from dissect.util.stream import RangeStream, RunlistStream

from dissect.extfs.c_ext import (
    EXT2,
    EXT3,
    EXT4,
    FILETYPES,
    XATTR_NAME_MAP,
    XATTR_PREFIX_MAP,
    c_ext,
)
from dissect.extfs.exceptions import (
    Error,
    FileNotFoundError,
    NotADirectoryError,
    NotASymlinkError,
)
from dissect.extfs.journal import JDB2

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from datetime import datetime

log = logging.getLogger(__name__)
log.setLevel(os.getenv("DISSECT_LOG_EXTFS", "CRITICAL"))


class ExtFS:
    def __init__(self, fh: BinaryIO):
        self.fh = fh

        fh.seek(c_ext.EXT2_SBOFF)
        sb = c_ext.ext4_super_block(fh)
        self.sb = sb

        if sb.s_magic != c_ext.EXT2_FS_MAGIC:
            raise Error("Not a valid ExtFS filesystem (magic mismatch)")

        if sb.s_inodes_count < 10:
            raise Error("Not a valid ExtFS filesystem (inum count < 10)")

        if sb.s_blocks_per_group == 0 or sb.s_inodes_per_group == 0:
            raise Error("Not a valid ExtFS filesystem (blocks or inodes per group is 0)")

        if sb.s_log_block_size != sb.s_log_cluster_size:
            raise NotImplementedError("Different size cluster than blocks is currently not supported")

        self.block_size = c_ext.EXT2_MIN_BLOCK_SIZE << sb.s_log_block_size
        if self.block_size == 0 or self.block_size % 512:
            raise Error("Not a valid ExtFS filesystem (invalid block size)")

        if sb.s_feature_incompat & c_ext.EXT4_FEATURE_INCOMPAT_EXTENTS:
            self.type = EXT4
        elif sb.s_feature_compat & c_ext.EXT3_FEATURE_COMPAT_HAS_JOURNAL:
            self.type = EXT3
        else:
            self.type = EXT2

        if sb.s_feature_incompat & c_ext.EXT2_FEATURE_INCOMPAT_FILETYPE:
            self._dirtype = c_ext.ext2_dir_entry_2
        else:
            self._dirtype = c_ext.ext2_dir_entry

        self.block_count = (sb.s_blocks_count_hi << 32) | sb.s_blocks_count_lo
        self.last_block = self.block_count - 1

        if (
            self.type == EXT4
            and self.sb.s_feature_incompat & c_ext.EXT4_FEATURE_INCOMPAT_64BIT
            and self.sb.s_desc_size >= 64
        ):
            self._group_desc_struct = c_ext.ext4_group_desc
        else:
            self._group_desc_struct = c_ext.ext2_group_desc
        self._group_desc_size = sb.s_desc_size if sb.s_desc_size else len(self._group_desc_struct)

        goff = c_ext.EXT2_SBOFF + self._group_desc_size
        self.groups_offset = goff if goff % self.block_size == 0 else goff + self.block_size - goff % self.block_size
        self.groups_count = ((self.last_block - sb.s_first_data_block) // sb.s_blocks_per_group) + 1

        self.uuid = UUID(bytes=sb.s_uuid)
        self.volume_name = sb.s_volume_name.split(b"\x00")[0].decode(errors="surrogateescape")
        self.last_mount = sb.s_last_mounted.split(b"\x00")[0].decode(errors="surrogateescape")

        self.root = self.get_inode(c_ext.EXT2_ROOT_INO, "/")

        self.get_inode = lru_cache(1024)(self.get_inode)
        self._read_group_desc = lru_cache(356)(self._read_group_desc)

    @cached_property
    def journal(self) -> JDB2:
        if not self.sb.s_feature_compat & c_ext.EXT3_FEATURE_COMPAT_HAS_JOURNAL:
            raise Error("Journal not supported")

        inum = self.sb.s_journal_inum
        if inum == 0:
            raise Error(f"Journal inum is 0, could be on external device (s_journal_uuid = {self.sb.s_journal_uuid})")

        inode = self.get_inode(inum)
        return JDB2(inode.open())

    def get(self, path_or_inum: str | int, node: INode | None = None) -> INode:
        if isinstance(path_or_inum, int):
            return self.get_inode(path_or_inum)

        node = node if node else self.root
        parts = path_or_inum.split("/")
        for part in parts:
            if not part:
                continue

            node = node.lookup(part)

            if node is None:
                raise FileNotFoundError(f"File not found: {path_or_inum}")

        return node

    def get_inode(
        self,
        inum: int,
        filename: str | None = None,
        filetype: int | None = None,
    ) -> INode:
        if inum < c_ext.EXT2_BAD_INO or inum > self.sb.s_inodes_count:
            raise Error(f"inum out of range {c_ext.EXT2_BAD_INO}-{self.sb.s_inodes_count}: {inum}")

        return INode(self, inum, filename, filetype)

    def _read_group_desc(self, group_num: int) -> c_ext.ext2_group_desc | c_ext.ext4_group_desc:
        if group_num >= self.groups_count:
            raise Error("Group number exceeds amount of groups")

        offset = self.groups_offset + group_num * self._group_desc_size
        self.fh.seek(offset)
        group_desc = self._group_desc_struct(self.fh)

        if self._group_desc_struct == c_ext.ext4_group_desc:
            block_bitmap = (group_desc.bg_block_bitmap_hi << 32) | group_desc.bg_block_bitmap_lo
            inode_bitmap = (group_desc.bg_inode_bitmap_hi << 32) | group_desc.bg_inode_bitmap_lo
            table_block = (group_desc.bg_inode_table_hi << 32) | group_desc.bg_inode_table_lo
        else:
            block_bitmap = group_desc.bg_block_bitmap_lo
            inode_bitmap = group_desc.bg_inode_bitmap_lo
            table_block = group_desc.bg_inode_table_lo

        if block_bitmap > self.last_block or inode_bitmap > self.last_block or table_block > self.last_block:
            raise Error("Group descriptor block locations exceed last block")

        return group_desc


class INode:
    def __init__(
        self,
        extfs: ExtFS,
        inum: int,
        filename: str | None = None,
        filetype: int | None = None,
    ):
        self.extfs = extfs
        self.inum = inum
        self.filename = filename
        self._filetype = filetype
        self._runlist = None

    def __repr__(self) -> str:
        return f"<inode {self.inum}>"

    @cached_property
    def inode(self) -> c_ext.ext4_inode:
        block_group_num, index = divmod(self.inum - 1, self.extfs.sb.s_inodes_per_group)
        block_group = self.extfs._read_group_desc(block_group_num)

        if self.extfs._group_desc_struct == c_ext.ext4_group_desc:
            table_block = (block_group.bg_inode_table_hi << 32) | block_group.bg_inode_table_lo
        else:
            table_block = block_group.bg_inode_table_lo

        offset = table_block * self.extfs.block_size + index * self.extfs.sb.s_inode_size
        self.extfs.fh.seek(offset)
        return c_ext.ext4_inode(self.extfs.fh)

    @cached_property
    def size(self) -> int:
        return (self.inode.i_size_high << 32) + self.inode.i_size_lo

    @property
    def filetype(self) -> int:
        if not self._filetype:
            self._filetype = stat.S_IFMT(self.inode.i_mode)
        return self._filetype

    @cached_property
    def link(self) -> str:
        if self.filetype != stat.S_IFLNK:
            raise NotASymlinkError(f"{self!r} is not a symlink")

        return self.open().read().decode(errors="surrogateescape")

    @cached_property
    def xattr(self) -> list[XAttr]:
        xattr = []

        if self.inode.i_extra.strip(b"\x00"):
            buf = io.BytesIO(self.inode.i_extra)
            hdr = c_ext.ext4_xattr_ibody_header(buf)
            if hdr.h_magic != c_ext.EXT4_XATTR_MAGIC:
                raise Error("Invalid xattr magic value")

            xattr.extend(_iter_xattr(self, buf, len(self.inode.i_extra), 4))

        if self.inode.i_file_acl_lo:
            block = (self.inode.i_file_acl_high << 32) | self.inode.i_file_acl_lo
            block_offset = block * self.extfs.block_size

            buf = RangeStream(self.extfs.fh, block_offset, self.extfs.block_size)
            hdr = c_ext.ext4_xattr_header(buf)
            if hdr.h_magic != c_ext.EXT4_XATTR_MAGIC:
                raise Error("Invalid xattr magic value")

            xattr.extend(_iter_xattr(self, buf, buf.size))

        return xattr

    @property
    def atime(self) -> datetime:
        return ts.from_unix_ns(self.atime_ns)

    @property
    def atime_ns(self) -> int:
        time = self.inode.i_atime
        time_extra = self.inode.i_atime_extra if self.extfs.sb.s_inode_size > 128 else 0

        return _parse_ns_ts(time, time_extra)

    @property
    def mtime(self) -> datetime:
        return ts.from_unix_ns(self.mtime_ns)

    @property
    def mtime_ns(self) -> int:
        time = self.inode.i_mtime
        time_extra = self.inode.i_mtime_extra if self.extfs.sb.s_inode_size > 128 else 0

        return _parse_ns_ts(time, time_extra)

    @property
    def ctime(self) -> datetime:
        return ts.from_unix_ns(self.ctime_ns)

    @property
    def ctime_ns(self) -> int:
        time = self.inode.i_ctime
        time_extra = self.inode.i_ctime_extra if self.extfs.sb.s_inode_size > 128 else 0

        return _parse_ns_ts(time, time_extra)

    @property
    def dtime(self) -> datetime:
        return ts.from_unix(self.inode.i_dtime)

    @property
    def crtime(self) -> datetime | None:
        time_ns = self.crtime_ns
        if time_ns is None:
            return None
        return ts.from_unix_ns(time_ns)

    @property
    def crtime_ns(self) -> int | None:
        if self.extfs.sb.s_inode_size <= 128:
            return None

        time = self.inode.i_crtime
        time_extra = self.inode.i_crtime_extra

        return _parse_ns_ts(time, time_extra)

    def listdir(self) -> dict[str, INode]:
        return {node.filename: node for node in self.iterdir()}

    dirlist = listdir

    def lookup(self, filename: str) -> INode | None:
        if self.inode.i_flags & c_ext.EXT4_INDEX_FL:
            entry = _htree_lookup(self, filename)

            if entry is not None:
                return entry

        for entry in self.iterdir():
            if entry.filename == filename:
                return entry

        return None

    def iterdir(self) -> Iterator[INode]:
        if self.filetype != stat.S_IFDIR:
            raise NotADirectoryError(f"{self!r} is not a directory")

        buf = self.open()

        return _iterdir_blocks(self, buf, self.size - 12)

    def dataruns(self) -> list[tuple[int | None, int]]:
        if not self._runlist:
            expected_runs = (self.size + self.extfs.block_size - 1) // self.extfs.block_size

            if self.inode.i_flags & c_ext.EXT4_EXTENTS_FL:
                buf = io.BytesIO(self.inode.i_block)

                runs = []
                run_offset = 0

                for extent in _parse_extents(self, buf):
                    # Account for uninitialized extents
                    if extent.ee_len > 0x8000:
                        uninitialized_gap = extent.ee_len - 0x8000
                        runs.append((None, uninitialized_gap))
                        run_offset += uninitialized_gap
                        continue

                    # Account for sparse gaps
                    if extent.ee_block != run_offset:
                        sparse_gap = extent.ee_block - run_offset
                        runs.append((None, sparse_gap))
                        run_offset += sparse_gap

                    runs.append(((extent.ee_start_hi << 32) | extent.ee_start_lo, extent.ee_len))
                    run_offset += extent.ee_len

                if run_offset < expected_runs:
                    runs.append((None, expected_runs - run_offset))

                self._runlist = runs
            else:
                i_blocks = c_ext.uint32[15](self.inode.i_block)
                num_blocks = (self.size + self.extfs.block_size - 1) // self.extfs.block_size
                num_direct_blocks = min(num_blocks, c_ext.EXT2_NDIR_BLOCKS)

                blocks = i_blocks[:num_direct_blocks]
                num_blocks -= num_direct_blocks

                if num_blocks > 0:
                    for level in range(c_ext.EXT2_NIND_BLOCKS):
                        indirect_offset = i_blocks[num_direct_blocks + level]
                        parsed_blocks = _parse_indirect(self, indirect_offset, num_blocks, level + 1)
                        num_blocks -= len(parsed_blocks)
                        blocks.extend(parsed_blocks)

                        if num_blocks == 0:
                            break

                runs = []
                if blocks:
                    run_offset = None
                    run_size = 1

                    for block in blocks:
                        if run_offset is None:
                            run_offset = block
                            continue

                        if block == run_offset + run_size:
                            run_size += 1
                        else:
                            if run_offset == 0:
                                runs.append((None, run_size))
                            else:
                                runs.append((run_offset, run_size))
                            run_offset = block
                            run_size = 1

                    runs.append((run_offset, run_size))

                self._runlist = runs

        return self._runlist

    def open(self) -> BinaryIO:
        if self.inode.i_flags & c_ext.EXT4_INLINE_DATA_FL or (self.filetype == stat.S_IFLNK and self.size < 60):
            buf = io.BytesIO(memoryview(self.inode.i_block)[: self.size])
            # Need to add a size attribute to maintain compatibility with dissect streams
            buf.size = self.size
            return buf
        return RunlistStream(self.extfs.fh, self.dataruns(), self.size, self.extfs.block_size)


class XAttr:
    def __init__(self, extfs: ExtFS, inode: INode, entry: c_ext.ext4_xattr_entry, value: bytes):
        self.extfs = extfs
        self.inode = inode
        self.entry = entry

        self.prefix = XATTR_PREFIX_MAP.get(entry.e_name_index, "unknown_prefix")
        self._name = XATTR_NAME_MAP.get(entry.e_name_index, entry.e_name.decode(errors="surrogateescape"))
        self.name = self.prefix + self._name
        self.value = value

    def __repr__(self) -> str:
        return f"<xattr name={self.name} value={self.value} inode={self.inode}>"


def _parse_indirect(inode: INode, offset: int, num_blocks: int, level: int) -> list[int]:
    offsets_per_block = inode.extfs.block_size // 4

    if level == 1:
        read_blocks = min(num_blocks, offsets_per_block)
        if offset == 0:
            return [0] * read_blocks
        inode.extfs.fh.seek(offset * inode.extfs.block_size)
        return c_ext.uint32[read_blocks](inode.extfs.fh)

    blocks = []

    max_level_blocks = offsets_per_block**level
    blocks_per_nest = max_level_blocks // offsets_per_block
    read_blocks = (num_blocks + blocks_per_nest - 1) // blocks_per_nest
    read_blocks = min(read_blocks, offsets_per_block)

    inode.extfs.fh.seek(offset * inode.extfs.block_size)
    for addr in c_ext.uint32[read_blocks](inode.extfs.fh):
        parsed_blocks = _parse_indirect(inode, addr, num_blocks, level - 1)
        num_blocks -= len(parsed_blocks)
        blocks.extend(parsed_blocks)

    return blocks


def _parse_extents(inode: INode, buf: bytes) -> Iterator[c_ext.ext4_extent]:
    extent_header = c_ext.ext4_extent_header(buf)

    if extent_header.eh_magic != 0xF30A:
        raise Error("Invalid extent_header magic")

    if extent_header.eh_depth == 0:
        for _ in range(extent_header.eh_entries):
            extent = c_ext.ext4_extent(buf)
            yield extent
    else:
        for _ in range(extent_header.eh_entries):
            idx = c_ext.ext4_extent_idx(buf)
            child = (idx.ei_leaf_hi << 32) | idx.ei_leaf_lo

            fh = inode.extfs.fh
            fh.seek(child * inode.extfs.block_size)
            blockbuf = io.BytesIO(fh.read(inode.extfs.block_size))
            yield from _parse_extents(inode, blockbuf)


def _iter_xattr(inode: INode, buf: BinaryIO, end: int, value_offset: int = 0) -> Iterator[XAttr]:
    offset = buf.tell()
    while True:
        try:
            if offset > end:
                break

            buf.seek(offset)
            entry = c_ext.ext4_xattr_entry(buf)

            if (entry.e_name_len, entry.e_name_index, entry.e_value_offs) == (0, 0, 0):
                break

            if entry.e_value_inum:
                value = inode.extfs.get_inode(entry.e_value_inum).open().read(entry.e_value_size)
            else:
                buf.seek(value_offset + entry.e_value_offs)
                value = buf.read(entry.e_value_size)

            yield XAttr(inode.extfs, inode, entry, value)

            offset += (len(entry) + c_ext.EXT4_XATTR_ROUND) & (~c_ext.EXT4_XATTR_ROUND & 0xFFFFFFFF)
        except EOFError:
            break


def _parse_ns_ts(time: int, time_extra: int) -> int:
    # The low 2 bits of time_extra are used to extend the time field
    # The remaining 30 bits are nanoseconds
    time |= (time_extra & 0b11) << 32
    ns = time_extra >> 2

    return (time * 1000000000) + ns


def _iterdir_blocks(inode: INode, buf: BinaryIO, size: int) -> Iterator[INode]:
    start = buf.tell()

    offset = 0

    while offset < size:
        direntry = inode.extfs._dirtype(buf)

        if direntry.rec_len == 0:
            log.critical("Zero-length directory entry in %s (offset 0x%x)", inode, offset)
            return

        # Sanity check if the direntry is valid
        if 0 < direntry.inode < inode.extfs.sb.s_inodes_count:
            fname = buf.read(direntry.name_len).decode(errors="surrogateescape")
            ftype = direntry.file_type if inode.extfs._dirtype == c_ext.ext2_dir_entry_2 else None

            if ftype:
                ftype = FILETYPES[ftype]

            yield inode.extfs.get_inode(direntry.inode, fname, ftype)

        offset += direntry.rec_len
        buf.seek(start + offset)


def _htree_lookup(inode: INode, filename: str) -> INode | None:
    if inode.filetype != stat.S_IFDIR:
        raise NotADirectoryError(f"{inode!r} is not a directory")

    buf = inode.open()

    for block in _htree_traverse_leafs(inode, buf, filename):
        offset = block * inode.extfs.block_size
        buf.seek(offset)

        scan_size = min(inode.extfs.block_size, inode.size - offset)

        for entry in _iterdir_blocks(inode, buf, scan_size):
            if entry.filename == filename:
                return entry

    return None


def _htree_traverse_leafs(inode: INode, buf: BinaryIO, filename: str) -> Iterator[int]:
    dx_root = c_ext.dx_root(buf)
    dx_entries = c_ext.dx_entry[dx_root.count - 1](buf)

    seed = c_ext.uint32[4](inode.extfs.sb.s_hash_seed)
    target_hash = _dx_hash(filename.encode(), dx_root.hash_version, seed)

    if target_hash is None:
        return None

    block_idx = _htree_next_block_index(dx_entries, target_hash)
    block = dx_entries[block_idx].block if block_idx is not None else dx_root.block
    stack = [(dx_root, dx_entries, block_idx)]

    for _ in range(dx_root.indirect_levels):
        buf.seek(block * inode.extfs.block_size)
        block = _htree_next_level(buf, stack, target_hash)

    yield block

    while True:
        next_sibling = _htree_next_sibling(buf, stack, inode.extfs.block_size)

        if next_sibling is not None:
            yield next_sibling
        else:
            return None


def _htree_next_sibling(buf: BinaryIO, stack: list[tuple], block_size: int) -> int | None:
    block = None
    levels = len(stack)

    while len(stack) > 0:
        dx_node, dx_entries, block_idx = stack.pop()

        next_idx = 0 if block_idx is None else block_idx + 1

        if next_idx >= len(dx_entries):
            continue

        next_hash = dx_entries[next_idx].hash

        if not next_hash & 1:
            return None

        stack.append((dx_node, dx_entries, next_idx))
        block = dx_entries[next_idx].block

        break
    else:
        return None

    while len(stack) < levels:
        buf.seek(block * block_size)

        dx_node = c_ext.dx_node(buf)
        dx_entries = c_ext.dx_entry[dx_node.count - 1](buf)

        block_idx = None
        block = dx_node.block

        stack.append((dx_node, dx_entries, block_idx))

    return block


def _htree_next_level(buf: BinaryIO, stack: list[tuple], target_hash: int) -> int:
    dx_node = c_ext.dx_node(buf)
    dx_entries = c_ext.dx_entry[dx_node.count - 1](buf)

    block_idx = _htree_next_block_index(dx_entries, target_hash)
    block = dx_entries[block_idx].block if block_idx is not None else dx_node.block

    stack.append((dx_node, dx_entries, block_idx))

    return block


def _htree_next_block_index(entries: list, target_hash: int) -> int | None:
    idx = None
    lo = 0
    hi = len(entries) - 1

    while lo <= hi:
        mid = (lo + hi + 1) // 2

        cmp_hash = entries[mid].hash

        if cmp_hash <= target_hash:
            lo = mid + 1
            idx = mid
        else:
            hi = mid - 1

    return idx


def _dx_hash_legacy(name: bytes, signed: bool) -> int:
    h0, h1 = 0x12A3FE2D, 0x37ABE8F9

    for b in name:
        if signed:
            b = b - 256 if b >= 128 else b

        h = h1 + (h0 ^ (b * 7152373))

        if h & 0x80000000:
            h -= 0x7FFFFFFF

        h1 = h0
        h0 = h

    return (h0 << 1) & 0xFFFFFFFF


def _str2hashbuf(msg: bytes, num: int, signed: bool) -> list[int]:
    buf = []
    length = len(msg)
    pad = length | (length << 8)
    pad = (pad | (pad << 16)) & 0xFFFFFFFF

    val = pad

    if length > num * 4:
        length = num * 4

    for i, b in enumerate(msg[:length]):
        if signed:
            b = b - 256 if b >= 128 else b

        val = (b + (val << 8)) & 0xFFFFFFFF

        if (i % 4) == 3:
            buf.append(val)
            val = pad
            num -= 1

    if length % 4 and num > 0:
        buf.append(val)
        num -= 1

    while num > 0:
        buf.append(pad)
        num -= 1

    return buf


def _rol32(word: int, shift: int) -> int:
    return (word << (shift & 31)) | (word >> ((-shift) & 31))


def _half_md4_round(fn: Callable[[int, int, int], int], a: int, b: int, c: int, d: int, x: int, s: int) -> int:
    a += fn(b, c, d) + x

    return _rol32(a & 0xFFFFFFFF, s) & 0xFFFFFFFF


def _half_md4_f(x: int, y: int, z: int) -> int:
    return z ^ (x & (y ^ z))


def _half_md4_g(x: int, y: int, z: int) -> int:
    return (x & y) + ((x ^ y) & z)


def _half_md4_h(x: int, y: int, z: int) -> int:
    return x ^ y ^ z


def _half_md4_transform(buf: list[int], inbuf: list[int]) -> None:
    k1 = 0
    k2 = 0x5A827999
    k3 = 0x6ED9EBA1

    a, b, c, d = buf

    a = _half_md4_round(_half_md4_f, a, b, c, d, inbuf[0] + k1, 3)
    d = _half_md4_round(_half_md4_f, d, a, b, c, inbuf[1] + k1, 7)
    c = _half_md4_round(_half_md4_f, c, d, a, b, inbuf[2] + k1, 11)
    b = _half_md4_round(_half_md4_f, b, c, d, a, inbuf[3] + k1, 19)
    a = _half_md4_round(_half_md4_f, a, b, c, d, inbuf[4] + k1, 3)
    d = _half_md4_round(_half_md4_f, d, a, b, c, inbuf[5] + k1, 7)
    c = _half_md4_round(_half_md4_f, c, d, a, b, inbuf[6] + k1, 11)
    b = _half_md4_round(_half_md4_f, b, c, d, a, inbuf[7] + k1, 19)

    a = _half_md4_round(_half_md4_g, a, b, c, d, inbuf[1] + k2, 3)
    d = _half_md4_round(_half_md4_g, d, a, b, c, inbuf[3] + k2, 5)
    c = _half_md4_round(_half_md4_g, c, d, a, b, inbuf[5] + k2, 9)
    b = _half_md4_round(_half_md4_g, b, c, d, a, inbuf[7] + k2, 13)
    a = _half_md4_round(_half_md4_g, a, b, c, d, inbuf[0] + k2, 3)
    d = _half_md4_round(_half_md4_g, d, a, b, c, inbuf[2] + k2, 5)
    c = _half_md4_round(_half_md4_g, c, d, a, b, inbuf[4] + k2, 9)
    b = _half_md4_round(_half_md4_g, b, c, d, a, inbuf[6] + k2, 13)

    a = _half_md4_round(_half_md4_h, a, b, c, d, inbuf[3] + k3, 3)
    d = _half_md4_round(_half_md4_h, d, a, b, c, inbuf[7] + k3, 9)
    c = _half_md4_round(_half_md4_h, c, d, a, b, inbuf[2] + k3, 11)
    b = _half_md4_round(_half_md4_h, b, c, d, a, inbuf[6] + k3, 15)
    a = _half_md4_round(_half_md4_h, a, b, c, d, inbuf[1] + k3, 3)
    d = _half_md4_round(_half_md4_h, d, a, b, c, inbuf[5] + k3, 9)
    c = _half_md4_round(_half_md4_h, c, d, a, b, inbuf[0] + k3, 11)
    b = _half_md4_round(_half_md4_h, b, c, d, a, inbuf[4] + k3, 15)

    buf[0] = (buf[0] + a) & 0xFFFFFFFF
    buf[1] = (buf[1] + b) & 0xFFFFFFFF
    buf[2] = (buf[2] + c) & 0xFFFFFFFF
    buf[3] = (buf[3] + d) & 0xFFFFFFFF


def _dx_hash_half_md4(name: bytes, buf: list[int], signed: bool) -> int:
    for i in range(0, len(name), 32):
        inbuf = _str2hashbuf(name[i:], 8, signed)
        _half_md4_transform(buf, inbuf)

    return buf[1]


def _tea_transform(buf: list[int], inbuf: list[int]) -> None:
    teasum = 0
    delta = 0x9E3779B9

    b0, b1 = buf[:2]
    a, b, c, d = inbuf[:4]

    for _ in range(16):
        teasum += delta

        b0 += ((b1 << 4) + a) ^ (b1 + teasum) ^ ((b1 >> 5) + b)
        b0 &= 0xFFFFFFFF

        b1 += ((b0 << 4) + c) ^ (b0 + teasum) ^ ((b0 >> 5) + d)
        b1 &= 0xFFFFFFFF

    buf[0] = (buf[0] + b0) & 0xFFFFFFFF
    buf[1] = (buf[1] + b1) & 0xFFFFFFFF


def _dx_hash_tea(name: bytes, buf: list[int], signed: bool) -> int:
    for i in range(0, len(name), 16):
        inbuf = _str2hashbuf(name[i:], 4, signed)
        _tea_transform(buf, inbuf)

    return buf[0]


def _dx_hash(name: bytes, hash_version: int, seed: list[int] | None = None) -> int | None:
    outhash = 0

    buf = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476] if seed is None or not any(seed) else [*seed]

    match hash_version:
        case c_ext.DX_HASH_LEGACY_UNSIGNED:
            outhash = _dx_hash_legacy(name, signed=False)
        case c_ext.DX_HASH_LEGACY:
            outhash = _dx_hash_legacy(name, signed=True)
        case c_ext.DX_HASH_HALF_MD4_UNSIGNED:
            outhash = _dx_hash_half_md4(name, buf, signed=False)
        case c_ext.DX_HASH_HALF_MD4:
            outhash = _dx_hash_half_md4(name, buf, signed=True)
        case c_ext.DX_HASH_TEA_UNSIGNED:
            outhash = _dx_hash_tea(name, buf, signed=False)
        case c_ext.DX_HASH_TEA:
            outhash = _dx_hash_tea(name, buf, signed=True)
        case c_ext.DX_HASH_SIPHASH | _:
            return None

    outhash &= 0xFFFFFFFE

    if outhash == (c_ext.EXT4_HTREE_EOF_32BIT << 1):
        outhash = (c_ext.EXT4_HTREE_EOF_32BIT - 1) << 1

    return outhash
