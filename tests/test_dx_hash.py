from __future__ import annotations

import uuid

from dissect.extfs import extfs

SEED = "16fd2706-8baf-433b-82eb-8c7fada847da"
SEED_ARR = extfs.c_ext.uint32_t[4](uuid.UUID(SEED).bytes)
NAME = b"hash-browns-recipe.txt"


def test_hash_legacy() -> None:
    """Hash generated with: debugfs -R "dx_hash -h legacy -s $SEED $NAME"."""
    assert extfs._dx_hash(NAME, extfs.c_ext.DX_HASH_LEGACY, SEED_ARR) == 0x684F907E


def test_hash_half_md4() -> None:
    """Hash generated with: debugfs -R "dx_hash -h half_md4 -s $SEED $NAME"."""
    assert extfs._dx_hash(NAME, extfs.c_ext.DX_HASH_HALF_MD4, SEED_ARR) == 0xF419E412


def test_hash_tea() -> None:
    """Hash generated with: debugfs -R "dx_hash -h tea -s $SEED $NAME"."""
    assert extfs._dx_hash(NAME, extfs.c_ext.DX_HASH_TEA, SEED_ARR) == 0xD66C67F0
