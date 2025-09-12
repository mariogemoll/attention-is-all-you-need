import struct


def to_uint16_le_bytes(values: list[int]) -> bytes:
    """Convert a list of integers to little-endian 16-bit unsigned integer bytes."""
    return struct.pack(f"<{len(values)}H", *values)


def from_uint16_le_bytes(data: bytes) -> list[int]:
    """Convert little-endian 16-bit unsigned integer bytes to a list of integers."""
    count = len(data) // 2
    return list(struct.unpack(f"<{count}H", data))
