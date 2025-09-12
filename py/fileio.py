import struct
from typing import BinaryIO


def read_uint32(file: BinaryIO) -> int:
    """Read a little-endian 32-bit unsigned integer from file."""
    result: int = struct.unpack("<I", file.read(4))[0]
    return result


def write_uint32(file: BinaryIO, value: int) -> None:
    """Write a little-endian 32-bit unsigned integer to file."""
    file.write(struct.pack("<I", value))


def read_uint8(file: BinaryIO) -> int:
    """Read an 8-bit unsigned integer from file."""
    result: int = struct.unpack("<B", file.read(1))[0]
    return result


def write_uint8(file: BinaryIO, value: int) -> None:
    """Write an 8-bit unsigned integer to file."""
    file.write(struct.pack("<B", value))


def read_uint16_array(file: BinaryIO, count: int) -> list[int]:
    """Read an array of little-endian 16-bit unsigned integers from file."""
    return list(struct.unpack(f"<{count}H", file.read(count * 2)))


def write_uint16_array(file: BinaryIO, values: list[int]) -> None:
    """Write an array of little-endian 16-bit unsigned integers to file."""
    file.write(struct.pack(f"<{len(values)}H", *values))
