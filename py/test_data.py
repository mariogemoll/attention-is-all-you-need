import struct

from data import from_uint16_le_bytes, to_uint16_le_bytes


def test_to_uint16_le_bytes() -> None:
    """Test converting integers to little-endian 16-bit bytes."""
    values = [100, 200, 300]
    result = to_uint16_le_bytes(values)
    expected = struct.pack(f"<{len(values)}H", *values)
    assert result == expected


def test_from_uint16_le_bytes() -> None:
    """Test converting little-endian 16-bit bytes to integers."""
    values = [1000, 2000, 3000]
    bytes_data = struct.pack(f"<{len(values)}H", *values)
    result = from_uint16_le_bytes(bytes_data)
    assert result == values


def test_round_trip_conversion() -> None:
    """Test that converting to bytes and back preserves the original values."""
    original = [42, 123, 456, 789, 1024]
    bytes_data = to_uint16_le_bytes(original)
    recovered = from_uint16_le_bytes(bytes_data)
    assert recovered == original


def test_empty_list() -> None:
    """Test handling of empty lists."""
    empty_list: list[int] = []
    bytes_data = to_uint16_le_bytes(empty_list)
    assert bytes_data == b""
    recovered = from_uint16_le_bytes(bytes_data)
    assert recovered == empty_list


def test_single_value() -> None:
    """Test handling of single values."""
    single_value = [65535]  # Max value for 16-bit unsigned int
    bytes_data = to_uint16_le_bytes(single_value)
    recovered = from_uint16_le_bytes(bytes_data)
    assert recovered == single_value
