import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input_file> <output_file>")
        sys.exit(1)

    _, input_file_name, output_file_name = sys.argv

    # read the file into memory completely
    with open(input_file_name, "rb") as in_file:
        text = in_file.read()

    # replace CR LF with LF
    text = text.replace(b"\x0d\x0a", b"\x0a")

    text = text.replace(b"\x0d", b"")

    text = text.replace(b"\xe2\x80\xa8", b"")

    with open(output_file_name, "wb") as out_file:
        out_file.write(text)
