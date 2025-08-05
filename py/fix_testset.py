import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_testset.py <input_file> <output_file>")
        sys.exit(1)

    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]
    with open(input_file_name, "rb") as input_file:
        with open(output_file_name, "wb") as output_file:
            for line_number, line in enumerate(input_file):
                if line_number == 506:
                    print("Replacing")
                    print(line[:-1].decode("utf-8"))
                    line = line.replace(b"\xc2\xb4", b" ")
                    print("with")
                    print(line[:-1].decode("utf-8"))
                output_file.write(line)
