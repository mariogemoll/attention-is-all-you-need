import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} input output")
        sys.exit(1)

    count = 0
    with open(sys.argv[1], "r") as input, open(sys.argv[2], "w") as output:
        for line in input:
            if "&" in line:
                converted = line.replace("&", "&amp;")
                output.write(converted)
                count += 1
            else:
                output.write(line)

    print(f"Converted {count} lines.")
