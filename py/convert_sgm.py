import sys
from xml.dom import minidom


def unescape(text: str) -> str:
    return text.replace("&amp;", "&")


def get_data(doc_nodes):  # type: ignore
    data = {}
    for doc in doc_nodes:
        docid = doc.attributes["docid"].value
        segments = {}
        for segment in doc.getElementsByTagName("seg"):
            segments[segment.attributes["id"].value] = segment.firstChild.data
        data[docid] = segments
    return data


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(f"Usage: python {sys.argv[0]} src_input tgt_input src_output tgt_output")
        sys.exit(1)

    src_input = minidom.parse(sys.argv[1])
    tgt_input = minidom.parse(sys.argv[2])
    src_doc_nodes = src_input.getElementsByTagName("doc")
    tgt_doc_nodes = tgt_input.getElementsByTagName("doc")
    src = get_data(src_doc_nodes)  # type: ignore
    tgt = get_data(tgt_doc_nodes)  # type: ignore

    assert src.keys() == tgt.keys()

    count = 0
    with open(sys.argv[3], "w") as src_output, open(sys.argv[4], "w") as tgt_output:
        for docid, segments in src.items():
            for segid, text in segments.items():
                count += 1
                src_output.write(unescape(text) + "\n")
                tgt_output.write(unescape(tgt[docid][segid]) + "\n")

    print(f"{count} segments.")
