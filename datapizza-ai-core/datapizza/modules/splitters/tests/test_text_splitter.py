from datapizza.modules.splitters.text_splitter import TextSplitter


def test_text_splitter():
    text_splitter = TextSplitter(max_char=10, overlap=0)
    chunks = text_splitter.run("This is a test string")
    assert len(chunks) == 3
    assert chunks[0].text == "This is a "
    assert chunks[1].text == "test strin"
    assert chunks[2].text == "g"


def test_text_splitter_with_overlap():
    text_splitter = TextSplitter(max_char=10, overlap=2)
    chunks = text_splitter.run("This is a test string")
    assert len(chunks) == 3

    assert chunks[0].text == "This is a "
    assert chunks[0].metadata.get("start_char") == 0
    assert chunks[0].metadata.get("end_char") == 10
    assert chunks[1].text == "a test str"
    assert chunks[1].metadata.get("start_char") == 8
    assert chunks[1].metadata.get("end_char") == 18
    assert chunks[2].text == "tring"
