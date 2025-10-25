import pytest
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
    assert chunks[1].text == "a test str"
    assert chunks[2].text == "tring"


# ============================================================================
# Additional Character-Level Splitting Tests
# ============================================================================

def test_char_split_exact_fit():
    """Test when text exactly fits max_char."""
    text_splitter = TextSplitter(max_char=10, overlap=0, split_level="char")
    chunks = text_splitter.run("1234567890")

    assert len(chunks) == 1
    assert chunks[0].text == "1234567890"


def test_char_split_under_max():
    """Test when text is shorter than max_char."""
    text_splitter = TextSplitter(max_char=100, overlap=0, split_level="char")
    chunks = text_splitter.run("Short text")

    assert len(chunks) == 1
    assert chunks[0].text == "Short text"


def test_char_split_empty_string():
    """Test with empty string."""
    text_splitter = TextSplitter(max_char=10, overlap=0, split_level="char")
    chunks = text_splitter.run("")

    assert len(chunks) == 0


def test_char_split_large_overlap():
    """Test with overlap close to max_char."""
    text_splitter = TextSplitter(max_char=10, overlap=8, split_level="char")
    chunks = text_splitter.run("ABCDEFGHIJKLMNOPQRST")

    # Should still make progress even with large overlap
    assert len(chunks) == 6
    # Each chunk should be max_char length (except possibly last)
    assert len(chunks[0].text) == 10
    # Verify specific output
    ##########################1234567890
    assert chunks[0].text == "ABCDEFGHIJ"
    assert chunks[1].text == "CDEFGHIJKL"
    assert chunks[2].text == "EFGHIJKLMN"
    assert chunks[3].text == "GHIJKLMNOP"
    assert chunks[4].text == "IJKLMNOPQR"
    assert chunks[5].text == "KLMNOPQRST"


# ============================================================================
# Word-Level Splitting Tests
# ============================================================================

def test_word_split_basic():
    """Test basic word-level splitting."""
    text = "one two three four five six seven eight nine ten"
    text_splitter = TextSplitter(max_char=20, overlap=0, split_level="word")
    chunks = text_splitter.run(text)

    assert len(chunks) == 3
    # Verify no chunk exceeds max_char
    for chunk in chunks:
        assert len(chunk.text) <= 20
    # Verify specific output
    assert chunks[0].text == "one two three four"
    assert chunks[1].text == "five six seven eight"
    assert chunks[2].text == "nine ten"


def test_word_split_with_word_overlap():
    """Test word splitting with word-based overlap."""
    text = "apple banana cherry date elderberry fig grape"
    text_splitter = TextSplitter(max_char=30, overlap=10, split_level="word", min_overlap_words=1)
    chunks = text_splitter.run(text)

    assert len(chunks) == 3
    # Check that chunks are created with proper size limits
    for chunk in chunks:
        assert len(chunk.text) <= 40  # max_char + overlap buffer
    # Verify specific output
    assert chunks[0].text == "apple banana cherry date"
    assert chunks[1].text == "date elderberry fig grape"
    assert chunks[2].text == "grape"


def test_word_split_long_word_overflow():
    """Test word splitting when a single word exceeds max_char."""
    text = "short VeryLongWordThatExceedsMaxChar short"
    text_splitter = TextSplitter(max_char=20, overlap=0, split_level="word")
    chunks = text_splitter.run(text)

    # Long word should be handled by character splitting fallback
    assert len(chunks) == 4
    assert chunks[0].text == "short"
    assert chunks[1].text == "VeryLongWordThatExce"
    assert chunks[2].text == "edsMaxChar"
    assert chunks[3].text == "short"


def test_word_split_no_words():
    """Test word splitting with text that has no whitespace."""
    text = "NoSpacesInThisText"
    text_splitter = TextSplitter(max_char=10, overlap=0, split_level="word")
    chunks = text_splitter.run(text)

    # Should fall back to character splitting
    assert len(chunks) == 2
    assert chunks[0].text == "NoSpacesIn"
    assert chunks[1].text == "ThisText"


def test_word_split_min_overlap_words():
    """Test min_overlap_words parameter."""
    text = "a b c d e f g h i j k l m n o p"
    text_splitter = TextSplitter(max_char=20, overlap=5, split_level="word", min_overlap_words=2)
    chunks = text_splitter.run(text)

    assert len(chunks) == 2
    # Verify chunks are created
    for chunk in chunks:
        assert len(chunk.text) > 0
    # Verify specific output
    assert chunks[0].text == "a b c d e f g h i j"
    assert chunks[1].text == "i j k l m n o p"


# ============================================================================
# Phrase-Level Splitting Tests
# ============================================================================

def test_phrase_split_basic():
    """Test phrase-level splitting on sentences."""
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    text_splitter = TextSplitter(max_char=30, overlap=0, split_level="phrase")
    chunks = text_splitter.run(text)

    assert len(chunks) == 4
    for chunk in chunks:
        assert len(chunk.text) <= 30
    # Verify specific output
    assert chunks[0].text == "First sentence."
    assert chunks[1].text == "Second sentence."
    assert chunks[2].text == "Third sentence."
    assert chunks[3].text == "Fourth sentence."


def test_phrase_split_with_overlap():
    """Test phrase splitting with overlap."""
    text = "One! Two? Three. Four! Five?"
    text_splitter = TextSplitter(max_char=20, overlap=5, split_level="phrase", min_overlap_words=1)
    chunks = text_splitter.run(text)

    assert len(chunks) == 3
    assert chunks[0].text == "One! Two? Three."
    assert chunks[1].text == "Three. Four! Five?"
    assert chunks[2].text == "Five?"


def test_phrase_split_no_delimiters():
    """Test phrase splitting with no sentence delimiters."""
    text = "This is all one long phrase without any sentence endings"
    text_splitter = TextSplitter(max_char=20, overlap=0, split_level="phrase")
    chunks = text_splitter.run(text)

    # Should fall back to word splitting
    assert len(chunks) == 3
    assert chunks[0].text == "This is all one long"
    assert chunks[1].text == "phrase without any"
    assert chunks[2].text == "sentence endings"


# ============================================================================
# Paragraph-Level Splitting Tests
# ============================================================================

def test_paragraph_split_basic():
    """Test paragraph-level splitting."""
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    text_splitter = TextSplitter(max_char=30, overlap=0, split_level="paragraph")
    chunks = text_splitter.run(text)

    assert len(chunks) == 3
    assert chunks[0].text == "First paragraph."
    assert chunks[1].text == "Second paragraph."
    assert chunks[2].text == "Third paragraph."


def test_paragraph_split_windows_newlines():
    """Test paragraph splitting with Windows-style newlines."""
    text = "Paragraph one.\r\n\r\nParagraph two.\r\n\r\nParagraph three."
    text_splitter = TextSplitter(max_char=30, overlap=0, split_level="paragraph")
    chunks = text_splitter.run(text)

    assert len(chunks) == 2
    assert chunks[0].text == "Paragraph one. Paragraph two."
    assert chunks[1].text == "Paragraph three."


def test_paragraph_split_single_paragraph():
    """Test paragraph splitting with no paragraph breaks."""
    text = "This is all one paragraph with no breaks"
    text_splitter = TextSplitter(max_char=20, overlap=0, split_level="paragraph")
    chunks = text_splitter.run(text)

    # Should fall back to word splitting
    assert len(chunks) == 3
    assert chunks[0].text == "This is all one"
    assert chunks[1].text == "paragraph with no"
    assert chunks[2].text == "breaks"


def test_paragraph_split_with_overlap():
    """Test paragraph splitting with overlap."""
    text = "Para one.\n\nPara to.\n\nPara three.\n\nParameter seven.\n\nPara forty.\n\nParameter sixteen."
    text_splitter = TextSplitter(max_char=20, overlap=5, split_level="paragraph", min_overlap_words=1)
    chunks = text_splitter.run(text)

    assert len(chunks) == 8
    ##########################12345678901234567890
    assert chunks[0].text == "Para one. Para"
    assert chunks[1].text == "Para to. Para"
    assert chunks[2].text == "Para three. Paramete"
    assert chunks[3].text == "Parameter seven."
    assert chunks[4].text == "seven. Para"
    assert chunks[5].text == "Para forty. Paramete"
    assert chunks[6].text == "Parameter sixteen."
    assert chunks[7].text == "sixteen."


# ============================================================================
# Edge Cases and Special Scenarios
# ============================================================================

def test_single_character():
    """Test with single character input."""
    text_splitter = TextSplitter(max_char=10, overlap=0, split_level="char")
    chunks = text_splitter.run("a")

    assert len(chunks) == 1
    assert chunks[0].text == "a"


def test_special_characters():
    """Test with special characters."""
    text = "Hello! @#$% World? {test} [brackets]"
    text_splitter = TextSplitter(max_char=15, overlap=0, split_level="word")
    chunks = text_splitter.run(text)

    assert len(chunks) > 0


def test_unicode_characters():
    """Test with Unicode characters."""
    text = "Hello 世界 🌍 café"
    text_splitter = TextSplitter(max_char=10, overlap=0, split_level="char")
    chunks = text_splitter.run(text)

    assert len(chunks) > 0


def test_multiple_consecutive_spaces():
    """Test with multiple consecutive spaces."""
    text = "word1    word2     word3"
    text_splitter = TextSplitter(max_char=15, overlap=0, split_level="word")
    chunks = text_splitter.run(text)

    assert len(chunks) > 0


def test_only_whitespace():
    """Test with only whitespace."""
    text = "     \n\n\t\t   "
    text_splitter = TextSplitter(max_char=10, overlap=0, split_level="char")
    chunks = text_splitter.run(text)

    assert len(chunks) >= 1


def test_max_char_equals_one():
    """Test with max_char=1."""
    text_splitter = TextSplitter(max_char=1, overlap=0, split_level="char")
    chunks = text_splitter.run("abc")

    assert len(chunks) == 3
    assert chunks[0].text == "a"
    assert chunks[1].text == "b"
    assert chunks[2].text == "c"


def test_overlap_equals_max_char():
    """Test when overlap equals max_char."""
    text_splitter = TextSplitter(max_char=10, overlap=10, split_level="char")
    chunks = text_splitter.run("ABCDEFGHIJKLMNOPQRST")

    # Should still make progress (step = 1)
    assert len(chunks) > 1


def test_overlap_greater_than_max_char():
    """Test when overlap is greater than max_char."""
    text_splitter = TextSplitter(max_char=10, overlap=15, split_level="char")
    chunks = text_splitter.run("ABCDEFGHIJKLMNOPQRST")

    # Should still make progress (step = 1)
    assert len(chunks) > 1


# ============================================================================
# Overlap Logic Tests
# ============================================================================

def test_overlap_appends_from_next_chunk():
    """Test that overlap appends text from next chunk to current."""
    text = "AAAA BBBB CCCC DDDD"
    text_splitter = TextSplitter(max_char=10, overlap=5, split_level="word", min_overlap_words=1)
    chunks = text_splitter.run(text)

    assert len(chunks) == 4
    assert chunks[0].text == "AAAA BBBB"
    assert chunks[1].text == "BBBB CCCC"
    assert chunks[2].text == "CCCC DDDD"
    assert chunks[3].text == "DDDD"


def test_overlap_with_insufficient_words():
    """Test overlap behavior when there aren't enough words."""
    text = "A B C D E F"
    text_splitter = TextSplitter(max_char=6, overlap=3, split_level="word", min_overlap_words=5)
    chunks = text_splitter.run(text)

    # Should fall back to character-based overlap when word count < min_overlap_words
    assert len(chunks) == 3
    assert chunks[0].text == "A B C "
    assert chunks[1].text == "C D E "
    assert chunks[2].text == "E F"


def test_overlap_last_chunk_has_no_suffix():
    """Test that the last chunk doesn't have overlap suffix."""
    text = "one two three four five"
    text_splitter = TextSplitter(max_char=15, overlap=5, split_level="word", min_overlap_words=1)
    chunks = text_splitter.run(text)

    assert len(chunks) == 3
    assert chunks[0].text == "one two three"
    assert chunks[1].text == "three four five"
    assert chunks[2].text == "five"


def test_overlap_word_vs_char_mode():
    """Test overlap switches between word and character mode."""
    # Case 1: Enough words - should use word-based overlap
    text1 = "apple banana cherry date elderberry"
    splitter1 = TextSplitter(max_char=25, overlap=10, split_level="word", min_overlap_words=2)
    chunks1 = splitter1.run(text1)

    assert len(chunks1) > 0

    # Case 2: Not enough words - should use char-based overlap
    text2 = "ab"
    splitter2 = TextSplitter(max_char=5, overlap=3, split_level="word", min_overlap_words=5)
    chunks2 = splitter2.run(text2)

    assert len(chunks2) >= 1


# ============================================================================
# Metadata Preservation Tests
# ============================================================================

def test_metadata_preserved():
    """Test that chunk metadata is preserved."""
    text_splitter = TextSplitter(max_char=10, overlap=0, split_level="char")
    chunks = text_splitter.run("Test text here")

    # All chunks should have metadata dict (even if empty)
    for chunk in chunks:
        assert hasattr(chunk, 'metadata')
        assert isinstance(chunk.metadata, dict)


def test_chunks_have_unique_ids():
    """Test that each chunk has a unique ID."""
    text_splitter = TextSplitter(max_char=10, overlap=0, split_level="char")
    chunks = text_splitter.run("This is a longer test string")

    ids = [chunk.id for chunk in chunks]
    assert len(ids) == len(set(ids))  # All IDs should be unique


# ============================================================================
# Error Handling Tests
# ============================================================================

def test_invalid_split_level():
    """Test that invalid split_level raises ValueError."""
    with pytest.raises(ValueError, match="Invalid split_level"):
        TextSplitter(max_char=10, overlap=0, split_level="invalid")


def test_non_string_input():
    """Test that non-string input raises TypeError."""
    text_splitter = TextSplitter(max_char=10, overlap=0, split_level="char")

    with pytest.raises(TypeError, match="TextSplitter expects a string input"):
        text_splitter.run(123)


def test_non_string_input_list():
    """Test that list input raises TypeError."""
    text_splitter = TextSplitter(max_char=10, overlap=0, split_level="char")

    with pytest.raises(TypeError, match="TextSplitter expects a string input"):
        text_splitter.run(["not", "a", "string"])


# ============================================================================
# Integration/Complex Scenarios
# ============================================================================

def test_real_world_paragraph():
    """Test with a realistic paragraph."""
    text = """
    Natural language processing (NLP) is a subfield of linguistics, computer science,
    and artificial intelligence concerned with the interactions between computers and
    human language. In particular, it focuses on how to program computers to process
    and analyze large amounts of natural language data. The goal is a computer capable
    of understanding the contents of documents, including the contextual nuances of
    the language within them.
    """
    text_splitter = TextSplitter(max_char=100, overlap=20, split_level="phrase", min_overlap_words=3)
    chunks = text_splitter.run(text.strip())

    assert len(chunks) > 0
    # Verify chunks are created
    assert chunks[0].text == ("Natural language processing (NLP) is a "
                           "subfield of linguistics, computer science, and artificial")
    assert chunks[1].text == ("science, and artificial intelligence concerned "
                              "with the interactions between computers and human")
    assert chunks[2].text == ("computers and human language. In particular, it "
                              "focuses on how to program computers to process and")
    assert chunks[3].text == ("In particular, it focuses on how to program computers "
                              "to process and analyze large amounts of")
    assert chunks[4].text == ("large amounts of natural language data. The goal is "
                              "a computer capable of understanding the contents")
    assert chunks[5].text == ("The goal is a computer capable of understanding the "
                              "contents of documents, including the contextual")
    assert chunks[6].text == "including the contextual nuances of the language within them."


def test_code_snippet():
    """Test splitting code-like text."""
    code = """def hello_world():
    print("Hello, World!")
    return True

def goodbye():
    print("Goodbye!")"""

    text_splitter = TextSplitter(max_char=50, overlap=10, split_level="paragraph", min_overlap_words=2)
    chunks = text_splitter.run(code)

    assert len(chunks) > 0
    assert chunks[0].text == '''def hello_world(): print("Hello, World!") return'''
    assert chunks[1].text == '''World!") return True def goodbye():'''
    assert chunks[2].text == '''def goodbye():
    print("Goodbye!")'''


def test_mixed_delimiters():
    """Test text with mixed sentence delimiters."""
    text = "Question? Answer! Statement. Another question? Final statement."
    text_splitter = TextSplitter(max_char=25, overlap=5, split_level="phrase", min_overlap_words=1)
    chunks = text_splitter.run(text)

    assert len(chunks) == 4
    ##########################1234567890123456789012345
    assert chunks[0].text == "Question? Answer! Stateme"
    assert chunks[1].text == "Statement. Another"
    assert chunks[2].text == "Another question? Final"
    assert chunks[3].text == "Final statement."


def test_very_long_text():
    """Test with very long text."""
    text = " ".join(["word"] * 1000)  # 1000 words
    text_splitter = TextSplitter(max_char=100, overlap=10, split_level="word", min_overlap_words=2)
    chunks = text_splitter.run(text)

    assert len(chunks) > 10
    # Verify chunks are created
    for chunk in chunks:
        assert len(chunk.text) > 0


def test_alternating_long_short_words():
    """Test with alternating long and short words."""
    text = "a supercalifragilisticexpialidocious b extraordinarily c d"
    text_splitter = TextSplitter(max_char=20, overlap=0, split_level="word")
    chunks = text_splitter.run(text)

    # Long words should be split at character level
    assert len(chunks) == 5
    ##########################12345678901234567890
    assert chunks[0].text == "a"
    assert chunks[1].text == "supercalifragilistic"
    assert chunks[2].text == "expialidocious"
    assert chunks[3].text == "b extraordinarily c"
    assert chunks[4].text == "d"
