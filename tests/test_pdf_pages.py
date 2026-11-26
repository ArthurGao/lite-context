"""A PDF's page count, measured once at conversion (design §3.1): also when the page objects sit
in compressed object streams (PDF 1.5+), and never at the cost of unbounded memory."""

import base64
import time
import zlib

from lite_context.adapters._common import MAX_INFLATED, _inflated, pdf_pages


def b64(body: bytes) -> str:
    return base64.b64encode(body).decode()


def object_stream(content: bytes, *, compress: bool = True) -> bytes:
    data = zlib.compress(content) if compress else content
    head = b"5 0 obj << /Type /ObjStm /N 3 /First 12 /Filter /FlateDecode /Length %d >>" % len(data)
    return head + b"\nstream\n" + data + b"\nendstream\nendobj\n"


def test_visible_page_objects_are_counted_as_before() -> None:
    body = b"%PDF-1.4\n" + b"<< /Type /Page >>\n" * 4 + b"<< /Type /Pages /Count 4 >>\n"
    assert pdf_pages(b64(body)) == 4


def test_pages_inside_a_compressed_object_stream_are_counted() -> None:
    body = b"%PDF-1.7\n" + object_stream(b"<< /Type /Page >> " * 6) + b"%%EOF\n"
    assert b"/Type /Page" not in body  # nothing to see without decompressing
    assert pdf_pages(b64(body)) == 6


def test_the_page_tree_count_is_used_when_no_page_object_is_visible() -> None:
    # The leaves are in a stream compressed some other way; the root /Pages dictionary says 9.
    body = b"%PDF-1.7\n1 0 obj << /Count 9 /Kids [2 0 R] /Type /Pages >> endobj\n"
    body += object_stream(b"\x00\x01garbage", compress=False)
    assert pdf_pages(b64(body)) == 9


def test_an_outline_count_is_not_taken_for_pages() -> None:
    body = b"%PDF-1.7\n1 0 obj << /Type /Outlines /Count 40 >> endobj\n"
    assert pdf_pages(b64(body)) is None


def test_a_stream_that_does_not_decompress_is_skipped() -> None:
    body = b"%PDF-1.7\n" + object_stream(b"not zlib at all", compress=False)
    body += object_stream(b"<< /Type /Page >> " * 2)
    assert pdf_pages(b64(body)) == 2


def test_a_decompression_bomb_stops_at_the_cap_and_gives_no_count() -> None:
    bomb = zlib.compress(b"\x00" * (100 * 1024 * 1024), 9)  # 100 MiB of zeros, about 100 KiB
    body = b"%PDF-1.7\n5 0 obj << /Filter /FlateDecode >>\nstream\n" + bomb + b"\nendstream\n"
    started = time.monotonic()
    assert pdf_pages(b64(body)) is None
    assert time.monotonic() - started < 10  # bounded work, not a hang
    assert len(_inflated(body)) == MAX_INFLATED  # it stopped at the cap, not at 100 MiB


def test_what_is_not_a_pdf_gives_no_count() -> None:
    assert pdf_pages(b64(b"just text")) is None
    assert pdf_pages("%%%not base64") is None


# --- from the v0.3.2 review: inputs that made the first version slow or wrong ---


def fast(body: bytes) -> int | None:
    started = time.monotonic()
    pages = pdf_pages(b64(body))
    assert time.monotonic() - started < 2, "page counting must stay linear"
    return pages


def test_many_stream_keywords_without_an_end_are_not_quadratic() -> None:
    assert fast(b"%PDF-1.7\n" + b"stream\n" * 40_000) is None


def test_a_dictionary_repeating_the_page_tree_type_is_not_quadratic() -> None:
    assert fast(b"%PDF-1.7\n<<" + b"/Type /Pages " * 20_000 + b"<<") is None


def test_a_page_tree_with_a_nested_dictionary_still_gives_its_count() -> None:
    after = b"1 0 obj << /Type /Pages /Resources << /Font 2 0 R >> /Count 5 /Kids [] >> endobj\n"
    before = b"1 0 obj << /Type /Pages /Count 7 /Resources << >> >> endobj\n"
    assert fast(b"%PDF-1.7\n" + after) == 5
    assert fast(b"%PDF-1.7\n" + before) == 7


def test_a_stream_without_a_line_break_before_endstream_does_not_swallow_the_next() -> None:
    first = (
        b"5 0 obj << /Filter /FlateDecode >>\nstream\n" + zlib.compress(b"junk") + b"endstream\n"
    )
    second = object_stream(b"<< /Type /Page >> " * 3)
    assert fast(b"%PDF-1.7\n" + first + second) == 3


def test_the_word_stream_in_text_does_not_hide_the_real_stream_after_it() -> None:
    body = b"%PDF-1.7\n% text that mentions stream words endstream\n" + object_stream(
        b"<< /Type /Page >> " * 6
    )
    assert fast(body) == 6


def test_a_count_before_a_nested_dictionary_that_precedes_the_type_is_found() -> None:
    body = b"%PDF-1.7\n1 0 obj << /Count 5 /Resources << /Font 2 0 R >> /Type /Pages >> endobj\n"
    assert fast(body) == 5


# --- from the second review ---


def test_many_closing_marks_near_each_page_tree_type_are_not_quadratic() -> None:
    node = b"<<" + b" " * 2000 + b">>" * 1000 + b"/Type /Pages"
    assert fast(b"%PDF-1.7\n" + object_stream(node * 1000)) is None
    assert fast(b"%PDF-1.7\n" + (b"/Type /Pages" + b"<<" * 1900 + b" " * 200) * 200) is None


def test_a_count_inside_a_nested_dictionary_is_not_the_page_count() -> None:
    body = b"%PDF-1.7\n1 0 obj << /Type /Pages /Resources << /Count 99 >> /Count 5 >> endobj\n"
    assert fast(body) == 5


def test_a_word_ending_in_stream_is_not_a_stream_start() -> None:
    body = b"%PDF-1.7\n% upstream\n" + object_stream(b"<< /Type /Page >> " * 10)
    assert fast(body) == 10
