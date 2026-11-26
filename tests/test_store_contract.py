"""The store contract (design §8), the same tests on every store: `InMemoryStore` and `PgStore`."""

import asyncio
from collections.abc import Awaitable, Callable

import pytest

from lite_context.errors import SeqConflict, UnknownTranscript
from lite_context.store import InMemoryStore, Record, Store


def rec(seq: int, text: str = "x") -> Record:
    return Record(seq, "message", {"text": text})


StoreFactory = Callable[[], Store]


@pytest.fixture(params=["memory", "pg"])
def make_store(request: pytest.FixtureRequest) -> StoreFactory:
    if request.param == "memory":
        return InMemoryStore
    factory: StoreFactory = request.getfixturevalue("pg_store_factory")
    return factory


def run(make_store: StoreFactory, body: Callable[[Store], Awaitable[None]]) -> None:
    async def main() -> None:
        store = make_store()
        try:
            await body(store)
        finally:
            await store.aclose()

    asyncio.run(main())


def test_appended_records_read_back_in_order(make_store: StoreFactory) -> None:
    async def body(store: Store) -> None:
        await store.create("t1")
        await store.append("t1", [rec(1, "a"), rec(2, "b")], expected_seq=0)
        await store.append("t1", [rec(3, "c")], expected_seq=2)
        assert await store.read("t1") == [rec(1, "a"), rec(2, "b"), rec(3, "c")]
        assert await store.read("t1", after=2) == [rec(3, "c")]
        assert await store.tail("t1") == 3

    run(make_store, body)


def test_creating_twice_is_harmless_and_transcripts_are_separate(make_store: StoreFactory) -> None:
    async def body(store: Store) -> None:
        await store.create("t1")
        await store.append("t1", [rec(1)], expected_seq=0)
        await store.create("t1")
        await store.create("t2")
        assert (await store.tail("t1"), await store.tail("t2")) == (1, 0)
        assert await store.read("t2") == []

    run(make_store, body)


def test_a_wrong_expected_seq_writes_nothing(make_store: StoreFactory) -> None:
    async def body(store: Store) -> None:
        await store.create("t1")
        await store.append("t1", [rec(1)], expected_seq=0)
        for expected in (0, 2):
            with pytest.raises(SeqConflict) as caught:
                await store.append(
                    "t1", [rec(expected + 1), rec(expected + 2)], expected_seq=expected
                )
            assert (caught.value.expected, caught.value.actual) == (expected, 1)
        assert await store.read("t1") == [rec(1)]

    run(make_store, body)


def test_a_batch_whose_seqs_do_not_follow_the_tail_is_refused_whole(
    make_store: StoreFactory,
) -> None:
    async def body(store: Store) -> None:
        await store.create("t1")
        for bad in ([rec(2)], [rec(1), rec(3)], [rec(1), rec(1)], []):
            with pytest.raises(ValueError, match=r"seq|at least one"):
                await store.append("t1", bad, expected_seq=0)
        assert await store.read("t1") == []

    run(make_store, body)


def test_an_unknown_transcript_is_refused(make_store: StoreFactory) -> None:
    async def body(store: Store) -> None:
        with pytest.raises(UnknownTranscript):
            await store.append("nope", [rec(1)], expected_seq=0)
        with pytest.raises(UnknownTranscript):
            await store.read("nope")
        with pytest.raises(UnknownTranscript):
            await store.tail("nope")

    run(make_store, body)


def test_of_appends_at_the_same_tail_exactly_one_succeeds(make_store: StoreFactory) -> None:
    async def body(store: Store) -> None:
        await store.create("t1")
        await store.append("t1", [rec(1)], expected_seq=0)
        # Eight writers at once, so that some of them overlap inside the database.
        outcomes = await asyncio.gather(
            *(store.append("t1", [rec(2, f"w{i}")], expected_seq=1) for i in range(8)),
            return_exceptions=True,
        )
        names = sorted(type(o).__name__ for o in outcomes)
        assert names == ["NoneType"] + ["SeqConflict"] * 7
        winner = f"w{outcomes.index(None)}"
        assert [r.data["text"] for r in await store.read("t1")] == ["x", winner]

    run(make_store, body)


def test_records_are_copies_the_caller_cannot_change_afterwards(make_store: StoreFactory) -> None:
    async def body(store: Store) -> None:
        await store.create("t1")
        data: dict[str, object] = {"text": "a", "list": [1]}
        await store.append("t1", [Record(1, "message", data)], expected_seq=0)
        data["text"] = "changed"
        read = await store.read("t1")
        read[0].data["list"].append(2)
        assert (await store.read("t1"))[0].data == {"text": "a", "list": [1]}

    run(make_store, body)


def test_object_keys_come_back_in_their_order(make_store: StoreFactory) -> None:
    # The adapters give these back to the vendor byte for byte: a reordered block would miss a
    # prompt-cache prefix.
    async def body(store: Store) -> None:
        await store.create("t1")
        data = {"type": "image", "source": {"z": 1, "a": 2}, "cache_control": {"type": "x"}}
        await store.append("t1", [Record(1, "message", data)], expected_seq=0)
        back = (await store.read("t1"))[0].data
        assert list(back) == ["type", "source", "cache_control"]
        assert list(back["source"]) == ["z", "a"]

    run(make_store, body)
