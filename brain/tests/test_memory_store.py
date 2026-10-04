"""The same behaviour from the in-memory store and the pgvector store on a real Postgres."""

import pytest

from brain.llm import Embeddings
from brain.memory import Chunk, ChunkKind, InMemoryMemoryStore, MemoryStore, PgMemoryStore

pytestmark = pytest.mark.anyio

DIM = 768
MODEL = "embed-model"


@pytest.fixture(params=["in_memory", "postgres"])
def memory_store(request) -> MemoryStore:
    if request.param == "in_memory":
        return InMemoryMemoryStore(dim=DIM)
    return PgMemoryStore(request.getfixturevalue("memory_pool"), dim=DIM)


def vec(*weights: float) -> list[float]:
    """A DIM-sized vector whose first components are `weights`."""
    return [*weights, *([0.0] * (DIM - len(weights)))]


def chunk(
    id: str, *, team="t-1", meeting="m-1", kind: ChunkKind = "transcript", text=None, **fields
) -> Chunk:
    return Chunk(id=id, team_id=team, meeting_id=meeting, kind=kind, text=text or id, **fields)


def embedded(*vectors: list[float], model: str = MODEL) -> Embeddings:
    return Embeddings(model=model, vectors=list(vectors))


async def test_search_returns_the_nearest_chunks_first_with_their_scores(memory_store):
    await memory_store.index(
        [chunk("refund"), chunk("email"), chunk("mixed")],
        embedded(vec(1, 0), vec(0, 1), vec(1, 1)),
    )

    hits = await memory_store.search("t-1", vec(1, 0.1), model=MODEL, k=3)

    assert [h.chunk.id for h in hits] == ["refund", "mixed", "email"]
    assert hits[0].score == pytest.approx(0.995, abs=1e-3)
    assert hits[0].score > hits[1].score > hits[2].score


async def test_search_keeps_every_field_of_the_chunk(memory_store):
    original = chunk(
        "c-1",
        kind="decision",
        text="Bob: refunds Wednesday",
        speaker_id="p-bob",
        speaker_name="Bob Okafor",
        ref_id="d-1",
        t_start=6.0,
        t_end=14.5,
    )
    await memory_store.index([original], embedded(vec(1)))

    (hit,) = await memory_store.search("t-1", vec(1), model=MODEL)

    assert hit.chunk == original


async def test_search_never_crosses_teams(memory_store):
    await memory_store.index(
        [chunk("ours"), chunk("theirs", team="t-2", meeting="m-2")],
        embedded(vec(0, 1), vec(1, 0)),
    )

    hits = await memory_store.search("t-1", vec(1, 0), model=MODEL)

    assert [h.chunk.id for h in hits] == ["ours"]


async def test_search_returns_at_most_k(memory_store):
    await memory_store.index(
        [chunk(f"c-{i}") for i in range(5)], embedded(*[vec(1, i) for i in range(5)])
    )

    assert len(await memory_store.search("t-1", vec(1), model=MODEL, k=2)) == 2


async def test_search_only_compares_vectors_from_the_same_model(memory_store):
    await memory_store.index([chunk("old")], embedded(vec(1), model="other-model"))
    await memory_store.index([chunk("new")], embedded(vec(0, 1)))

    hits = await memory_store.search("t-1", vec(1), model=MODEL)

    assert [h.chunk.id for h in hits] == ["new"]


async def test_indexing_a_chunk_again_replaces_it(memory_store):
    await memory_store.index([chunk("c-1", text="first")], embedded(vec(1)))
    await memory_store.index([chunk("c-1", text="second")], embedded(vec(0, 1)))

    hits = await memory_store.search("t-1", vec(0, 1), model=MODEL)

    assert [(h.chunk.id, h.chunk.text) for h in hits] == [("c-1", "second")]
    assert hits[0].score == pytest.approx(1)


async def test_deleting_a_transcript_keeps_the_report_and_other_meetings(memory_store):
    await memory_store.index(
        [
            chunk("m1-turn"),
            chunk("m1-summary", kind="summary"),
            chunk("m1-decision", kind="decision"),
            chunk("m1-task", kind="task"),
            chunk("m2-turn", meeting="m-2"),
        ],
        embedded(*[vec(1, i) for i in range(5)]),
    )

    await memory_store.delete_meeting_transcript("m-1")

    hits = await memory_store.search("t-1", vec(1), model=MODEL, k=10)
    assert sorted(h.chunk.id for h in hits) == ["m1-decision", "m1-summary", "m1-task", "m2-turn"]


async def test_replacing_a_meeting_drops_its_stale_chunks_only(memory_store):
    await memory_store.index(
        [chunk("m1-old"), chunk("m1-kept"), chunk("m2-turn", meeting="m-2")],
        embedded(vec(1), vec(1), vec(1)),
    )

    await memory_store.replace_meeting(
        "m-1", [chunk("m1-kept", text="updated"), chunk("m1-new")], embedded(vec(1), vec(1))
    )

    hits = await memory_store.search("t-1", vec(1), model=MODEL, k=10)
    assert sorted((h.chunk.id, h.chunk.text) for h in hits) == [
        ("m1-kept", "updated"),
        ("m1-new", "m1-new"),
        ("m2-turn", "m2-turn"),
    ]


async def test_replacing_a_meeting_with_nothing_empties_it(memory_store):
    await memory_store.index([chunk("m1-turn")], embedded(vec(1)))

    await memory_store.replace_meeting("m-1", [], embedded())

    assert await memory_store.search("t-1", vec(1), model=MODEL) == []


async def test_vectors_must_match_the_chunks_and_the_dimensions(memory_store):
    with pytest.raises(ValueError, match="2 chunks"):
        await memory_store.index([chunk("a"), chunk("b")], embedded(vec(1)))
    with pytest.raises(ValueError, match=str(DIM)):
        await memory_store.index([chunk("a")], embedded([1.0, 0.0]))
    with pytest.raises(ValueError, match=str(DIM)):
        await memory_store.search("t-1", [1.0, 0.0], model=MODEL)


async def test_an_empty_store_finds_nothing(memory_store):
    assert await memory_store.search("t-1", vec(1), model=MODEL) == []


async def test_the_store_reports_its_dimensions(memory_store):
    assert memory_store.dim == DIM


async def test_the_in_memory_store_can_take_any_dimensions():
    assert InMemoryMemoryStore().dim is None
