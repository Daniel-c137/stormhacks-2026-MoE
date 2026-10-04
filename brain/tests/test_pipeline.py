"""The background runner, the write-up's dependencies and an interrupted write-up."""

import asyncio
import logging

import pytest
from api_support import ALEX, SARAH, TEAM
from fastapi import HTTPException
from pipeline_support import GatedLLM
from starlette.requests import Request

from brain.agent.pipeline import PipelineRunner, ReportPipeline
from brain.api.deps import get_llm, get_memory, get_pipeline
from brain.config import Settings
from brain.llm import GeminiEmbedder
from brain.main import create_app
from brain.memory import PgMemoryStore
from brain.report import ReportExtraction
from brain.store import InMemoryStore
from contracts import TranscriptSegment

pytestmark = pytest.mark.anyio


# the runner


async def test_the_runner_runs_one_write_up_per_meeting_at_a_time():
    runner = PipelineRunner()
    gate = asyncio.Event()
    runs: list[str] = []

    async def run(meeting_id: str) -> None:
        runs.append(meeting_id)
        await gate.wait()

    assert runner.start("m1", run)
    assert not runner.start("m1", run)
    assert runner.start("m2", run)
    assert runner.running("m1") and runner.running("m2")

    gate.set()
    await runner.drain()

    assert runs == ["m1", "m2"]
    assert not runner.running("m1")
    assert runner.start("m1", run)  # once finished, the meeting can run again
    await runner.drain()
    assert runs == ["m1", "m2", "m1"]


async def test_a_failed_write_up_is_logged_and_drain_still_returns(caplog):
    runner = PipelineRunner()

    async def boom(meeting_id: str) -> None:
        raise RuntimeError("model fell over")

    with caplog.at_level(logging.WARNING):
        runner.start("m1", boom)
        await runner.drain()

    assert "m1" in caplog.text and "model fell over" in caplog.text
    assert not runner.running("m1")


async def test_the_app_keeps_one_runner():
    app = create_app()

    assert isinstance(app.state.pipeline_runner, PipelineRunner)


# an interrupted run


async def test_an_interrupted_write_up_says_so_and_saves_nothing():
    store = InMemoryStore(teams=[TEAM], people=[ALEX, SARAH])
    meeting = await store.create_meeting(TEAM.id, "Refund sync", ALEX.id)
    await store.add_segments(
        meeting.id,
        [
            TranscriptSegment(
                seg_id="s-1",
                meeting_id=meeting.id,
                speaker_id=SARAH.id,
                speaker_name=SARAH.name,
                text="I'll refund the users.",
                is_final=True,
                t_start=0,
                t_end=3,
            )
        ],
    )
    await store.transition_status(meeting.id, {"live"}, "processing")
    llm = GatedLLM(structured={ReportExtraction: ReportExtraction(summary="Refunds.")})
    pipeline = ReportPipeline(store, lambda: llm)

    task = asyncio.create_task(pipeline.run(meeting.id))
    await llm.called.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    progress = await store.report_progress(meeting.id)
    assert progress is not None and "interrupted" in (progress.error or "")
    assert (await store.meeting(meeting.id)).status == "processing"


# dependencies


def request_for(app) -> Request:
    return Request({"type": "http", "app": app})


EMBEDDINGS = {
    "gemini_api_key": "test-key",
    "gemini_embedding_model": "embedding-model",
    "gemini_embedding_dim": 768,
}


def test_memory_needs_both_the_database_pool_and_embeddings():
    app = create_app()
    request = request_for(app)
    configured = Settings(_env_file=None, **EMBEDDINGS)

    assert get_memory(request, configured) is None  # no database pool yet

    app.state.db_pool = object()
    memory = get_memory(request, configured)
    assert memory is not None
    assert isinstance(memory.store, PgMemoryStore) and memory.store.pool is app.state.db_pool
    assert isinstance(memory.embedder, GeminiEmbedder)

    assert get_memory(request, Settings(_env_file=None, gemini_api_key=None)) is None


def test_memory_with_vectors_the_database_cannot_hold_is_unavailable():
    app = create_app()
    app.state.db_pool = object()
    wrong = Settings(_env_file=None, **{**EMBEDDINGS, "gemini_embedding_dim": 1536})

    assert get_memory(request_for(app), wrong) is None


async def test_get_llm_is_a_503_when_gemini_is_not_configured():
    with pytest.raises(HTTPException) as raised:
        await get_llm(Settings(_env_file=None, gemini_api_key=None, gemini_model=None))

    assert raised.value.status_code == 503
    assert "Gemini is not configured" in raised.value.detail


def test_the_pipeline_waits_for_the_configured_settle_time():
    assert Settings(_env_file=None).pipeline_settle_seconds == 8

    pipeline = get_pipeline(
        store=InMemoryStore(),
        settings=Settings(_env_file=None, pipeline_settle_seconds=2.5),
        make_llm=lambda: None,
        memory=None,
    )

    assert isinstance(pipeline, ReportPipeline)
    assert pipeline.settle_seconds == 2.5
