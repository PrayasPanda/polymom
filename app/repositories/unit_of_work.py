"""Unit of Work: one session and transaction shared by every repository.

Multi-table writes (a stage result plus its utterances, speakers and search rows)
commit together or not at all. Leaving the context without ``commit()`` discards them.
"""

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.repositories.meeting_repository import MeetingRepository, SqlAlchemyMeetingRepository
from app.repositories.results_repository import ResultsRepository, SqlAlchemyResultsRepository
from app.repositories.search import SearchRepository, build_search_repository


class UnitOfWork:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.meetings: MeetingRepository = SqlAlchemyMeetingRepository(session)
        self.results: ResultsRepository = SqlAlchemyResultsRepository(session)
        self.search: SearchRepository = build_search_repository(session)

    async def commit(self) -> None:
        await self.session.commit()

    async def rollback(self) -> None:
        await self.session.rollback()


UnitOfWorkFactory = Callable[[], AbstractAsyncContextManager[UnitOfWork]]


def unit_of_work_factory(sessionmaker: async_sessionmaker[AsyncSession]) -> UnitOfWorkFactory:
    @asynccontextmanager
    async def _uow() -> AsyncIterator[UnitOfWork]:
        async with sessionmaker() as session:
            # Closing the session discards anything uncommitted without expiring loaded
            # objects, so entities read in a unit of work stay usable afterwards.
            yield UnitOfWork(session)

    return _uow
