"""Create an API key. The plaintext is printed once and never stored.

uv run python -m scripts.create_api_key --label ci --owner alice
uv run python -m scripts.create_api_key --list
uv run python -m scripts.create_api_key --revoke pk_abc123...
"""

import argparse
import asyncio
import sys

from app.core.config import Settings
from app.db.session import create_engine, create_sessionmaker
from app.repositories.unit_of_work import unit_of_work_factory


async def _create(label: str, owner: str | None, scopes: str) -> None:
    settings = Settings()
    engine = create_engine(settings.resolved_database_url)
    try:
        factory = unit_of_work_factory(create_sessionmaker(engine))
        async with factory() as uow:
            record, token = await uow.api_keys.create(label=label, owner=owner, scopes=scopes)
            await uow.commit()
        print(f"Label:  {record.label}")
        print(f"Owner:  {record.owner or '-'}")
        print(f"Scopes: {record.scopes or '-'}")
        print(f"Prefix: {record.prefix}")
        print()
        print("Token (store now, it is not shown again):")
        print(f"  {token}")
    finally:
        await engine.dispose()


async def _list() -> None:
    settings = Settings()
    engine = create_engine(settings.resolved_database_url)
    try:
        factory = unit_of_work_factory(create_sessionmaker(engine))
        async with factory() as uow:
            keys = await uow.api_keys.list_all()
        if not keys:
            print("No API keys.")
            return
        print(f"{'prefix':<14}{'active':<7}{'owner':<20}{'label':<24}{'last_used':<20}")
        for k in keys:
            print(
                f"{k.prefix:<14}{'yes' if k.active else 'no':<7}{(k.owner or '-'):<20}"
                f"{k.label:<24}{(k.last_used_at.isoformat() if k.last_used_at else '-'):<20}"
            )
    finally:
        await engine.dispose()


async def _revoke(prefix: str) -> None:
    settings = Settings()
    engine = create_engine(settings.resolved_database_url)
    try:
        factory = unit_of_work_factory(create_sessionmaker(engine))
        async with factory() as uow:
            ok = await uow.api_keys.deactivate(prefix)
            await uow.commit()
        print("revoked" if ok else "no such key")
    finally:
        await engine.dispose()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list", action="store_true", help="list keys and exit")
    group.add_argument("--revoke", metavar="PREFIX", help="deactivate a key by its prefix")
    parser.add_argument("--label", default="unlabelled")
    parser.add_argument("--owner")
    parser.add_argument("--scopes", default="")
    args = parser.parse_args(argv)
    if args.list:
        asyncio.run(_list())
    elif args.revoke:
        asyncio.run(_revoke(args.revoke))
    else:
        asyncio.run(_create(args.label, args.owner, args.scopes))


if __name__ == "__main__":
    main(sys.argv[1:])
