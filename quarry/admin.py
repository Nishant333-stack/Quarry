"""Operational CLI.

    python -m quarry.admin init            # schema, roles, RLS (needs QUARRY_ADMIN_DATABASE_URL)
    python -m quarry.admin seed            # demo business data for two tenants
    python -m quarry.admin ingest DIR -t meridian
    python -m quarry.admin ask "question" -t meridian
    python -m quarry.admin hash-key my-secret-key
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import Settings
from .pipeline import Quarry
from .security import hash_key
from .seed import seed_demo
from .store import init_schema


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="quarry.admin")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    sub.add_parser("seed")
    p = sub.add_parser("ingest")
    p.add_argument("directory", type=Path)
    p.add_argument("-t", "--tenant", required=True)
    p = sub.add_parser("ask")
    p.add_argument("question")
    p.add_argument("-t", "--tenant", required=True)
    p = sub.add_parser("hash-key")
    p.add_argument("key")
    args = ap.parse_args(argv)

    settings = Settings.from_env()
    if args.cmd == "hash-key":
        print(hash_key(args.key))
        return
    if args.cmd == "init":
        admin = os.environ.get("QUARRY_ADMIN_DATABASE_URL")
        if not admin:
            raise SystemExit("set QUARRY_ADMIN_DATABASE_URL (an owner connection) to run init")
        init_schema(admin, settings.embed_dim)
        print(f"schema ready (embedding dim {settings.embed_dim})")
        return

    quarry = Quarry.from_settings(settings)
    try:
        if args.cmd == "seed":
            print(json.dumps(seed_demo(quarry.store), indent=2))
        elif args.cmd == "ingest":
            results = quarry.ingestor.ingest_dir(args.tenant, args.directory)
            for r in results:
                print(f"{r.status:9} {r.source_uri}  embedded={r.chunks_embedded} reused={r.chunks_reused} deleted={r.chunks_deleted}")
            print(f"{len(results)} documents, {sum(r.chunks_embedded for r in results)} chunks embedded")
        elif args.cmd == "ask":
            print(json.dumps(quarry.query(args.tenant, args.question).as_dict(), indent=2))
    finally:
        quarry.store.close()


if __name__ == "__main__":
    main()
