"""Static guard for model-written SQL.

The model's SQL is untrusted input. Before it reaches the database it is parsed
to an AST and must satisfy every rule below; anything the guard does not
explicitly recognise is rejected (allow-list, not block-list).

This is one of four independent layers. The others live in the database:
a SELECT-only role, a READ ONLY transaction with a statement timeout, and
row-level security for tenant isolation.
"""
from __future__ import annotations

import logging

import sqlglot
from sqlglot import exp

# sqlglot warns when it falls back to a raw Command node; those are rejected below anyway.
logging.getLogger("sqlglot").setLevel(logging.ERROR)


class SqlRejected(ValueError):
    """Raised with a short, model-readable reason."""


# Function *node types* the model may use. Unknown functions parse as
# exp.Anonymous and are rejected by name, which is what blocks set_config(),
# pg_sleep(), pg_read_file(), dblink() and friends.
_ALLOWED_FUNCS: tuple[type, ...] = (
    exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max,
    exp.Coalesce, exp.Nullif, exp.Case, exp.If,
    exp.Cast, exp.TryCast, exp.Round, exp.Abs, exp.Ceil, exp.Floor,
    exp.Lower, exp.Upper, exp.Trim, exp.Length, exp.Substring, exp.Concat,
    exp.Extract, exp.DateTrunc, exp.TimestampTrunc, exp.CurrentDate, exp.CurrentTimestamp,
    exp.ToChar, exp.DateDiff, exp.TsOrDsDiff,
    exp.And, exp.Or, exp.Not,             # some sqlglot versions model these as Func
)
_ALLOWED_ANON = frozenset({"age", "date_part"})

_FORBIDDEN_NODES: tuple[type, ...] = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop, exp.Alter,
    exp.TruncateTable, exp.Command, exp.Set, exp.Transaction, exp.Commit, exp.Rollback,
    exp.Grant, exp.Copy, exp.Into, exp.Lock, exp.Pragma, exp.Use,
)


def guard(sql: str, allowed_tables: set[str], max_rows: int = 200) -> str:
    """Validate ``sql`` and return a normalised, row-capped version of it."""
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        raise SqlRejected("empty query")
    try:
        statements = sqlglot.parse(sql, read="postgres")
    except sqlglot.errors.SqlglotError as e:
        raise SqlRejected(f"could not parse SQL: {str(e).splitlines()[0][:160]}") from None

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise SqlRejected("exactly one statement is allowed")
    tree = statements[0]

    if not isinstance(tree, (exp.Select, exp.SetOperation)):
        raise SqlRejected("only SELECT queries are allowed")

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    allowed = {t.lower() for t in allowed_tables}

    for node in tree.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise SqlRejected(f"{type(node).__name__.upper()} is not allowed")

        if isinstance(node, exp.Table):
            name = node.name.lower()
            if node.args.get("db") or node.args.get("catalog"):
                raise SqlRejected("schema-qualified table names are not allowed")
            if not name:
                raise SqlRejected("table functions are not allowed")
            if name not in allowed and name not in cte_names:
                raise SqlRejected(f"table '{name}' is not available")

        elif isinstance(node, exp.Anonymous):
            if node.name.lower() not in _ALLOWED_ANON:
                raise SqlRejected(f"function '{node.name.lower()}' is not allowed")

        elif isinstance(node, exp.Func) and not isinstance(node, _ALLOWED_FUNCS):
            raise SqlRejected(f"function '{node.sql_name().lower()}' is not allowed")

    # Cap the result size regardless of what the model asked for.
    limit = tree.args.get("limit")
    current = None
    if limit is not None:
        try:
            current = int(limit.expression.name)
        except (AttributeError, ValueError):
            raise SqlRejected("LIMIT must be a plain integer") from None
    if current is None or current > max_rows:
        tree = tree.limit(max_rows)

    return tree.sql(dialect="postgres")
