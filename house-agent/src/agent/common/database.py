"""Safe, lazy database access shared by graph nodes.

The application must still boot when MySQL is temporarily unavailable.  Connections
are therefore opened only inside a node, use short timeouts, and are always closed.
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from typing import Any, Iterator

import pymysql
from pymysql.connections import Connection


class DatabaseUnavailable(RuntimeError):
    """Raised when the configured housing database cannot be reached."""


CONNECT_TIMEOUT_SECONDS = 5
IO_TIMEOUT_SECONDS = 10
MAX_QUERY_ROWS = 100


def database_configured() -> bool:
    """Return whether all required MySQL environment variables are present."""

    return all(os.getenv(name) for name in ("DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME"))


@contextmanager
def mysql_connection() -> Iterator[Connection]:
    """Yield a short-lived MySQL connection with bounded network waits."""

    if not database_configured():
        raise DatabaseUnavailable("房源数据库尚未配置")

    try:
        connection = pymysql.connect(
            host=os.environ["DB_HOST"],
            port=int(os.getenv("DB_PORT", "3306")),
            user=os.environ["DB_USER"],
            password=os.environ["DB_PASSWORD"],
            database=os.environ["DB_NAME"],
            charset="utf8mb4",
            connect_timeout=CONNECT_TIMEOUT_SECONDS,
            read_timeout=IO_TIMEOUT_SECONDS,
            write_timeout=IO_TIMEOUT_SECONDS,
            autocommit=True,
            cursorclass=pymysql.cursors.DictCursor,
        )
    except (OSError, ValueError, pymysql.MySQLError) as exc:
        raise DatabaseUnavailable("暂时无法连接房源数据库") from exc

    try:
        yield connection
    finally:
        connection.close()


_READ_ONLY_SQL = re.compile(r"^\s*(SELECT|SHOW|DESCRIBE|DESC|EXPLAIN)\b", re.IGNORECASE)
_DANGEROUS_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|REPLACE|GRANT|REVOKE|CALL|LOAD|OUTFILE)\b",
    re.IGNORECASE,
)


def execute_readonly(query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    """Execute one read-only statement and return dictionaries.

    LLM-produced SQL is rejected unless it is an unambiguously read-only, single
    statement.  A row cap prevents accidental large responses.
    """

    normalized = query.strip().rstrip(";").strip()
    if ";" in normalized or not _READ_ONLY_SQL.match(normalized) or _DANGEROUS_SQL.search(normalized):
        raise ValueError("仅允许执行单条只读 SQL 查询")

    with mysql_connection() as connection, connection.cursor() as cursor:
        cursor.execute(normalized, params)
        rows = cursor.fetchmany(MAX_QUERY_ROWS)
        return list(rows)


def quote_identifier(identifier: str) -> str:
    """Quote a database-discovered identifier."""

    return f"`{identifier.replace('`', '``')}`"
