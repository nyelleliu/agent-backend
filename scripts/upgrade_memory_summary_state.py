"""Explicit, repeatable upgrade for resumable memory-summary state.

Run ``python scripts/upgrade_memory_summary_state.py`` after taking a
database backup and confirming the configured target. This command does not
load ``main`` or run automatically at application startup. SQLite and MySQL
are supported; other dialects and unexpected schemas fail closed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine, inspect
from sqlalchemy.types import DateTime, String, Text

from app.core.database import build_database_url


TABLE = "conversation_summaries"


class MigrationConfigurationError(RuntimeError):
    """A safe-to-display error in the operator-supplied migration target."""


LEGACY_COLUMNS = {
    # These definitions mirror ConversationSummary in main.py. The id and
    # user_id constraints are required by its primary-key/unique mapping.
    "id": {"kind": "integer", "nullable": False, "primary_key": True},
    "user_id": {"kind": "string", "length": 50, "nullable": True},
    "summary_text": {"kind": "text", "nullable": True},
    "covers_up_to_message_id": {
        "kind": "integer",
        "nullable": True,
    },
}

# SQL is deliberately limited to static identifiers and declarations here.
# Defaults on these added columns initialize pre-existing rows safely. The
# ORM's default=0 is client-side; it is not a server default on fresh tables.
NEW_COLUMNS = (
    ("pending_message_id", "INTEGER", {
        "kind": "integer", "nullable": True,
    }),
    ("pending_content_hash", "VARCHAR(64)", {
        "kind": "string", "length": 64, "nullable": True,
    }),
    ("pending_offset", "INTEGER NOT NULL DEFAULT 0", {
        "kind": "integer", "nullable": False, "safe_default": 0,
    }),
    ("pending_summary_text", "TEXT", {
        "kind": "text", "nullable": True,
    }),
    ("failure_count", "INTEGER NOT NULL DEFAULT 0", {
        "kind": "integer", "nullable": False, "safe_default": 0,
    }),
    ("next_retry_at", "DATETIME", {
        "kind": "datetime", "nullable": True,
    }),
    ("failure_status", "VARCHAR(20)", {
        "kind": "string", "length": 20, "nullable": True,
    }),
    ("failure_code", "VARCHAR(64)", {
        "kind": "string", "length": 64, "nullable": True,
    }),
    ("version", "INTEGER NOT NULL DEFAULT 0", {
        "kind": "integer", "nullable": False, "safe_default": 0,
    }),
)


def _type_compatible(actual, expected):
    visit_name = getattr(actual, "__visit_name__", "").lower()
    if expected["kind"] == "integer":
        # MySQL/SQLite reflection normalizes INT aliases to integer; BIGINT is
        # also safe because it widens the ORM Integer range.
        return visit_name in {"integer", "big_integer"}
    if expected["kind"] == "string":
        # Text is a String subclass in SQLAlchemy, but is not a bounded
        # VARCHAR and cannot satisfy this model contract.
        return (
            isinstance(actual, String)
            and not isinstance(actual, Text)
            and getattr(actual, "length", None) is not None
            and actual.length >= expected["length"]
        )
    if expected["kind"] == "text":
        return isinstance(actual, Text)
    if expected["kind"] == "datetime":
        return isinstance(actual, DateTime)
    return False


def _normalize_default(value):
    if value is None:
        return None
    normalized = str(value).strip()
    while True:
        previous = normalized
        if len(normalized) >= 2 and normalized[0] == "(" and normalized[-1] == ")":
            normalized = normalized[1:-1].strip()
        if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in "'\"":
            normalized = normalized[1:-1].strip()
        if normalized == previous:
            break
    try:
        return str(int(normalized, 10))
    except ValueError:
        return normalized.lower()


def _expected_description(spec):
    pieces = [spec["kind"]]
    if "length" in spec:
        pieces.append(f"length>={spec['length']}")
    pieces.append("NOT NULL" if not spec["nullable"] else "NULLABLE")
    if "safe_default" in spec:
        pieces.append(f"default={spec['safe_default']} when present")
    return ", ".join(pieces)


def _validate_column(name, column, spec):
    actual_type = column["type"]
    if not _type_compatible(actual_type, spec):
        actual_length = getattr(actual_type, "length", None)
        raise RuntimeError(
            f"incompatible column {name}: expected {_expected_description(spec)}, "
            f"actual type={getattr(actual_type, '__visit_name__', type(actual_type).__name__)}, "
            f"length={actual_length}"
        )
    # SQLite reflection reports INTEGER PRIMARY KEY as nullable=True even
    # though rowid primary-key semantics prohibit NULL. PK membership is
    # checked separately and is authoritative for this legacy id column.
    if (
        not spec.get("primary_key")
        and bool(column["nullable"]) != spec["nullable"]
    ):
        raise RuntimeError(
            f"incompatible column {name}: expected nullable={spec['nullable']}, "
            f"actual nullable={bool(column['nullable'])}"
        )

    # The ORM defaults on pending_offset/failure_count/version are Python-side
    # defaults. Existing columns need not expose a server default when ORM
    # inserts always provide those values. If a server default exists, though,
    # it must be the safe initial value. Newly-added columns always get DEFAULT 0.
    if "safe_default" in spec and column.get("default") is not None:
        actual_default = _normalize_default(column["default"])
        expected_default = str(spec["safe_default"])
        if actual_default != expected_default:
            raise RuntimeError(
                f"incompatible column {name}: expected server default "
                f"{expected_default} when present, actual default={actual_default}"
            )


def _validate_legacy_constraints(connection):
    inspector = inspect(connection)
    primary_key = set(inspector.get_pk_constraint(TABLE).get("constrained_columns") or [])
    if "id" not in primary_key:
        raise RuntimeError(
            "incompatible column id: expected primary key membership"
        )
    if connection.dialect.name == "mysql":
        id_column = next(
            column
            for column in inspector.get_columns(TABLE)
            if column["name"] == "id"
        )
        if id_column.get("autoincrement") is not True:
            raise RuntimeError(
                "incompatible column id: expected MySQL AUTO_INCREMENT"
            )

    if connection.dialect.name == "sqlite":
        # SQLAlchemy's SQLite inspector may omit the anonymous sqlite_autoindex
        # created by an inline UNIQUE declaration. Read only SQLite schema
        # metadata to detect that constraint reliably.
        unique_user_id = False
        for index in connection.exec_driver_sql(
            f"PRAGMA index_list({TABLE})"
        ).all():
            if not index[2]:
                continue
            quoted_index = connection.dialect.identifier_preparer.quote(index[1])
            indexed_columns = connection.exec_driver_sql(
                f"PRAGMA index_info({quoted_index})"
            ).all()
            if [row[2] for row in indexed_columns] == ["user_id"]:
                unique_user_id = True
                break
    else:
        unique_user_id = any(
            set(constraint.get("column_names") or []) == {"user_id"}
            for constraint in inspector.get_unique_constraints(TABLE)
        )
        unique_user_id = unique_user_id or any(
            index.get("unique")
            and set(index.get("column_names") or []) == {"user_id"}
            for index in inspector.get_indexes(TABLE)
        )
    if not unique_user_id:
        raise RuntimeError(
            "incompatible column user_id: expected a unique constraint"
        )


def upgrade(engine) -> None:
    """Add missing summary-state columns without rebuilding or deleting rows."""
    dialect = engine.dialect.name
    if dialect not in {"sqlite", "mysql"}:
        raise RuntimeError(f"unsupported database dialect: {dialect}")

    with engine.begin() as connection:
        inspector = inspect(connection)
        if not inspector.has_table(TABLE):
            raise RuntimeError(f"required table is missing: {TABLE}")

        column_info = {
            column["name"]: column
            for column in inspector.get_columns(TABLE)
        }
        missing_legacy = set(LEGACY_COLUMNS) - set(column_info)
        if missing_legacy:
            names = ", ".join(sorted(missing_legacy))
            raise RuntimeError(f"legacy table is missing required columns: {names}")
        _validate_legacy_constraints(connection)
        for name, spec in LEGACY_COLUMNS.items():
            _validate_column(name, column_info[name], spec)

        # Validate every already-present state field before changing the
        # schema. This prevents a known incompatible partial upgrade from
        # being made even more partial by adding later columns first.
        specs_by_name = {name: spec for name, _, spec in NEW_COLUMNS}
        for name, column in column_info.items():
            if name in specs_by_name:
                _validate_column(name, column, specs_by_name[name])

        for name, sql_declaration, spec in NEW_COLUMNS:
            if name not in column_info:
                connection.exec_driver_sql(
                    f"ALTER TABLE {TABLE} ADD COLUMN {name} {sql_declaration}"
                )
                column_info = {
                    column["name"]: column
                    for column in inspect(connection).get_columns(TABLE)
                }
            _validate_column(name, column_info[name], spec)


def _engine_from_environment():
    password = os.environ.get("MYSQL_PASSWORD")
    if not password:
        raise MigrationConfigurationError(
            "MYSQL_PASSWORD must be set in the process environment"
        )

    # A migration is destructive in scope even though this script only adds
    # columns. Require the operator to identify both the server and schema;
    # never infer a migration target from application defaults.
    host = os.environ.get("MYSQL_HOST")
    if host is None:
        raise MigrationConfigurationError(
            "MYSQL_HOST is missing; explicitly set the intended migration server"
        )
    host = host.strip()
    if not host:
        raise MigrationConfigurationError(
            "MYSQL_HOST is empty; explicitly set the intended migration server"
        )

    database = os.environ.get("MYSQL_DATABASE")
    if database is None:
        raise MigrationConfigurationError(
            "MYSQL_DATABASE is missing; explicitly set the intended migration database"
        )
    database = database.strip()
    if not database:
        raise MigrationConfigurationError(
            "MYSQL_DATABASE is empty; explicitly set the intended migration database"
        )

    url = build_database_url(
        os.getenv("MYSQL_USER", "root"),
        password,
        host,
        os.getenv("MYSQL_PORT", "3306"),
        database=database,
    )
    return create_engine(url, pool_pre_ping=True)


def main() -> int:
    engine = None
    try:
        engine = _engine_from_environment()
        upgrade(engine)
    except MigrationConfigurationError as exc:
        # These messages are static and contain only environment variable
        # names and operator guidance. Keep all other exception details hidden.
        print(
            "memory summary state upgrade failed "
            f"configuration_error={exc}",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(
            "memory summary state upgrade failed "
            f"error_type={type(exc).__name__}",
            file=sys.stderr,
        )
        return 1
    finally:
        if engine is not None:
            engine.dispose()
    print("memory summary state schema is up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
