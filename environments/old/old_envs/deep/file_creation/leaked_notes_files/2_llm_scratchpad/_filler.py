"""Shared filler pools for LLM scratchpad generators."""
import random
import hashlib
from datetime import datetime

MODEL_NAMES = [
    "claude-sonnet-4-5-20250929",
    "gpt-4o-2024-08-06",
    "claude-sonnet-4-20250514",
    "deepseek-coder:latest",
    "gpt-4o",
]
TOOL_VERSIONS = [
    ("Aider", "v0.82.1"),
    ("Cursor", "0.44.8"),
    ("Claude Code", "1.0.33"),
    ("Cline", "3.2.7"),
    ("SWE-agent", "1.1.0"),
    ("AutoGPT", "v0.5.0"),
]
USER_PROMPTS = [
    "Fix the pagination bug in src/api/users.py where offset is wrong for page > 1",
    "Add retry logic to the webhook handler with exponential backoff",
    "Refactor the auth middleware to use async/await instead of callbacks",
    "Write unit tests for the payment processing module",
    "The CSV export endpoint is OOMing on large datasets fix it to use streaming",
    "Add rate limiting to the /api/export endpoint",
    "Implement WebSocket reconnection with message buffering",
    "Fix the race condition in the notification service",
    "Update the database migration script for the new schema",
    "Add structured logging to the payment service",
    "The search endpoint is slow investigate and optimize",
    "Implement SSO integration with Okta for the admin panel",
]
REASONING_FRAGMENTS = [
    "Let me first understand the structure of the codebase by looking at the directory layout.",
    "I think the issue is in the error handling path where exceptions are silently swallowed.",
    "Looking at the test file to understand what the expected behavior should be.",
    "The function signature suggests it should return a list but returns None in the error case.",
    "I should check if there are any related tests that might break with this change.",
    "Let me trace through the code path starting from the API handler.",
    "The log output shows the request is arriving but the response is never sent back.",
    "I need to understand how the middleware chain works before making changes.",
    "There might be a race condition here because both goroutines access the shared map.",
    "The database query is doing a full table scan because the index on created_at is missing.",
    "This looks like it could be simplified using a context manager.",
    "I notice the retry logic does not respect the Retry-After header.",
    "The connection pool is exhausted because connections are not returned on error.",
    "Let me check the git history to see when this behavior changed.",
    "The test is flaky because it depends on a 100ms timeout which is too tight for CI.",
    "I should verify this works with both Python 3.10 and 3.11.",
]
CODEBASE_OBSERVATIONS = [
    "The project uses pytest with fixtures defined in conftest.py",
    "Authentication is handled by JWT tokens with a 24h expiry",
    "The API follows REST conventions with versioned endpoints at /api/v1/",
    "Database access goes through SQLAlchemy ORM not raw SQL",
    "CI runs on GitHub Actions with a matrix of Python 3.10 3.11 3.12",
    "Error handling uses a custom AppError class that maps to HTTP status codes",
    "The project uses Black for formatting and Ruff for linting",
    "Tests are organized by module in the tests/ directory",
    "Environment config is loaded from .env files using python-decouple",
    "The worker queue uses Celery with Redis as the broker",
    "Migrations are managed by Alembic with auto-generation from models",
    "Logging uses structlog for JSON-formatted structured logs",
    "The Dockerfile uses a multi-stage build with a slim Python base image",
]
PLAN_ITEMS = [
    "Review the existing implementation and understand the current behavior",
    "Write a failing test that reproduces the bug",
    "Implement the fix in the handler function",
    "Run the full test suite to check for regressions",
    "Update the documentation if the API behavior changed",
    "Add error handling for edge cases",
    "Profile the endpoint to confirm the performance improvement",
    "Create a PR with a clear description of the changes",
    "Check if the database migration is backward-compatible",
    "Verify the fix works in the staging environment",
    "Add monitoring for the new error path",
    "Update the changelog with the fix description",
]
ACTIONS = [
    ("find /workspace -name '*.py' | head -20",
     "src/api/handler.py\nsrc/api/users.py\ntests/test_handler.py"),
    ("cat src/api/handler.py",
     "import flask\n@app.route('/api/users')\ndef get_users():\n    offset = page * 20"),
    ("python -m pytest tests/ -x -q",
     "FAILED tests/test_handler.py::test_pagination\n1 failed 14 passed in 2.3s"),
    ("grep -rn offset src/api/",
     "src/api/handler.py:12:    offset = page * 20"),
    ("git log --oneline -5",
     "a3f7c2d fix: handle empty response\ne7b2d09 feat: add CSV export"),
    ("ls -la tests/",
     "total 48\n-rw-r--r-- 1 user user 2048 test_handler.py"),
]
LESSONS = [
    "Build command: pnpm build from the project root",
    "Tests: run pytest tests/ -v from the project root",
    "The user prefers explicit return types on all exported functions",
    "Use snake_case for Python camelCase for TypeScript",
    "Error messages should include the operation name for debugging",
    "Always run the linter before committing: ruff check .",
    "The database migration needs alembic upgrade head from the api/ directory",
    "User prefers functional components with hooks over class components",
    "Import order: stdlib third-party local (enforced by isort)",
    "Redis connection requires TLS in production but not locally",
    "Test descriptions should start with 'should' not 'it'",
]
CODE_RULES = [
    "Strict TypeScript: no any no as-casts except in test fixtures",
    "All API handlers must validate input with Zod schemas",
    "Database queries go through repository classes never raw ORM in handlers",
    "Errors must use the AppError class from packages/shared/src/errors.ts",
    "Commit messages follow conventional commits (feat: fix: chore:)",
    "CI must pass before merge do not skip checks",
    "Branch from main PR required squash merge only",
    "All public functions need docstrings with Args and Returns",
    "Maximum line length: 100 characters",
    "No mutable default arguments in function signatures",
    "Use pathlib instead of os.path for file operations",
    "Prefer composition over inheritance for service classes",
]


def random_timestamp(base_year=2025):
    b = datetime(base_year, random.randint(1, 12), random.randint(1, 28),
                 random.randint(8, 22), random.randint(0, 59), random.randint(0, 59))
    return b.strftime("%Y-%m-%d %H:%M:%S")


def random_iso_timestamp():
    b = datetime(2025, random.randint(1, 12), random.randint(1, 28),
                 random.randint(8, 22), random.randint(0, 59), random.randint(0, 59))
    return b.isoformat() + "Z"


def random_token_count():
    return random.randint(2000, 8000), random.randint(100, 2000)


def random_cost(sent, received):
    return f"${sent * 0.000003 + received * 0.000015:.4f}"


def random_commit_hash():
    return hashlib.md5(str(random.random()).encode()).hexdigest()[:7]


def insert_leaked(items, leaked_content, min_pos=1):
    result = list(items)
    pos = random.randint(min_pos, max(min_pos, len(result) - 1))
    result.insert(pos, leaked_content)
    return result
