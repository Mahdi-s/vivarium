from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from aam.persistence import TraceDb


class DomainStateHandler(Protocol):
    """
    Protocol for domain-specific state handlers.

    Each domain (e.g., social media, trading) can implement this to manage
    its own state tables and validation logic.
    """

    def init_schema(self, conn: sqlite3.Connection) -> None:
        """Initialize domain-specific tables."""
        ...

    def handle_action(
        self, *, action_name: str, arguments: Dict[str, Any], run_id: str, time_step: int, agent_id: str, conn: sqlite3.Connection
    ) -> Dict[str, Any]:
        """
        Handle a domain-specific action and update state.

        Returns a dictionary with the action outcome.
        """
        ...

    def get_state_snapshot(self, *, run_id: str, time_step: int, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Get a snapshot of domain state at a specific time_step."""
        ...


@dataclass
class GenericDomainState:
    """
    Generic domain state manager that allows registering custom handlers.

    This provides a pluggable system for domain-specific state management
    (e.g., posts, users, trades, etc.).
    """

    def __init__(self, trace_db: TraceDb):
        self._trace_db = trace_db
        self._handlers: Dict[str, DomainStateHandler] = {}

    def register_handler(self, domain: str, handler: DomainStateHandler) -> None:
        """Register a domain state handler."""
        self._handlers[domain] = handler
        # Initialize schema for this handler
        handler.init_schema(self._trace_db.conn)

    def handle_action(
        self, *, domain: str, action_name: str, arguments: Dict[str, Any], run_id: str, time_step: int, agent_id: str
    ) -> Dict[str, Any]:
        """Route an action to the appropriate domain handler."""
        if domain not in self._handlers:
            return {"success": False, "error": f"Unknown domain: {domain}"}

        handler = self._handlers[domain]
        return handler.handle_action(
            action_name=action_name,
            arguments=arguments,
            run_id=run_id,
            time_step=time_step,
            agent_id=agent_id,
            conn=self._trace_db.conn,
        )

    def get_state_snapshot(self, *, domain: str, run_id: str, time_step: int) -> Dict[str, Any]:
        """Get state snapshot for a domain."""
        if domain not in self._handlers:
            return {}

        handler = self._handlers[domain]
        return handler.get_state_snapshot(run_id=run_id, time_step=time_step, conn=self._trace_db.conn)


@dataclass
class SocialMediaDomainHandler:
    """
    Example domain handler for social media (posts, likes, etc.).

    This demonstrates how to implement domain-specific state management.
    """

    def init_schema(self, conn: sqlite3.Connection) -> None:
        """Initialize social media tables."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS posts (
              post_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              author_id TEXT NOT NULL,
              content TEXT NOT NULL,
              likes INTEGER DEFAULT 0,
              time_step INTEGER NOT NULL,
              created_at REAL NOT NULL,
              FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_posts_run_step ON posts(run_id, time_step);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_posts_author ON posts(run_id, author_id);")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS likes (
              like_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              post_id TEXT NOT NULL,
              user_id TEXT NOT NULL,
              time_step INTEGER NOT NULL,
              created_at REAL NOT NULL,
              FOREIGN KEY(run_id) REFERENCES runs(run_id),
              FOREIGN KEY(post_id) REFERENCES posts(post_id)
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_likes_post ON likes(post_id);")

    def handle_action(
        self, *, action_name: str, arguments: Dict[str, Any], run_id: str, time_step: int, agent_id: str, conn: sqlite3.Connection
    ) -> Dict[str, Any]:
        """Handle social media actions."""
        import time
        import uuid

        if action_name == "create_post":
            post_id = str(uuid.uuid4())
            content = str(arguments.get("content", ""))
            if not content:
                return {"success": False, "error": "content is required"}

            conn.execute(
                """
                INSERT INTO posts(post_id, run_id, author_id, content, likes, time_step, created_at)
                VALUES (?, ?, ?, ?, 0, ?, ?);
                """,
                (post_id, run_id, agent_id, content, time_step, time.time()),
            )
            return {"success": True, "data": {"post_id": post_id}}

        elif action_name == "like_post":
            post_id = str(arguments.get("post_id", ""))
            if not post_id:
                return {"success": False, "error": "post_id is required"}

            # Check if post exists
            post = conn.execute("SELECT post_id FROM posts WHERE post_id = ?;", (post_id,)).fetchone()
            if not post:
                return {"success": False, "error": "post not found"}

            # Check if already liked
            existing = conn.execute(
                "SELECT 1 FROM likes WHERE post_id = ? AND user_id = ?;", (post_id, agent_id)
            ).fetchone()
            if existing:
                return {"success": False, "error": "already liked"}

            like_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO likes(like_id, run_id, post_id, user_id, time_step, created_at)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (like_id, run_id, post_id, agent_id, time_step, time.time()),
            )

            # Update post like count
            conn.execute("UPDATE posts SET likes = likes + 1 WHERE post_id = ?;", (post_id,))

            return {"success": True, "data": {"like_id": like_id}}

        return {"success": False, "error": f"Unknown action: {action_name}"}

    def get_state_snapshot(self, *, run_id: str, time_step: int, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Get social media state snapshot."""
        posts = conn.execute(
            """
            SELECT post_id, author_id, content, likes, time_step
            FROM posts
            WHERE run_id = ? AND time_step <= ?
            ORDER BY time_step DESC;
            """,
            (run_id, time_step),
        ).fetchall()

        return {
            "posts": [dict(p) for p in posts],
            "post_count": len(posts),
        }

