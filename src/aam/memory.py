from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from aam.types import Observation


class MemorySystem(Protocol):
    """
    Protocol for pluggable memory systems (FR-05).

    Supports:
    - Short-Term Memory: Context window management
    - Long-Term Memory: Vector retrieval
    - Reflection: Summarization of past events
    """

    def store(self, *, agent_id: str, time_step: int, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store a memory entry."""
        ...

    def retrieve(
        self, *, agent_id: str, query: str, limit: int = 10, time_step: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories via vector search."""
        ...

    def summarize(self, *, agent_id: str, up_to_time_step: int) -> Optional[str]:
        """Generate a summary of past events (reflection)."""
        ...

    def get_short_term_context(
        self, *, agent_id: str, time_step: int, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get recent memories for context window (short-term memory)."""
        ...


@dataclass
class SimpleMemorySystem:
    """
    Simple in-memory implementation for MVP (no vector DB required).

    This provides the memory interface but uses simple text matching.
    For production, replace with a vector DB implementation (ChromaDB, surrealdb.py).
    """

    def __init__(self, *, llm_gateway: Optional[Any] = None, model: Optional[str] = None):
        """
        Initialize simple memory system.

        Args:
            llm_gateway: Optional LLM gateway for intelligent summarization
            model: Optional model name for LLM summarization
        """
        self._memories: Dict[str, List[Dict[str, Any]]] = {}  # agent_id -> list of memories
        self._llm_gateway = llm_gateway
        self._model = model or "gpt-3.5-turbo"

    def store(
        self, *, agent_id: str, time_step: int, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store a memory entry."""
        if agent_id not in self._memories:
            self._memories[agent_id] = []

        memory = {
            "time_step": time_step,
            "content": content,
            "metadata": metadata or {},
            "id": hashlib.sha256(f"{agent_id}:{time_step}:{content}".encode()).hexdigest()[:16],
        }
        self._memories[agent_id].append(memory)

    def retrieve(
        self, *, agent_id: str, query: str, limit: int = 10, time_step: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Simple text-based retrieval (no vector search).
        For production, implement with vector embeddings.
        """
        if agent_id not in self._memories:
            return []

        memories = self._memories[agent_id]
        if time_step is not None:
            memories = [m for m in memories if m["time_step"] <= time_step]

        # Simple keyword matching
        query_lower = query.lower()
        scored = []
        for mem in memories:
            content = str(mem.get("content", "")).lower()
            score = sum(1 for word in query_lower.split() if word in content)
            if score > 0:
                scored.append((score, mem))

        # Sort by score and time_step (most recent first)
        scored.sort(key=lambda x: (-x[0], -x[1]["time_step"]))
        return [mem for _, mem in scored[:limit]]

    def summarize(self, *, agent_id: str, up_to_time_step: int) -> Optional[str]:
        """
        Generate a summary of past events (reflection).
        
        If LLM gateway is available, uses it for intelligent summarization.
        Otherwise, provides a structured summary of key events.
        """
        if agent_id not in self._memories:
            return None

        memories = [m for m in self._memories[agent_id] if m["time_step"] <= up_to_time_step]
        if not memories:
            return None

        # Try LLM-based summarization if gateway is available
        if self._llm_gateway is not None:
            try:
                # Prepare context for LLM
                recent_memories = sorted(memories, key=lambda x: -x["time_step"])[:20]
                context = "\n".join(
                    [
                        f"Time {m['time_step']}: {m.get('content', '')}"
                        for m in recent_memories
                    ]
                )

                messages = [
                    {
                        "role": "system",
                        "content": f"Summarize the agent's behavior and key events up to time step {up_to_time_step}. Focus on patterns, decisions, and important interactions.",
                    },
                    {"role": "user", "content": f"Agent memories:\n{context}"},
                ]

                response = self._llm_gateway.chat(model=self._model, messages=messages, temperature=0.3)
                if isinstance(response, dict) and "choices" in response:
                    content = response["choices"][0].get("message", {}).get("content")
                    if content:
                        return content
            except Exception:
                # Fall back to simple summary on error
                pass

        # Simple text-based summary (fallback)
        actions = [m for m in memories if m.get("metadata", {}).get("type") == "action"]
        observations = [m for m in memories if m.get("metadata", {}).get("type") == "observation"]

        summary_parts = [
            f"Agent {agent_id} summary (up to time_step {up_to_time_step}):",
            f"- Total memories: {len(memories)}",
            f"- Actions taken: {len(actions)}",
            f"- Observations: {len(observations)}",
        ]

        # List recent actions
        if actions:
            recent_actions = sorted(actions, key=lambda x: -x["time_step"])[:5]
            action_types = {}
            for a in recent_actions:
                action_name = a.get("metadata", {}).get("action_name", "unknown")
                action_types[action_name] = action_types.get(action_name, 0) + 1
            if action_types:
                summary_parts.append(f"- Recent action types: {', '.join(f'{k}({v})' for k, v in action_types.items())}")

        return "\n".join(summary_parts)

    def get_short_term_context(
        self, *, agent_id: str, time_step: int, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get recent memories for context window."""
        if agent_id not in self._memories:
            return []

        memories = [m for m in self._memories[agent_id] if m["time_step"] <= time_step]
        # Return most recent first
        memories.sort(key=lambda x: -x["time_step"])
        return memories[:limit]


class MemoryManager:
    """
    Manager for agent memory systems.

    Integrates with WorldEngine to automatically store observations and actions.
    """

    def __init__(self, memory_system: MemorySystem):
        self._memory = memory_system

    def store_observation(self, *, agent_id: str, time_step: int, observation: Observation) -> None:
        """Store an observation in memory."""
        # Extract key information from observation
        messages = observation.get("messages", [])
        if messages:
            # Store the most recent message
            last_msg = messages[-1] if messages else None
            if last_msg:
                content = f"Message from {last_msg.get('author_id', 'unknown')}: {last_msg.get('content', '')}"
                self._memory.store(
                    agent_id=agent_id,
                    time_step=time_step,
                    content=content,
                    metadata={"type": "observation", "time_step": time_step},
                )

    def store_action(self, *, agent_id: str, time_step: int, action_name: str, arguments: Dict[str, Any]) -> None:
        """Store an action in memory."""
        content = f"Action: {action_name} with args: {arguments}"
        self._memory.store(
            agent_id=agent_id,
            time_step=time_step,
            content=content,
            metadata={"type": "action", "action_name": action_name, "time_step": time_step},
        )

    def enrich_observation(
        self, *, agent_id: str, time_step: int, observation: Observation, query: Optional[str] = None
    ) -> Observation:
        """
        Enrich an observation with long-term memory context.

        If query is provided, performs vector search. Otherwise, uses short-term context.
        """
        if query:
            # Long-term memory retrieval
            memories = self._memory.retrieve(agent_id=agent_id, query=query, limit=5, time_step=time_step)
        else:
            # Short-term memory
            memories = self._memory.get_short_term_context(agent_id=agent_id, time_step=time_step, limit=10)

        # Add memory context to observation
        enriched = dict(observation)
        enriched["memory_context"] = [
            {
                "time_step": m["time_step"],
                "content": m["content"],
                "metadata": m.get("metadata", {}),
            }
            for m in memories
        ]

        # Add reflection summary if available
        summary = self._memory.summarize(agent_id=agent_id, up_to_time_step=time_step)
        if summary:
            enriched["memory_summary"] = summary

        return enriched


@dataclass
class ChromaDBMemorySystem:
    """
    ChromaDB-based memory system with vector embeddings for long-term memory (FR-05).

    Uses ChromaDB for persistent vector storage and similarity search.
    Requires: pip install chromadb sentence-transformers
    """

    def __init__(
        self,
        *,
        persist_directory: Optional[str] = None,
        collection_name: str = "agent_memories",
        llm_gateway: Optional[Any] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize ChromaDB memory system.

        Args:
            persist_directory: Directory to persist ChromaDB data (None = in-memory)
            collection_name: Name of the ChromaDB collection
            llm_gateway: Optional LLM gateway for intelligent summarization
            model: Optional model name for LLM summarization
        """
        try:
            import chromadb  # type: ignore
            from chromadb.config import Settings  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "ChromaDB is not installed. Install extras: `pip install -e .[memory]` or `pip install chromadb sentence-transformers`"
            ) from e

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers is not installed. Install extras: `pip install -e .[memory]` or `pip install sentence-transformers`"
            ) from e

        # Initialize embedding model (use a lightweight model for efficiency)
        self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Initialize ChromaDB client
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=persist_directory, settings=Settings(anonymized_telemetry=False)
            )
        else:
            self._client = chromadb.Client(settings=Settings(anonymized_telemetry=False))

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        # LLM summarization support
        self._llm_gateway = llm_gateway
        self._model = model or "gpt-3.5-turbo"

    def store(
        self, *, agent_id: str, time_step: int, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store a memory entry with vector embedding."""
        memory_id = hashlib.sha256(f"{agent_id}:{time_step}:{content}".encode()).hexdigest()[:16]

        # Generate embedding
        embedding = self._embedding_model.encode(content, convert_to_numpy=True).tolist()

        # Prepare metadata
        doc_metadata = {
            "agent_id": agent_id,
            "time_step": time_step,
            "content": content,
            **(metadata or {}),
        }

        # Store in ChromaDB
        self._collection.add(
            ids=[f"{agent_id}_{memory_id}"],
            embeddings=[embedding],
            documents=[content],
            metadatas=[doc_metadata],
        )

    def retrieve(
        self, *, agent_id: str, query: str, limit: int = 10, time_step: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories via vector similarity search.
        """
        # Generate query embedding
        query_embedding = self._embedding_model.encode(query, convert_to_numpy=True).tolist()

        # Build where clause for filtering
        where_clause: Dict[str, Any] = {"agent_id": agent_id}
        if time_step is not None:
            where_clause["time_step"] = {"$lte": time_step}

        # Query ChromaDB
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where_clause,
        )

        # Convert to expected format
        memories = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                memory = {
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "time_step": results["metadatas"][0][i].get("time_step", 0) if results["metadatas"] else 0,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                }
                memories.append(memory)

        return memories

    def summarize(self, *, agent_id: str, up_to_time_step: int) -> Optional[str]:
        """
        Generate a summary of past events (reflection).

        If LLM gateway is available, uses it for intelligent summarization.
        Otherwise, provides a structured summary of key events.
        """
        # Retrieve recent memories
        where_clause = {"agent_id": agent_id, "time_step": {"$lte": up_to_time_step}}
        results = self._collection.get(where=where_clause, limit=100)

        if not results["ids"]:
            return None

        memories = []
        for i in range(len(results["ids"])):
            memories.append(
                {
                    "time_step": results["metadatas"][i].get("time_step", 0) if results["metadatas"] else 0,
                    "content": results["documents"][i] if results["documents"] else "",
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                }
            )

        # Try LLM-based summarization if gateway is available
        if self._llm_gateway is not None:
            try:
                # Prepare context for LLM
                recent_memories = sorted(memories, key=lambda x: -x["time_step"])[:20]
                context = "\n".join([f"Time {m['time_step']}: {m.get('content', '')}" for m in recent_memories])

                messages = [
                    {
                        "role": "system",
                        "content": f"Summarize the agent's behavior and key events up to time step {up_to_time_step}. Focus on patterns, decisions, and important interactions.",
                    },
                    {"role": "user", "content": f"Agent memories:\n{context}"},
                ]

                response = self._llm_gateway.chat(model=self._model, messages=messages, temperature=0.3)
                if isinstance(response, dict) and "choices" in response:
                    content = response["choices"][0].get("message", {}).get("content")
                    if content:
                        return content
            except Exception:
                # Fall back to simple summary on error
                pass

        # Simple text-based summary (fallback)
        actions = [m for m in memories if m.get("metadata", {}).get("type") == "action"]
        observations = [m for m in memories if m.get("metadata", {}).get("type") == "observation"]

        summary_parts = [
            f"Agent {agent_id} summary (up to time_step {up_to_time_step}):",
            f"- Total memories: {len(memories)}",
            f"- Actions taken: {len(actions)}",
            f"- Observations: {len(observations)}",
        ]

        # List recent actions
        if actions:
            recent_actions = sorted(actions, key=lambda x: -x["time_step"])[:5]
            action_types = {}
            for a in recent_actions:
                action_name = a.get("metadata", {}).get("action_name", "unknown")
                action_types[action_name] = action_types.get(action_name, 0) + 1
            if action_types:
                summary_parts.append(
                    f"- Recent action types: {', '.join(f'{k}({v})' for k, v in action_types.items())}"
                )

        return "\n".join(summary_parts)

    def get_short_term_context(
        self, *, agent_id: str, time_step: int, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get recent memories for context window."""
        where_clause = {"agent_id": agent_id, "time_step": {"$lte": time_step}}
        results = self._collection.get(where=where_clause, limit=limit)

        if not results["ids"]:
            return []

        memories = []
        for i in range(len(results["ids"])):
            memory = {
                "id": results["ids"][i],
                "content": results["documents"][i] if results["documents"] else "",
                "time_step": results["metadatas"][i].get("time_step", 0) if results["metadatas"] else 0,
                "metadata": results["metadatas"][i] if results["metadatas"] else {},
            }
            memories.append(memory)

        # Sort by time_step (most recent first)
        memories.sort(key=lambda x: -x["time_step"])
        return memories[:limit]

