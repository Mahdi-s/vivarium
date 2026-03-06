from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Tuple


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_step_leaf_hash(*, step_id: str, agent_id: str, prompt_hash: str, activation_hash: str) -> str:
    """
    Compute a deterministic leaf hash for one (step_id, agent_id) record.

    Leaf = SHA256(step_id | agent_id | prompt_hash | activation_hash)
    """
    payload = f"{step_id}|{agent_id}|{prompt_hash}|{activation_hash}".encode("utf-8")
    return _sha256_hex(payload)


def compute_merkle_root(leaves: List[str]) -> str:
    """
    Compute a Merkle root (SHA256 pairwise) from leaf hashes.

    - If leaves is empty, returns SHA256(b"") (stable sentinel).
    - If odd number of nodes at a level, duplicates the last node.
    """
    if not leaves:
        return _sha256_hex(b"")

    level = [str(x) for x in leaves]
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        nxt: List[str] = []
        for i in range(0, len(level), 2):
            nxt.append(_sha256_hex((level[i] + level[i + 1]).encode("utf-8")))
        level = nxt
    return level[0]


@dataclass
class MerkleLogger:
    """
    Incremental Merkle log builder. Stores leaf hashes in insertion order.
    """

    leaves: List[str] = field(default_factory=list)

    def add_step(self, *, step_id: str, agent_id: str, prompt_hash: str, activation_hash: str) -> Tuple[str, str]:
        leaf = compute_step_leaf_hash(
            step_id=str(step_id),
            agent_id=str(agent_id),
            prompt_hash=str(prompt_hash),
            activation_hash=str(activation_hash),
        )
        self.leaves.append(leaf)
        root = compute_merkle_root(self.leaves)
        return leaf, root

