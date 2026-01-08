# Dive Session

**Intent:** plan
**Started:** 2026-01-07
**Status:** Complete

## Focus

P2P sharing - connecting Datasphere instances to create a mesh of shared context over the internet.

## Outcome

Produced `SPEC-P2P.md` (v3) - a comprehensive design for:

- Internet-native P2P sync using iroh-net (NAT traversal + relay)
- Project-level sharing (explicit opt-in, private by default)
- Isolated peer stores (read-only snapshots)
- Pull-based incremental sync with cursors
- Fan-out queries across local + peer DBs

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| iroh-net transport | NAT traversal + relay, Rust-native |
| Project-level sharing | Right granularity, explicit privacy control |
| Snapshot semantics | Simplicity - no update/delete propagation |
| Feature flag (`--features p2p`) | Keeps base binary small |
| Isolated peer stores | No conflicts, clear provenance |

## Implementation Phases

1. Foundation - identity, config files, share commands
2. Multi-DB Query - fan-out queries, sneakernet import
3. Sync Protocol - iroh-net, cursor-based sync
4. Peer Management - add/remove/list, auth
5. Polish - daemon integration, progress indicators

## Reviews

- sg review x2 - validated approach, caught gaps (deduplication, concurrency, relay dependency)

## Next Steps

Ready for implementation. Start with Phase 1 (Foundation).
