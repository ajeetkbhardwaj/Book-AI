# DBMS Pipeline — Modular Subsystem Architecture

> *"A database is not a monolith. It's a pipeline of cooperating subsystems, each with a single responsibility, communicating through well-defined interfaces."*

## 🧠 Core Concepts
### 1. Separation of Concerns
Real DBMSs (PostgreSQL, MySQL, CockroachDB) decompose into:
- **Transport Layer**: Network I/O, protocol parsing (TCP/HTTP)
- **Query Processor**: Lexer, parser, optimizer → `ExecutionPlan`
- **Execution Engine**: Orchestrator, pushes data through operators
- **Lock Manager**: Concurrency control, deadlock detection
- **Buffer Manager**: Page cache, LRU/MRU eviction, pin counting
- **Storage Engine**: Block device abstraction, WAL, checkpointing

This project implements a minimal, synchronous pipeline to demonstrate how a single request traverses these layers.

### 2. Request Flow (`READ 50`)
1. `Transport` receives raw string
2. `QueryProcessor` parses → `ExecutionPlan{op: READ, target: 50}`
3. `ExecutionEngine` requests `LockManager.acquire(50, SHARED)`
4. `ExecutionEngine` calls `BufferManager.pin_page(50)`
5. If cache miss → `BufferManager` delegates to `StorageEngine.read_page(50)`
6. Page loaded into RAM, pinned, returned
7. Data extracted, `BufferManager.unpin_page(50, dirty=false)`
8. `LockManager.release(50)`
9. Result returned to client

### 3. Why Interfaces?
Using abstract base classes (`IStorageEngine`, `IBufferManager`, etc.) enables:
- **Pluggable components**: Swap `DiskStorageEngine` for `MemoryStorageEngine` without touching execution logic
- **Testability**: Mock `ILockManager` to test concurrency paths deterministically
- **Clear contracts**: Each subsystem only knows its immediate neighbors

## 🏗️ Architecture Map
```
[Client/Transport]
        │
        ▼
[Query Processor] ── parses "READ 50" ──▶ [ExecutionPlan]
                                              │
                                              ▼
                                      [Execution Engine]
                                    ┌───────┴────────┐
                                    ▼                ▼
                             [Lock Manager]    [Buffer Manager]
                                                    │
                                              (Cache Miss?)
                                                    ▼
                                           [Storage Engine] ── Disk I/O
                                                    │
                                              (Return Page*)
                                                    ▼
                                     [Execution Engine] ── Extract Data
                                            │
                                            ▼
                                    [Lock Manager] ── Release
                                            │
                                            ▼
                                    [Transport] ── Send Response
```

## 🚀 Quick Start
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(sysctl -n hw.ncpu)
./dbms_pipeline
```

## ⚠️ Limitations & Real-World Gaps
| Limitation | Why It's Simplified | Production Equivalent |
|------------|---------------------|------------------------|
| **Sync Locking** | Blocks thread; no deadlock detection | Wait-for graphs, lock escalation, timeout policies |
| **Naive Buffer Pool** | No eviction, no replacement policy | Clock-sweep LRU, MRU, double buffering |
| **No WAL/Recovery** | Pages written synchronously | Write-Ahead Log, checkpoint threads, redo/undo |
| **Single-Threaded** | No concurrent query execution | Thread pools, coroutine-based schedulers |
| **No Optimizer** | Parser directly creates plan | Cost-based optimizer, cardinality estimation, join reordering |

## 🔧 Practice & Advancement
1. **Add a Transaction Manager**: Wrap lock acquire/release in `begin_txn()` / `commit()` / `rollback()`. Implement 2PL (Two-Phase Locking).
2. **Implement LRU Eviction**: Track page access order, evict unpinned pages when buffer pool is full.
3. **Add Write-Ahead Log**: Force log to disk before buffer dirty page flush. Simulate crash recovery.
4. **Make it Async**: Use `std::async` or coroutines (`co_await`) for non-blocking disk I/O.
5. **Add a Cost-Based Optimizer**: Parse `SELECT * FROM t WHERE id=50`, choose index scan vs. full table scan based on mock statistics.

## 📚 Theory Connections
- **Modular Architecture**: *Database Internals* Ch.1: "Database management systems are modular systems consisting of a transport layer, query processor, execution engine, and storage engine."
- **Buffer Pool**: The "pin count" pattern prevents eviction during active use (PostgreSQL buffer manager, InnoDB buffer pool).
- **Lock Granularity**: Row-level vs page-level vs table-level. SHARED vs EXCLUSIVE demonstrates basic concurrency control.
- **Pluggable Storage**: MySQL's InnoDB/MyRocks, MongoDB's WiredTiger, and PostgreSQL's table access methods all follow this interface pattern.

> 💡 **Next Step**: Replace the simple `ExecutionPlan` with a real operator tree (Scan → Filter → Project) and implement a pull-based execution model.
