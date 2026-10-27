# AppendOnlyHeapStore — Heap File + In-Memory Hash Index

> *"Append-only writes are sequential, predictable, and fast. Combine them with an in-memory hash map, and you get O(1) point lookups with minimal write overhead."*

## 🧠 Core Concepts
### 1. Append-Only Heap File
Records are written sequentially to disk in insertion order. No in-place updates, no sorting, no page splits. This eliminates random I/O during writes and allows the OS/disk controller to optimize for sequential throughput.

### 2. In-Memory Hash Index
A `std::unordered_map<ID, file_offset>` lives entirely in RAM. It maps logical record keys to physical byte positions on disk. Enables O(1) average-case point lookups without scanning the file.

### 3. The Volatility Trade-off
The index is **not persisted**. On crash/restart, the index must be rebuilt by scanning the file. Real systems solve this with:
- Write-Ahead Logs (WAL)
- Periodic index checkpointing (serialize hash map to disk)
- Recovery at startup (scan file, rebuild index)

## 🏗️ Architecture Decisions
| Component | Choice | Rationale |
|-----------|--------|-----------|
| `fstream` mode | `ios::in | ios::out` + `seekp(end)` | Explicit append control, avoids platform-specific `app` quirks |
| Record layout | Fixed-size `char[N]` + `double` | Zero-copy `read/write()`, predictable offsets, no serialization overhead |
| Index type | `unordered_map` | O(1) average lookups, low constant factor, standard library only |
| Durability | Manual `Sync()` + `flush()` | Demonstrates explicit fsync; real engines use WAL + async checkpoint |

## 🚀 Quick Start
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(sysctl -n hw.ncpu)
./heap_store
```

## ⚠️ Limitations & Bottlenecks
| Limitation | Impact | Real-World Mitigation |
|------------|--------|------------------------|
| **Index Volatility** | Lost on crash; full scan recovery needed | WAL + periodic index checkpoints |
| **Memory Bound** | Index size = O(N); cannot scale beyond RAM | Disk-based hash indexes, LSM-Trees, or B-Trees |
| **No Range Scans** | Hash maps don't support `WHERE id BETWEEN a AND b` | Secondary B-Tree/LSM index, or sort-on-insert |
| **Update/Delete Cost** | Requires tombstones or versioning + compaction | Log-Structured Merge (LSM) or Multi-Version Concurrency Control (MVCC) |
| **File Fragmentation** | Deleted/updated records leave gaps | Periodic rewrite/compaction phases |

## 🔧 Practice & Advancement
1. **Add Persistence**: Serialize `index_` to `heap_store.idx` on `Sync()`. Rebuild on startup.
2. **Implement Tombstones**: `Delete(id)` appends a marker. `Lookup` skips deleted records. Add compaction.
3. **Handle Updates**: Append new version + update hash map. Keep old versions for MVCC.
4. **Add LRU Eviction**: Cache recently accessed records in a `std::list` + `unordered_map` to reduce disk reads.
5. **Benchmark Recovery**: Simulate crash, measure time to rebuild index by sequential scan.

## 📚 Theory Connections
- **Heap Files**: *Database Internals* Ch.1: "Heap files require additional index structures... to make them searchable."
- **Append-Only**: Foundation of LSM-Trees, Kafka logs, and write-optimized storage.
- **Hash Indexes**: Used in Dynamo, Redis, and early MySQL Memory engine. O(1) but no ordering.
- **Durability Gap**: Illustrates why modern DBMS separate compute (index) from storage (disk) but sync via WAL.

> 💡 **Next Step**: Add range query support by replacing the hash map with a B-Tree (Practical 3) or convert to an LSM-Tree (Practical 4).
