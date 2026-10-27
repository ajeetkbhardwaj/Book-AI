# RowsDB — Row-Oriented Storage Engine Practicum

> *"Row stores optimize for spatial locality and record-level access. They are the foundation of traditional OLTP systems."*  
> — Inspired by *Database Internals*, Chapter 1

## Overview
`RowsDB` is a minimal, educational storage engine that implements a **row-oriented (Array-of-Structs)** layout in C++17. It stores fixed-size `Employee` records contiguously on disk and demonstrates why this layout excels at record-level operations but struggles with analytical aggregations.

This project bridges theoretical database architecture and real systems engineering. By reading this guide, compiling the code, and running the benchmarks, you will internalize:
- How physical data layout impacts I/O patterns
- Why spatial locality matters for cache & disk performance
- The trade-offs that drive OLTP vs. OLAP storage design

##  Core Concepts (Theory)

### 1. Array-of-Structs (AoS) Layout
Data is stored as a contiguous sequence of complete records:
```
[Record0 | Record1 | Record2 | ... | RecordN]
```
Each record contains all columns (`ID`, `Name`, `Dept`, `Salary`) packed together. This mirrors how C/C++ `struct`s are laid out in memory and on disk.

### 2. Spatial Locality & Cache Behavior
When you read one field (e.g., `Salary`), the CPU and disk subsystem load an entire cache line / disk block (typically 4KB–64KB). In row storage, **adjacent fields of the same record are co-located**, making `SELECT *` or `WHERE id = ?` extremely efficient.

### 3. OLTP Optimization
Row stores shine in:
- Point lookups (`WHERE id = ?`)
- Full-record inserts/updates
- Short-lived transactions with low concurrency overhead
- Workloads where most fields of a record are accessed together

### 4. The I/O Penalty for Analytics
To compute `AVG(Salary)`, a row store must read **every byte of every record**, even though 80–90% of the data is irrelevant to the query. This is called **read amplification** and is the primary bottleneck for analytical workloads.

##  Architecture & Design Decisions

| Component | Implementation | Rationale |
|-----------|----------------|-----------|
| `Employee` struct | Fixed-size `char[N]` fields, `double`, `int32_t` | Predictable binary layout, no serialization overhead, avoids variable-length string complexity |
| File format | Raw binary dump (`std::ios::binary`) | Eliminates text parsing overhead, mirrors real DBMS data pages |
| I/O pattern | Chunked reads (`4KB` buffers) | Matches typical disk block size, avoids loading entire file into RAM |
| Aggregation loop | `reinterpret_cast` + pointer arithmetic | Zero-copy parsing, demonstrates low-level storage access patterns |
| Memory alignment | `static_assert(sizeof(Employee) <= 64)` | Catches compiler padding early; 64B aligns with CPU cache lines |

### 💡 Systems Perspective
The query loop processes data sequentially:
```cpp
for (std::streamsize off = 0; off + sizeof(Employee) <= bytes; off += sizeof(Employee)) {
  const Employee *emp = reinterpret_cast<const Employee*>(buffer.data() + off);
  sum += emp->salary;
}
```
This is **O(N)** with excellent prefetching behavior, but it wastes bandwidth on non-salary fields. Real databases solve this with indexes, compression, or by switching to columnar layouts.

##  Quick Start & Usage

### Prerequisites
- C++17 compiler (Clang 10+, GCC 9+, or MSVC 2019+)
- CMake 3.16+
- macOS / Linux / Windows

### Build
```bash
cd RowsDB
mkdir -p build && cd build

# macOS
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(sysctl -n hw.ncpu)

# Linux
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### Run Benchmark
```bash
./rowsdb
```

### Expected Output
```
=== Row-Oriented Storage Benchmark ===
Records: 1000000
Query:   SELECT AVG(salary)

[1/3] Generating 1000000 employees...
[2/3] Writing row_store.bin...
  Written: 64000000 bytes
  Time: ~40 ms

[3/3] Querying AVG(salary)...
[Row-Oriented (AoS)]
  Avg Salary:   $119948.60
  Bytes Read:   64000000 (61.04 MB)
  I/O Time:     ~15 ms
  Throughput:   ~4000 MB/s
  Bytes/Record: 64
```

##  Benchmark Interpretation

| Metric | What It Means | Why It Matters |
|--------|---------------|----------------|
| `Bytes/Record: 64` | Full struct size including padding | Disk I/O is proportional to total struct size, not query selectivity |
| `Throughput: ~4000 MB/s` | Sequential read speed from RAM/disk cache | High bandwidth, but includes "useless" data movement |
| `I/O Time: ~15 ms` | Wall-clock time to scan 64MB | Predictable, but scales linearly with table width |
| `Avg Salary matches ColsDB` | Deterministic layout & identical RNG seed | Validates correctness across storage paradigms |

##  Practice & Advancement Paths

###  Beginner Extensions
1. **Add a secondary index**: Build a hash map `id → file_offset` and benchmark point queries vs. full scans.
2. **Implement a WAL (Write-Ahead Log)**: Append changes to a log before updating `row_store.bin`. Practice recovery logic.
3. **Add variable-length strings**: Replace `char[N]` with `std::string` + length prefixes. Observe fragmentation & serialization overhead.

###  Intermediate/Advanced
4. **Simulate a buffer pool**: Cache frequently accessed pages in `std::vector<std::vector<char>>`. Implement LRU eviction.
5. **Add transactional semantics**: Implement snapshot isolation using MVCC-style versioning per record.
6. **Page-level organization**: Divide the file into 8KB pages with a page header (slot count, free space pointer). This mirrors real DBMS slotted pages.

###  Research-Grade
7. **NUMA-aware allocation**: Pin threads to cores and allocate buffers per-NUMA-node. Measure cross-socket bandwidth penalties.
8. **Custom allocator**: Replace `std::vector` with a bump allocator or memory-mapped arena. Profile allocation overhead.

##  Limitations & Bottlenecks

| Limitation | Root Cause | Real-World Impact |
|------------|------------|-------------------|
| **Read Amplification** | Must read full records for partial queries | 80–90% I/O waste for analytical aggregates |
| **Cache Pollution** | Loading unused fields evicts useful cache lines | Higher miss rates on wide tables with many columns |
| **Poor Compression** | Mixed data types per cache line | Low entropy → dictionary/delta compression ineffective |
| **Write Amplification on Updates** | In-place updates require rewriting full records | High WAL volume, slower concurrent updates |
| **No Late Materialization** | All columns materialized at read time | Cannot push down projections or filters efficiently |

###  Mathematical Perspective
For a table with `C` columns, where a query accesses `k` columns:
- **Row I/O Cost**: `O(N × sizeof(struct))`
- **Column I/O Cost**: `O(N × Σ sizeof(col_i))`
- **Efficiency Ratio**: `Σ sizeof(col_i) / sizeof(struct) ≈ k/C`

When `k << C`, row storage is fundamentally suboptimal. This is why modern analytical databases default to columnar layouts.

## Further Reading
- *Database Internals*, Chapter 1: Storage Engines & Data Layout
- Hellerstein et al., *Architecture of a Database System* (2007)
- PostgreSQL `HEAP` storage documentation
- SQLite file format & page layout specification
- *Designing Data-Intensive Applications*, Ch. 3 (Storage & Retrieval)

##  Notes & Attribution
- Built as a companion to Chapter 1 of *Database Internals* by Alex Petrov
- Uses only C++ standard library + `<chrono>` for timing
- No external dependencies; portable across POSIX & Windows
- All benchmarks use deterministic seeding (`SEED=42`) for reproducibility

> **Next Step**: Compare with `../ColsDB` to see how vertical partitioning flips the I/O equation.
