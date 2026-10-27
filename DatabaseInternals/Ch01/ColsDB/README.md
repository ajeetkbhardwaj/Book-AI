# ColsDB — Column-Oriented Storage Engine Practicum

> *"Column stores partition data vertically, enabling selective I/O, vectorized execution, and high compression ratios. They are the backbone of modern OLAP systems."*

> — Inspired by *Database Internals*, Chapter 1

## 📖 Overview

`ColsDB` is a minimal, educational storage engine that implements a **column-oriented (Struct-of-Arrays)** layout in C++17. Instead of storing complete records together, it partitions data vertically into separate binary files per column. This project demonstrates why analytical queries execute faster, use less I/O, and enable advanced optimizations like late materialization and SIMD processing.

## 🧠 Core Concepts (Theory)

### 1. Struct-of-Arrays (SoA) Layout

Data is split into parallel arrays, one per column:

```

ids.bin:      [ID0 | ID1 | ID2 | ... | IDN]

names.bin:    [Name0 | Name1 | ... | NameN]

depts.bin:    [Dept0 | Dept1 | ... | DeptN]

salary.bin:   [Sal0 | Sal1 | ... | SalN]

```

Each file contains homogeneous data types, stored contiguously.

### 2. Selective I/O & Late Materialization

To compute `AVG(Salary)`, `ColsDB` reads **only** `salary.bin`. Other columns remain untouched on disk. Row reconstruction (if needed) happens *after* filtering/aggregation, minimizing wasted bandwidth.

### 3. OLAP Optimization

Column stores excel at:

- Aggregations (`SUM`, `AVG`, `COUNT`, `GROUP BY`)
- Wide-table analytics (100+ columns, few accessed per query)
- Compression (homogeneous data + run-length/delta encoding)
- Vectorized execution (SIMD instructions on contiguous arrays)

### 4. The Throughput Paradox

You may observe **lower raw MB/s** compared to row storage. This is not a regression—it reflects *less data moved*. The real metric is **useful work per byte**, where column stores dominate.

## 🏗️ Architecture & Design Decisions

| Component | Implementation | Rationale |
|-----------|----------------|-----------|
| File-per-column | `col_store_salary.bin`, `col_store_id.bin`, etc. | Independent I/O streams, enables selective reads |
| Chunked scanning | `4KB` buffers, `memcpy` to `double` | Matches disk block size, avoids memory mapping overhead |
| Row reconstruction | Offset-based indexing across files | Deferred until query requires full record (late materialization) |
| Homogeneous layout | `std::vector<double>` dumped raw | Enables SIMD, dictionary compression, and bit-packing |
| Deterministic generation | Same RNG seed as `RowsDB` | Guarantees apples-to-apples benchmark comparison |

### 💡 Systems Perspective

The query loop processes only relevant data:

```cpp

for (std::streamsize off = 0; off + sizeof(double) <= bytes; off += sizeof(double)) {

double sal;

  std::memcpy(&sal,buffer.data() + off, sizeof(double));

  sum += sal;

}

```

This is **O(N)** but with `1/8` the I/O volume. Modern column stores extend this with:

- Positional indexes (implicit row IDs via array offset)
- Zone maps / min-max metadata per chunk
- Predicate pushdown during scan

## 🚀 Quick Start & Usage

### Prerequisites

- C++17 compiler
- CMake 3.16+
- macOS / Linux / Windows

### Build

```bash

cdColsDB

mkdir-pbuild && cdbuild


# macOS

cmake-DCMAKE_BUILD_TYPE=Release..

make-j$(sysctl -n hw.ncpu)


# Linux

cmake-DCMAKE_BUILD_TYPE=Release..

make-j$(nproc)

```

### Run Benchmark

```bash

./colsdb

```

### Expected Output

```

=== Column-Oriented Storage Benchmark ===

Records: 1000000

Query:   SELECT AVG(salary)


[1/4] Generating 1000000 employees...

[2/4] Writing column files...

  Written: ~8000000 bytes (salary column only)

  Time: ~56 ms


[3/4] Querying AVG(salary)...

[Column-Oriented (SoA)]

  Avg Salary:   $119948.60

  Bytes Read:   8000000 (7.63 MB)

  I/O Time:     ~4.6 ms

  Throughput:   ~1635 MB/s

  Bytes/Record: 8

```

## 📊 Benchmark Interpretation

| Metric | What It Means | Why It Matters |
|--------|---------------|----------------|
| `Bytes/Record: 8` | Only `double salary` read | 87.5% I/O reduction vs row storage |
| `Throughput: ~1635 MB/s` | Lower bandwidth, but higher efficiency | Less useless data movement = faster wall-clock time |
| `I/O Time: ~4.6 ms` | ~3.5x faster than row scan | Directly proportional to bytes read |
| `Avg Salary matches RowsDB` | Deterministic, correct aggregation | Validates storage layout correctness |

## 🔧 Practice & Advancement Paths

### 🛠️ Beginner Extensions

1.**Add run-length encoding (RLE)**: Compress repeated department names. Benchmark read/write trade-offs.

2.**Implement min/max zone maps**: Store chunk-level statistics. Skip reading chunks that don't match `WHERE salary > 150000`.

3.**Add a position list index**: Store row IDs matching a predicate, then fetch only those salaries.

### 🚀 Intermediate/Advanced

4.**Vectorized aggregation**: Use compiler intrinsics (`_mm256_add_pd`) or `std::simd` (C++20) to sum 4 doubles per cycle.

5.**Dictionary encoding**: Map `department` strings to integer IDs. Benchmark scan + decode pipeline.

6.**Delta encoding for IDs**: Store `ID[i] - ID[i-1]` instead of raw values. Observe compression on sequential data.

### 🔬 Research-Grade

7.**Implement late materialization**: Filter on `dept` column, collect matching offsets, then fetch `salary` via scatter-gather I/O.

8.**SIMD-aware chunking**: Align files to 32/64-byte boundaries. Use non-temporal stores for write-heavy paths.

9.**Parquet-like column chunks**: Add compression codecs (ZSTD, LZ4) and column metadata footers. Practice schema evolution.

## ⚠️ Limitations & Bottlenecks

| Limitation | Root Cause | Real-World Impact |
|------------|------------|-------------------|
| **Row Reconstruction Cost** | Must gather from N files to rebuild record | High latency for `SELECT *` or point lookups |
| **Write Amplification** | Insert/update scatters data across files | Slower OLTP, requires batched/micro-batch writes |
| **Complex Join Overhead** | Must materialize positional indexes before joining | High memory pressure for multi-table analytics |
| **No Natural Ordering** | Columns stored independently | Range scans on non-primary columns require full scan or secondary index |
| **Metadata Bloat** | Zone maps, dictionaries, chunk headers add overhead | Small tables suffer from storage fragmentation |

### 📉 Mathematical Perspective

For a query selecting `k` columns out of `C`:

-**Row I/O**: `O(N × C)`

-**Column I/O**: `O(N × k) + O(materialization)`

-**Break-even point**: When `k ≈ C` or when reconstruction overhead dominates

Column stores optimize for `k << C`. When `k → C`, row stores regain parity. This is why modern systems use **hybrid layouts** or **adaptive storage**.

## 📚 Further Reading

-*Database Internals*, Chapter 1: Column- vs Row-Oriented DBMS

- Abadi et al., *The Design and Implementation of Modern Column-Oriented Database Systems* (2013)
- Apache Parquet & ORC file format specifications
- ClickHouse & DuckDB architecture whitepapers

-*Vectorized Query Execution* (MonetDB/Intel research)

## 🤝 Notes & Attribution

- Built as a companion to Chapter 1 of *Database Internals* by Alex Petrov
- Uses only C++ standard library; no external compression libs (for clarity)
- Deterministic seeding ensures reproducible cross-project comparisons
- Designed for educational benchmarking, not production workloads

> 💡 **Next Step**: Run `../compare_full.sh` to automatically compute I/O reduction, latency speedup, and efficiency ratios across both projects.
>
