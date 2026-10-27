# BTreePager — Paged Storage & Disk-Resident B-Tree

> *"B-Trees solve the RAM-bound index problem by organizing data into fixed-size disk pages. The tree grows vertically, keeping depth logarithmic and I/O predictable."*

## 🧠 Core Concepts
### 1. Fixed-Size Pages (4KB)
Modern disks/SSDs read/write in blocks. B-Trees align with this by storing multiple keys/values in 4KB pages. A single page read brings dozens/hundreds of entries into memory, amortizing I/O latency.

### 2. Disk-Resident Hierarchy
Unlike in-memory trees, B-Tree nodes live on disk. Internal nodes act as routing tables (`key → child_page_id`). Leaf nodes store actual data. The tree stays balanced by splitting full nodes during inserts.

### 3. Node Splitting & Root Growth
When a leaf exceeds capacity, it splits into two. The median key propagates upward. If the root splits, a new root is created, increasing tree depth by exactly 1. This guarantees `O(log N)` height.

## 🏗️ Architecture Map
```
[PageManager]
  ├─ allocates/free_pages() via linked free-list
  ├─ read/write_page() via seekg/seekp + memcpy
  └─ maintains metadata in Page 0 (root_id, next_free)

[BTree]
  ├─ insert_recursive() → finds leaf, inserts, handles overflow
  ├─ split_leaf() → halves data, creates new page, links siblings
  ├─ split_internal() → propagates median key upward, splits children
  └─ search() → traverses from root to leaf using key comparisons
```

## 🚀 Quick Start
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(sysctl -n hw.ncpu)
./btree_pager
```

## ⚠️ Limitations & Production Gaps
| Limitation | Educational Simplification | Production Equivalent |
|------------|----------------------------|------------------------|
| **Sync I/O** | Blocking `read/write` | Async I/O (`io_uring`), double buffering, background pre-fetch |
| **No WAL** | Crashes corrupt tree | Write-Ahead Log + checkpointing + undo/redo recovery |
| **No Concurrency** | Single-threaded only | Latches per page, B-Tree latching protocol (B-link trees) |
| **Fixed Types** | `int64_t` only | Variable-length serialization, type metadata, TOAST for large values |
| **No Range Scans** | Missing leaf traversal logic | `next_sibling_id` chain, scan continuation, key boundaries |

## 🔧 Practice & Advancement
1. **Add WAL**: Append `INSERT` records to `btree.wal` before modifying pages. Simulate crash recovery by replaying log.
2. **Implement LRU Buffer Pool**: Cache hot pages in RAM. Evict using clock-sweep or LRU-K. Count physical vs logical reads.
3. **Add Range Queries**: Use `next_sibling_id` to traverse leaves. Return `std::vector<Value>` for `key ∈ [a, b]`.
4. **Support Deletes**: Implement lazy deletion (tombstones) + periodic merge/compaction of underfull nodes.
5. **B+ Tree Conversion**: Move all data to leaves. Make internal nodes key-only routing. Benchmark insert/search throughput.

## 📚 Theory Connections
- **Paged Storage**: *Database Internals* Ch.2: "Disk-based storage structures often have a form of wide and short trees... optimized for block-wise access."
- **B-Tree vs Hash**: Hash indexes are `O(1)` but unbounded in RAM and lack range support. B-Trees trade constant factor for `O(log N)` disk access and bounded memory footprint.
- **Split Mechanics**: Root splits are rare but critical. They increase depth predictably, avoiding unbalanced growth.
- **Page Fill Factor**: ~50-75% in practice due to deletes/splits. B-link trees and optimistic concurrency improve fill rates.

> 💡 **Next Step**: Add a Write-Ahead Log (WAL) to make this crash-safe. Then integrate it into the `DBMS_Pipeline` from Practical 4 as the storage engine layer.
