#include "append_only_heap_store.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <filesystem>
#include <iomanip>

namespace chrono = std::chrono;
namespace fs = std::filesystem;

int main() {
    constexpr size_t NUM_RECORDS = 1'000'000;
    constexpr uint32_t SEED = 42;
    const std::string DB_FILE = "heap_store.dat";

    std::cout << "=== Append-Only Heap File + In-Memory Hash Index ===\n";
    std::cout << "Records: " << NUM_RECORDS << "\n";
    std::cout << "File: " << DB_FILE << "\n\n";

    // Clean up previous run
    if (fs::exists(DB_FILE)) fs::remove(DB_FILE);

    // Initialize storage engine
    AppendOnlyHeapStore store(DB_FILE);

    // 1. BULK INSERT BENCHMARK
    std::cout << "[1/3] Inserting records (append-only)...\n";
    std::mt19937 rng(SEED);
    std::uniform_int_distribution<int32_t> id_dist(0, NUM_RECORDS - 1);
    std::uniform_real_distribution<double> metric_dist(0.0, 1000.0);

    auto t_start = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < NUM_RECORDS; ++i) {
        Record r;
        r.id = static_cast<int32_t>(i);
        r.metric = metric_dist(rng);
        std::snprintf(r.payload, sizeof(r.payload), "record_%zu", i);
        store.Insert(r);
    }
    auto t_end = chrono::high_resolution_clock::now();
    double insert_ms = chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "  Time: " << std::fixed << std::setprecision(2) << insert_ms << " ms\n";
    std::cout << "  Throughput: " << std::fixed << std::setprecision(0) 
              << (NUM_RECORDS / (insert_ms / 1000.0)) << " recs/sec\n";
    std::cout << "  Index Size: " << store.IndexSize() << " entries\n\n";

    // Force sync to disk
    store.Sync();

    // 2. POINT LOOKUP BENCHMARK (Hash Index vs Full Scan)
    std::cout << "[2/3] Point lookups via hash index...\n";
    std::vector<int32_t> lookup_ids;
    for (size_t i = 0; i < 10'000; ++i) {
        lookup_ids.push_back(id_dist(rng));
    }

    t_start = chrono::high_resolution_clock::now();
    size_t hits = 0;
    for (int32_t id : lookup_ids) {
        auto res = store.Lookup(id);
        if (res) ++hits;
    }
    t_end = chrono::high_resolution_clock::now();
    double lookup_ms = chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "  Queries: " << lookup_ids.size() << "\n";
    std::cout << "  Hits: " << hits << "\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(2) << lookup_ms << " ms\n";
    std::cout << "  Latency/query: " << std::fixed << std::setprecision(3) 
              << (lookup_ms / lookup_ids.size()) << " ms\n\n";

    // 3. MEMORY & I/O ANALYSIS
    std::cout << "[3/3] Storage Analysis\n";
    size_t file_size = fs::file_size(DB_FILE);
    double file_mb = file_size / (1024.0 * 1024.0);
    size_t index_mem = store.IndexSize() * (sizeof(int32_t) + sizeof(std::streampos) + 
                     (sizeof(typename std::unordered_map<int32_t, std::streampos>::value_type) * 0.5)); // rough overhead
    double index_mb = index_mem / (1024.0 * 1024.0);
    
    std::cout << "  Disk Usage: " << std::fixed << std::setprecision(2) << file_mb << " MB\n";
    std::cout << "  Index RAM:  ~" << std::fixed << std::setprecision(2) << index_mb << " MB\n";
    std::cout << "  Avg Record Size: " << (file_size / NUM_RECORDS) << " bytes\n";

    std::cout << "\n✅ Benchmark complete. File: " << DB_FILE << "\n";
    return 0;
}
