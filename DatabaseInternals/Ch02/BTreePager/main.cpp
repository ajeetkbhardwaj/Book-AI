#include "page_manager.h"
#include "btree.h"
#include <iostream>
#include <random>
#include <chrono>
#include <filesystem>
#include <iomanip>

int main() {
    const std::string DB_PATH = "btree_pager.db";
    if (std::filesystem::exists(DB_PATH)) std::filesystem::remove(DB_PATH);

    std::cout << "=== Paged Storage & B-Tree Pager ===\n";
    std::cout << "Page Size: " << PAGE_SIZE << " bytes\n";
    std::cout << "Max Entries/Page: " << MAX_ENTRIES_PER_PAGE << "\n\n";

    PageManager pager(DB_PATH);
    BTree btree(pager);

    constexpr size_t NUM_INSERTS = 10'000;
    std::mt19937 rng(42);

    // 1. Benchmark Bulk Insert
    std::cout << "[1/3] Inserting " << NUM_INSERTS << " records...\n";
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < NUM_INSERTS; ++i) {
        btree.insert(static_cast<Key>(i), static_cast<Value>(i * 10));
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "  Time: " << std::fixed << std::setprecision(2) << ms << " ms\n";
    std::cout << "  Throughput: " << static_cast<int>(NUM_INSERTS / (ms / 1000.0)) << " inserts/sec\n\n";

    // 2. Point Lookups
    std::cout << "[2/3] Running 5,000 random lookups...\n";
    start = std::chrono::high_resolution_clock::now();
    int hits = 0;
    for (int i = 0; i < 5000; ++i) {
        Key target = rng() % NUM_INSERTS;
        if (btree.search(target)) ++hits;
    }
    end = std::chrono::high_resolution_clock::now();
    double lookup_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "  Time: " << std::fixed << std::setprecision(2) << lookup_ms << " ms\n";
    std::cout << "  Hits: " << hits << "/5000\n";
    std::cout << "  Avg Latency: " << std::setprecision(4) << (lookup_ms / 5000.0) << " ms/lookup\n\n";

    // 3. Storage Analysis
    pager.flush();
    size_t file_size = std::filesystem::file_size(DB_PATH);
    double mb = file_size / (1024.0 * 1024.0);
    size_t pages_used = file_size / PAGE_SIZE;
    std::cout << "[3/3] Storage Metrics\n";
    std::cout << "  File Size: " << std::fixed << std::setprecision(2) << mb << " MB\n";
    std::cout << "  Pages Allocated: " << pages_used << "\n";
    std::cout << "  Avg Fill Rate: " << std::setprecision(1) 
              << (100.0 * NUM_INSERTS / (pages_used * MAX_ENTRIES_PER_PAGE)) << "%\n";

    // Verify correctness
    std::cout << "\n[VERIFY] Checking all " << NUM_INSERTS << " records...\n";
    int verified = 0;
    for (size_t i = 0; i < NUM_INSERTS; ++i) {
        auto val = btree.search(static_cast<Key>(i));
        if (val && *val == static_cast<Value>(i * 10)) ++verified;
    }
    std::cout << "  Verified: " << verified << "/" << NUM_INSERTS 
              << (verified == static_cast<int>(NUM_INSERTS) ? " ✅" : " ❌") << "\n";

    std::cout << "\n✅ B-Tree Pager complete. File: " << DB_PATH << "\n";
    return 0;
}
