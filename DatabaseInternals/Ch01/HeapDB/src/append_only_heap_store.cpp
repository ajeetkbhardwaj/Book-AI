#include "append_only_heap_store.h"
#include <iostream>

AppendOnlyHeapStore::AppendOnlyHeapStore(const std::string& filepath)
    : filepath_(filepath) {
    // Open for read/write in binary mode. Create if missing.
    file_.open(filepath, std::ios::binary | std::ios::in | std::ios::out);
    if (!file_.is_open()) {
        // File doesn't exist yet: create it
        file_.open(filepath, std::ios::binary | std::ios::out | std::ios::trunc);
        if (!file_.is_open()) {
            throw std::runtime_error("Failed to create data file: " + filepath);
        }
        file_.close();
        file_.open(filepath, std::ios::binary | std::ios::in | std::ios::out);
    }
    // Position write pointer at end for append-only behavior
    file_.seekp(0, std::ios::end);
}

AppendOnlyHeapStore::~AppendOnlyHeapStore() {
    if (file_.is_open()) file_.close();
}

void AppendOnlyHeapStore::Insert(const Record& record) {
    // Ensure we're at the logical end before appending
    file_.seekp(0, std::ios::end);
    
    // Capture byte offset before writing
    auto pos = file_.tellp();
    if (pos == -1) throw std::runtime_error("Failed to get file position");

    // Zero-copy binary write
    file_.write(reinterpret_cast<const char*>(&record), sizeof(Record));
    if (!file_.good()) throw std::runtime_error("Disk write failed");

    // Update in-memory hash index
    index_[record.id] = pos;
}

std::optional<Record> AppendOnlyHeapStore::Lookup(int32_t id) {
    auto it = index_.find(id);
    if (it == index_.end()) return std::nullopt;

    // Jump directly to record offset
    file_.seekg(it->second);
    if (!file_.good()) return std::nullopt;

    Record record;
    file_.read(reinterpret_cast<char*>(&record), sizeof(Record));
    if (!file_.good()) return std::nullopt;

    return record;
}

size_t AppendOnlyHeapStore::IndexSize() const {
    return index_.size();
}

void AppendOnlyHeapStore::Sync() {
    file_.flush();
    // POSIX: file_.rdbuf()->pubsync(); // Ensures data hits physical disk
}
