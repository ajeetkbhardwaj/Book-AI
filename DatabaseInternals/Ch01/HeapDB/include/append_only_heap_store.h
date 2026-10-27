#pragma once
#include "record.h"
#include <fstream>
#include <unordered_map>
#include <optional>
#include <string>
#include <stdexcept>

class AppendOnlyHeapStore {
public:
    explicit AppendOnlyHeapStore(const std::string& filepath);
    ~AppendOnlyHeapStore();

    // Append record to end of file; O(1) amortized
    void Insert(const Record& record);
    
    // Point lookup via in-memory hash index; O(1) average
    std::optional<Record> Lookup(int32_t id);
    
    size_t IndexSize() const;
    
    // Flush OS buffers to disk
    void Sync();

private:
    std::string filepath_;
    std::fstream file_;
    std::unordered_map<int32_t, std::streampos> index_;
};
