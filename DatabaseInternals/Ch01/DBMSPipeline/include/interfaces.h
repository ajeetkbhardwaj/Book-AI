#pragma once
#include <memory>
#include <string>
#include <optional>
#include <unordered_map>

// ===== Data Structures =====
struct Page {
    int page_id;
    std::string data;
    bool is_dirty = false;
};

enum class LockType { SHARED, EXCLUSIVE };
enum class OpType { READ, WRITE };

struct ExecutionPlan {
    OpType op;
    int target_id;
    std::string payload; // For WRITE operations
};

// ===== Abstract Interfaces =====
class ILockManager {
public:
    virtual bool acquire(int resource_id, LockType type) = 0;
    virtual void release(int resource_id) = 0;
    virtual ~ILockManager() = default;
};

class IStorageEngine {
public:
    virtual std::optional<Page> read_page(int page_id) = 0;
    virtual void write_page(const Page& page) = 0;
    virtual ~IStorageEngine() = default;
};

class IBufferManager {
public:
    virtual Page* pin_page(int page_id) = 0;
    virtual void unpin_page(int page_id, bool dirty) = 0;
    virtual ~IBufferManager() = default;
};

class IQueryProcessor {
public:
    virtual std::unique_ptr<ExecutionPlan> parse(const std::string& raw_query) = 0;
    virtual ~IQueryProcessor() = default;
};

class IExecutionEngine {
public:
    virtual std::string execute(const ExecutionPlan& plan) = 0;
    virtual ~IExecutionEngine() = default;
};
