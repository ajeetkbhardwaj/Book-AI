#pragma once
#include "interfaces.h"
#include <mutex>
#include <unordered_map>
#include <iostream>

class SimpleLockManager : public ILockManager {
public:
    bool acquire(int resource_id, LockType type) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = locks_.find(resource_id);
        if (it == locks_.end()) {
            locks_[resource_id] = type;
            log("Acquired", resource_id, type);
            return true;
        }
        // Simplified: allow multiple SHARED, block EXCLUSIVE if SHARED exists, etc.
        if (type == LockType::SHARED) {
            log("Acquired", resource_id, type);
            return true;
        }
        std::cout << "[LOCK] Denied: conflicting lock on ID " << resource_id << "\n";
        return false;
    }

    void release(int resource_id) override {
        std::lock_guard<std::mutex> lock(mutex_);
        locks_.erase(resource_id);
        log("Released", resource_id, LockType::SHARED);
    }

private:
    std::mutex mutex_;
    std::unordered_map<int, LockType> locks_;

    void log(const std::string& action, int id, LockType type) {
        std::cout << "[LOCK] " << action << " (ID: " << id 
                  << ", Type: " << (type == LockType::SHARED ? "S" : "X") << ")\n";
    }
};
