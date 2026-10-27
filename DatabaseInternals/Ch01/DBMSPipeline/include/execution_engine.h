#pragma once
#include "interfaces.h"
#include <iostream>

class BasicExecutionEngine : public IExecutionEngine {
public:
    BasicExecutionEngine(std::shared_ptr<ILockManager> lock_mgr,
                         std::shared_ptr<IBufferManager> buf_mgr)
        : lock_mgr_(std::move(lock_mgr)), buf_mgr_(std::move(buf_mgr)) {}

    std::string execute(const ExecutionPlan& plan) override {
        std::cout << "[EXEC] Executing plan: " << (plan.op == OpType::READ ? "READ" : "WRITE") 
                  << " ID=" << plan.target_id << "\n";

        // 1. Acquire Lock
        LockType lock_type = (plan.op == OpType::READ) ? LockType::SHARED : LockType::EXCLUSIVE;
        if (!lock_mgr_->acquire(plan.target_id, lock_type)) {
            return "ERROR: Lock acquisition failed";
        }

        // 2. Pin Page in Buffer Pool
        Page* page = buf_mgr_->pin_page(plan.target_id);

        std::string result;
        if (plan.op == OpType::READ) {
            result = page->data;
            std::cout << "[EXEC] Read data: \"" << result << "\"\n";
            buf_mgr_->unpin_page(plan.target_id, false);
        } else {
            page->data = plan.payload;
            std::cout << "[EXEC] Wrote data: \"" << plan.payload << "\"\n";
            buf_mgr_->unpin_page(plan.target_id, true); // Mark dirty
        }

        // 3. Release Lock
        lock_mgr_->release(plan.target_id);

        return result;
    }

private:
    std::shared_ptr<ILockManager> lock_mgr_;
    std::shared_ptr<IBufferManager> buf_mgr_;
};
