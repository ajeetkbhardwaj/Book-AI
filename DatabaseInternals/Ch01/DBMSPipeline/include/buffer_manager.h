#pragma once
#include "interfaces.h"
#include <iostream>
#include <unordered_map>

class LRUBufferManager : public IBufferManager {
public:
    explicit LRUBufferManager(std::shared_ptr<IStorageEngine> disk) : disk_(std::move(disk)) {}

    Page* pin_page(int page_id) override {
        // Check if already in buffer pool
        auto it = pool_.find(page_id);
        if (it != pool_.end()) {
            log("Cache HIT", page_id);
            pin_counts_[page_id]++;
            return &it->second;
        }

        log("Cache MISS", page_id);
        // Fetch from disk
        auto disk_page = disk_->read_page(page_id);
        if (!disk_page) {
            // Page doesn't exist on disk yet; allocate empty
            disk_page = Page{page_id, "", false};
        }

        pool_[page_id] = *disk_page;
        pin_counts_[page_id] = 1;
        return &pool_[page_id];
    }

    void unpin_page(int page_id, bool dirty) override {
        if (pin_counts_[page_id] > 0) {
            pin_counts_[page_id]--;
        }
        if (dirty) {
            pool_[page_id].is_dirty = true;
        }
        if (pin_counts_[page_id] == 0) {
            log("Unpinned (candidate for eviction)", page_id);
            // Real LRU would move to free list; simplified here
        }
    }

    void flush_dirty() {
        for (auto& [id, page] : pool_) {
            if (page.is_dirty) {
                disk_->write_page(page);
                page.is_dirty = false;
            }
        }
    }

private:
    std::shared_ptr<IStorageEngine> disk_;
    std::unordered_map<int, Page> pool_;
    std::unordered_map<int, int> pin_counts_;

    void log(const std::string& action, int page_id) {
        std::cout << "[BUF] " << action << " (Page " << page_id << ")\n";
    }
};
