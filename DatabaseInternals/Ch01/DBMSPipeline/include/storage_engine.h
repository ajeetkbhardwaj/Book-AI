#pragma once
#include "interfaces.h"
#include <iostream>
#include <thread>
#include <chrono>

class DiskStorageEngine : public IStorageEngine {
public:
    std::optional<Page> read_page(int page_id) override {
        log("Reading page from disk...", page_id);
        std::this_thread::sleep_for(std::chrono::milliseconds(5)); // Simulate disk latency

        // Simulate persistent store (in real DB: block device / file)
        if (store_.count(page_id)) {
            return store_[page_id];
        }
        // Page doesn't exist yet (first read)
        return std::nullopt;
    }

    void write_page(const Page& page) override {
        log("Writing page to disk...", page.page_id);
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
        store_[page.page_id] = page;
    }

private:
    std::unordered_map<int, Page> store_;

    void log(const std::string& msg, int page_id) {
        std::cout << "[DISK] " << msg << " (Page " << page_id << ")\n";
    }
};
