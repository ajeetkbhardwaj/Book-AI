#include "interfaces.h"
#include "lock_manager.h"
#include "storage_engine.h"
#include "buffer_manager.h"
#include "query_processor.h"
#include "execution_engine.h"
#include <iostream>
#include <memory>
#include <vector>

int main() {
    std::cout << "=== DBMS Subsystem Pipeline ===\n\n";

    // 1. Instantiate components (Dependency Injection)
    auto disk = std::make_shared<DiskStorageEngine>();
    auto buf_mgr = std::make_shared<LRUBufferManager>(disk);
    auto lock_mgr = std::make_shared<SimpleLockManager>();
    auto parser = std::make_unique<SimpleQueryProcessor>();
    auto engine = std::make_unique<BasicExecutionEngine>(lock_mgr, buf_mgr);

    // 2. Simulate Transport Layer receiving requests
    std::vector<std::string> requests = {
        "READ 50",          // Cache miss → disk fetch
        "READ 50",          // Cache hit
        "WRITE 50 hello",   // Exclusive lock → modify in buffer → mark dirty
        "READ 50"           // Should see updated data
    };

    for (const auto& req : requests) {
        std::cout << "\n>>> TRANSPORT: Received request: \"" << req << "\"\n";
        
        try {
            auto plan = parser->parse(req);
            std::string result = engine->execute(*plan);
            std::cout << ">>> RESPONSE: \"" << result << "\"\n";
        } catch (const std::exception& e) {
            std::cerr << ">>> ERROR: " << e.what() << "\n";
        }
        std::cout << "----------------------------------------\n";
    }

    // 3. Buffer Manager checkpoint (flush dirty pages to disk)
    std::cout << "\n[CHECKPOINT] Flushing dirty pages to disk...\n";
    buf_mgr->flush_dirty();
    std::cout << "✅ Pipeline complete. System ready for shutdown.\n";
    
    return 0;
}
