#pragma once
#include "types.h"
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>

class PageManager {
public:
    explicit PageManager(const std::string& db_path, bool create_new = true);
    ~PageManager();

    // Allocate a new 4KB page from free list or extend file
    PageID allocate_page();
    void free_page(PageID id);
    
    // Read/Write pages (goes through in-memory cache)
    void read_page(PageID id, Page& out_page) const;
    void write_page(PageID id, const Page& page);
    
    // Flush all dirty cached pages to disk
    void flush();
    
    uint32_t root_id() const { return root_id_; }
    void set_root(PageID id);

    uint32_t page_count() const { return file_size_pages_; }

private:
    mutable std::fstream file_;
    std::string db_path_;
    PageID root_id_;
    Page metadata_page_;
    uint32_t next_free_id_;
    uint32_t file_size_pages_;

    // In-memory page cache (write-back)
    struct CachedPage {
        Page page;
        bool dirty = false;
    };
    mutable std::unordered_map<PageID, CachedPage> cache_;

    void disk_read(PageID id, Page& out_page) const;
    void disk_write(PageID id, const Page& page);
    void read_metadata();
    void write_metadata();
};
