#include "page_manager.h"
#include <filesystem>
#include <cstring>
#include <iostream>

PageManager::PageManager(const std::string& db_path, bool create_new)
    : db_path_(db_path), root_id_(0), next_free_id_(0), file_size_pages_(0) {
    
    if (create_new || !std::filesystem::exists(db_path)) {
        file_.open(db_path, std::ios::binary | std::ios::out | std::ios::in | std::ios::trunc);
        if (!file_.is_open()) throw std::runtime_error("Cannot create DB file");
        
        // Reserve page 0 for metadata
        std::memset(&metadata_page_, 0, sizeof(metadata_page_));
        metadata_page_.type = NodeType::INTERNAL;
        file_size_pages_ = 1;
        
        // Write initial metadata page to disk
        disk_write(0, metadata_page_);
        
        // Allocate the initial root page (will be page 1)
        PageID root = allocate_page();
        Page root_page;
        std::memset(&root_page, 0, sizeof(root_page));
        root_page.type = NodeType::LEAF;
        root_page.count = 0;
        root_page.parent_id = 0;
        root_page.next_sibling_id = 0;
        write_page(root, root_page);
        
        root_id_ = root;
        write_metadata();
        flush();
    } else {
        file_.open(db_path, std::ios::binary | std::ios::in | std::ios::out);
        if (!file_.is_open()) throw std::runtime_error("Cannot open DB file");
        
        file_.seekg(0, std::ios::end);
        auto size = file_.tellg();
        file_size_pages_ = static_cast<uint32_t>(size / PAGE_SIZE);
        
        read_metadata();
    }
}

PageManager::~PageManager() {
    if (file_.is_open()) {
        write_metadata();
        flush();
        file_.close();
    }
}

// ===== Disk I/O (raw, no cache) =====

void PageManager::disk_read(PageID id, Page& out_page) const {
    file_.seekg(static_cast<std::streamoff>(id) * PAGE_SIZE);
    if (!file_.read(reinterpret_cast<char*>(&out_page), PAGE_SIZE)) {
        file_.clear(); // Clear error flags for recovery
        throw std::runtime_error("Disk read failed for page " + std::to_string(id));
    }
}

void PageManager::disk_write(PageID id, const Page& page) {
    file_.seekp(static_cast<std::streamoff>(id) * PAGE_SIZE);
    if (!file_.write(reinterpret_cast<const char*>(&page), PAGE_SIZE)) {
        throw std::runtime_error("Disk write failed for page " + std::to_string(id));
    }
}

// ===== Cached Page I/O =====

void PageManager::read_page(PageID id, Page& out_page) const {
    // Check cache first
    auto it = cache_.find(id);
    if (it != cache_.end()) {
        out_page = it->second.page;
        return;
    }
    
    // Cache miss: read from disk and cache
    if (id >= file_size_pages_) {
        throw std::out_of_range("Page ID " + std::to_string(id) + 
                                " out of bounds (max " + std::to_string(file_size_pages_) + ")");
    }
    CachedPage& cached = cache_[id];
    disk_read(id, cached.page);
    cached.dirty = false;
    out_page = cached.page;
}

void PageManager::write_page(PageID id, const Page& page) {
    if (id >= file_size_pages_) {
        throw std::out_of_range("Page ID " + std::to_string(id) + 
                                " out of bounds (max " + std::to_string(file_size_pages_) + ")");
    }
    CachedPage& cached = cache_[id];
    cached.page = page;
    cached.dirty = true;
}

void PageManager::flush() {
    for (auto& [id, cached] : cache_) {
        if (cached.dirty) {
            disk_write(id, cached.page);
            cached.dirty = false;
        }
    }
    // Also write metadata
    disk_write(0, metadata_page_);
    file_.flush();
}

// ===== Metadata =====

void PageManager::read_metadata() {
    disk_read(0, metadata_page_);
    root_id_ = static_cast<PageID>(metadata_page_.keys[0]);
    next_free_id_ = static_cast<PageID>(metadata_page_.keys[1]);
}

void PageManager::write_metadata() {
    metadata_page_.keys[0] = root_id_;
    metadata_page_.keys[1] = next_free_id_;
    // Keep metadata in cache too
    cache_[0] = {metadata_page_, true};
}

// ===== Page Allocation =====

PageID PageManager::allocate_page() {
    if (next_free_id_ != 0) {
        PageID id = next_free_id_;
        Page free_pg;
        read_page(id, free_pg);
        next_free_id_ = static_cast<PageID>(free_pg.keys[0]);
        
        // Zero out the reused page
        std::memset(&free_pg, 0, sizeof(free_pg));
        write_page(id, free_pg);
        return id;
    }
    
    // Extend file: write a zeroed page at the new position
    PageID new_id = file_size_pages_++;
    Page empty_page;
    std::memset(&empty_page, 0, sizeof(empty_page));
    // Write directly to disk to extend the file
    disk_write(new_id, empty_page);
    // Also put in cache
    cache_[new_id] = {empty_page, false};
    return new_id;
}

void PageManager::free_page(PageID id) {
    if (id == 0 || id == root_id_) return;
    
    Page page;
    std::memset(&page, 0, sizeof(page));
    page.keys[0] = next_free_id_;
    next_free_id_ = id;
    write_page(id, page);
}

void PageManager::set_root(PageID id) {
    root_id_ = id;
    write_metadata();
}
