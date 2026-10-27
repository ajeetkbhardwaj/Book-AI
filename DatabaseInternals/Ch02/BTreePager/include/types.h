#pragma once
#include <cstdint>
#include <cstddef>

constexpr uint32_t PAGE_SIZE = 4096;
constexpr uint16_t MAX_ENTRIES_PER_PAGE = 200; // Fits comfortably in 4KB with header

using PageID = uint32_t;
using Key = int64_t;
using Value = int64_t;

enum class NodeType : uint8_t {
    INTERNAL = 0,
    LEAF = 1
};

// Disk-resident page layout (packed to avoid padding)
// Header: 1(type) + 2(count) + 4(parent) + 4(sibling) + 8(reserved) = 19 bytes
// Keys: 200 * 8 = 1600 bytes  |  Union: max(200*8, 201*4) = 1600 bytes
// Total body: 19 + 1600 + 1600 = 3219 → pad to 4096
#pragma pack(push, 1)
struct Page {
    NodeType type;
    uint16_t count;          // Number of keys currently stored
    PageID parent_id;
    PageID next_sibling_id;  // For future range scans
    uint8_t reserved[8];     // Align header to 16 bytes

    Key keys[MAX_ENTRIES_PER_PAGE];
    
    // Union-like behavior: 
    // If LEAF: values[] holds actual data
    // If INTERNAL: child_ids[] holds PageID pointers (size = count + 1)
    union {
        Value values[MAX_ENTRIES_PER_PAGE];
        PageID child_ids[MAX_ENTRIES_PER_PAGE + 1];
    };
    
    // Explicit padding to make struct exactly PAGE_SIZE for correct disk I/O
    uint8_t _pad[877]; // 4096 - 3219 = 877
};
#pragma pack(pop)

static_assert(sizeof(Page) == PAGE_SIZE, "Page struct must be exactly 4KB");

struct PageHeader {
    NodeType type;
    uint16_t count;
    PageID parent_id;
};
