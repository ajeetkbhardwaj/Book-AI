#pragma once
#include <cstdint>
#include <cstring>

// Fixed-size record for predictable binary layout & zero-copy I/O
// Packed to eliminate compiler padding (4+64+8 = 76 bytes exactly)
struct __attribute__((packed)) Record {
    int32_t id;
    char payload[64];
    double metric;

    Record() : id(0), metric(0.0) {
        std::memset(payload, 0, sizeof(payload));
    }
};

// Compile-time size check
static_assert(sizeof(Record) == 76, "Record size mismatch (expect 4 + 64 + 8 = 76)");
