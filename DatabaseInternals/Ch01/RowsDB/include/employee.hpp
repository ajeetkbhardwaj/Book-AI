#pragma once

#include <cstdint>
#include <string>
#include <random>
#include <cstring>

namespace rowsdb {

// Fixed-size fields for predictable binary layout (no padding surprises)
struct Employee {
    int32_t id;                    // 4 bytes
    char name[32];                 // 32 bytes (null-terminated)
    char department[16];           // 16 bytes (null-terminated)
    double salary;                 // 8 bytes
                                   // Total: 60 bytes (may pad to 64 for alignment)
    
    Employee() : id(0), salary(0.0) {
        std::memset(name, 0, sizeof(name));
        std::memset(department, 0, sizeof(department));
    }
    
    // Helper for random generation
    static Employee GenerateRandom(std::mt19937 &rng, int32_t start_id) {
        Employee emp;
        emp.id = start_id;
        
        // Random names from a small pool (for reproducibility)
        static const char* names[] = {"Alice", "Bob", "Charlie", "Diana", "Eve", 
                                      "Frank", "Grace", "Henry", "Ivy", "Jack"};
        static const char* depts[] = {"Engineering", "Sales", "Marketing", 
                                      "HR", "Finance", "Ops"};
        
        std::uniform_int_distribution<> name_dist(0, 9);
        std::uniform_int_distribution<> dept_dist(0, 5);
        std::uniform_real_distribution<> salary_dist(40000.0, 200000.0);
        
        std::strncpy(emp.name, names[name_dist(rng)], sizeof(emp.name) - 1);
        std::strncpy(emp.department, depts[dept_dist(rng)], sizeof(emp.department) - 1);
        emp.salary = salary_dist(rng);
        
        return emp;
    }
};

// Compile-time size check (catch padding issues early)
static_assert(sizeof(Employee) <= 64, "Employee struct unexpectedly large");

}  // namespace rowsdb