#pragma once

#include <cstdint>
#include <cstring>
#include <random>

namespace colsdb {

// Fixed-size struct for predictable layout
struct Employee {
    int32_t id;                    // 4 bytes
    char name[32];                 // 32 bytes
    char department[16];           // 16 bytes  
    double salary;                 // 8 bytes
                                   // Total: 60 bytes (pads to 64 bytes)

    Employee() : id(0), salary(0.0) {
        std::memset(name, 0, sizeof(name));
        std::memset(department, 0, sizeof(department));
    }

    static Employee GenerateRandom(std::mt19937 &rng, int32_t start_id) {
        Employee emp;
        emp.id = start_id;
        
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
}  // namespace colsdb