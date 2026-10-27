#pragma once

#include "employee.hpp"
#include <vector>
#include <string>
#include <cstdint>

namespace rowsdb {

class RowStore {
 public:
  // Write entire array as contiguous Array-of-Structs (AoS)
  static bool WriteToFile(const std::string &filename, 
                         const std::vector<Employee> &employees);
  
  // Read entire file into memory (simulates SELECT *)
  static bool ReadAll(const std::string &filename, 
                     std::vector<Employee> &out_employees);
  
  // Query: Compute average salary (reads FULL records)
  static double QueryAverageSalary(const std::string &filename, 
                                  uint64_t &bytes_read,
                                  double &io_time_ms);
  // ✅ Fixed: Matched name used in main.cpp
  static uint64_t GetExpectedFileSize(size_t num_employees) {
    return sizeof(Employee) * num_employees;
    }
};

}  // namespace rowsdb