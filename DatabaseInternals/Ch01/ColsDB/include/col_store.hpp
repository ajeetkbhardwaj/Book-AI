#pragma once

#include "employee.hpp"
#include <vector>
#include <string>
#include <cstdint>

namespace colsdb {

class ColumnStore {
 public:
  // Write data as separate column files (SoA - Structure of Arrays)
  static bool WriteToFile(const std::string &base_path, 
                         const std::vector<Employee> &employees);
  
  // Read all columns back into memory (Tuple Reconstruction)
  static bool ReadAll(const std::string &base_path, 
                     std::vector<Employee> &out_employees);
  
  // Query: Compute average salary (reads ONLY salary column)
  static double QueryAverageSalary(const std::string &base_path,
                                  uint64_t &bytes_read,
                                  double &io_time_ms);
  
  static std::string ColFile(const std::string &base, const std::string &col) {
    return base + "_" + col + ".bin";
  }
  
  // For our query, only the salary column matters (8 bytes each)
  static uint64_t QueryBytes(size_t num_employees) {
    return sizeof(double) * num_employees;
  }
};
}  // namespace colsdb