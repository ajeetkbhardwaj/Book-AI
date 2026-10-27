#include "col_store.hpp"
#include <fstream>
#include <chrono>
#include <iostream>
#include <vector>
#include <cstring>

namespace colsdb {

bool ColumnStore::WriteToFile(const std::string &base_path,
                             const std::vector<Employee> &employees) {
  // Extract salary column (SoA transformation)
  std::vector<double> salaries;
  std::vector<int32_t> ids;
  std::vector<char> names(employees.size() * 32);
  std::vector<char> depts(employees.size() * 16);

  salaries.reserve(employees.size());
  ids.reserve(employees.size());
  
  for (size_t i = 0; i < employees.size(); ++i) {
    const auto &e = employees[i];
    salaries.push_back(e.salary);
    ids.push_back(e.id);
    std::memcpy(&names[i * 32], e.name, 32);
    std::memcpy(&depts[i * 16], e.department, 16);
  }
  
  // Write salary column
  std::ofstream sal_out(ColFile(base_path, "salary"), std::ios::binary | std::ios::trunc);
  if (!sal_out) {
    std::cerr << "Failed to write salary column\n";
    return false;
  }
  sal_out.write(reinterpret_cast<const char*>(salaries.data()), salaries.size() * sizeof(double));
  
  // Write ID column
  std::ofstream id_out(ColFile(base_path, "id"), std::ios::binary | std::ios::trunc);
  id_out.write(reinterpret_cast<const char*>(ids.data()), ids.size() * sizeof(int32_t));
  
  // Write name column
  std::ofstream name_out(ColFile(base_path, "name"), std::ios::binary | std::ios::trunc);
  name_out.write(names.data(), names.size());
  
  // Write department column
  std::ofstream dept_out(ColFile(base_path, "department"), std::ios::binary | std::ios::trunc);
  dept_out.write(depts.data(), depts.size());
  
  return sal_out.good() && id_out.good() && name_out.good() && dept_out.good();
}

bool ColumnStore::ReadAll(const std::string &base_path, std::vector<Employee> &out_employees) {
  std::ifstream id_in(ColFile(base_path, "id"), std::ios::binary | std::ios::ate);
  if (!id_in) return false;
  
  size_t num_employees = id_in.tellg() / sizeof(int32_t);
  id_in.seekg(0, std::ios::beg);
  out_employees.resize(num_employees);
  
  std::vector<int32_t> ids(num_employees);
  id_in.read(reinterpret_cast<char*>(ids.data()), num_employees * sizeof(int32_t));
  
  std::vector<double> salaries(num_employees);
  std::ifstream sal_in(ColFile(base_path, "salary"), std::ios::binary);
  sal_in.read(reinterpret_cast<char*>(salaries.data()), num_employees * sizeof(double));
  
  std::vector<char> names(num_employees * 32);
  std::ifstream name_in(ColFile(base_path, "name"), std::ios::binary);
  name_in.read(names.data(), num_employees * 32);
  
  std::vector<char> depts(num_employees * 16);
  std::ifstream dept_in(ColFile(base_path, "department"), std::ios::binary);
  dept_in.read(depts.data(), num_employees * 16);
  
  for (size_t i = 0; i < num_employees; ++i) {
    out_employees[i].id = ids[i];
    out_employees[i].salary = salaries[i];
    std::memcpy(out_employees[i].name, &names[i * 32], 32);
    std::memcpy(out_employees[i].department, &depts[i * 16], 16);
  }
  
  return true;
}

double ColumnStore::QueryAverageSalary(const std::string &base_path,
                                      uint64_t &bytes_read,
                                      double &io_time_ms) {
  std::ifstream in(ColFile(base_path, "salary"), std::ios::binary);
  if (!in) return -1.0;
  
  auto start = std::chrono::high_resolution_clock::now();
  
  constexpr size_t CHUNK = 4096;
  std::vector<double> buffer(CHUNK / sizeof(double)); // Native double buffer
  uint64_t total_read = 0;
  double sum = 0.0;
  size_t count = 0;
  
  while (in.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(double)) || in.gcount() > 0) {
    std::streamsize doubles_read = in.gcount() / sizeof(double);
    total_read += in.gcount();
    for (std::streamsize i = 0; i < doubles_read; ++i) {
      sum += buffer[i];
      ++count;
    }
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  io_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
  bytes_read = total_read;
  
  return count > 0 ? sum / count : 0.0;
}
}  // namespace colsdb