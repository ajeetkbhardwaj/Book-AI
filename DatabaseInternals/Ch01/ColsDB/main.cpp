#include "employee.hpp"
#include "col_store.hpp"
#include "benchmark.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <unordered_map>
#include <algorithm>

using namespace colsdb;
// Query 2: Filtered aggregation (column-store can skip non-matching rows AFTER reading dept column)
double QueryAvgSalaryByDept(const std::string &base_path,
                           const std::string &dept_filter,
                           uint64_t &bytes_read,
                           double &io_time_ms) {
  // Step 1: Read department column to find matching row indices
  std::ifstream dept_in(ColumnStore::ColFile(base_path, "dept"), std::ios::binary);
  if (!dept_in) return -1.0;
  
  dept_in.seekg(0, std::ios::end);
  size_t num_records = dept_in.tellg() / 16;  // 16 bytes per dept field
  dept_in.seekg(0, std::ios::beg);
  
  std::vector<bool> matches(num_records, false);
  std::vector<char> buffer(4096);
  
  while (dept_in) {
    dept_in.read(buffer.data(), 4096);
    std::streamsize bytes = dept_in.gcount();
    if (bytes <= 0) break;
    
    for (std::streamsize off = 0; off + 16 <= bytes; off += 16) {
      size_t idx = (dept_in.tellg() - bytes + off) / 16;
      if (std::strncmp(buffer.data() + off, dept_filter.c_str(), 16) == 0) {
        matches[idx] = true;
      }
    }
  }
  
  // Step 2: Read ONLY salary values for matching rows
  std::ifstream sal_in(ColumnStore::ColFile(base_path, "salary"), std::ios::binary);
  if (!sal_in) return -1.0;
  
  auto start = Benchmark::Now();
  uint64_t total_read = 0;
  double sum = 0.0;
  size_t count = 0;
  
  for (size_t i = 0; i < num_records; ++i) {
    if (matches[i]) {
      double salary;
      sal_in.seekg(i * sizeof(double));
      sal_in.read(reinterpret_cast<char*>(&salary), sizeof(double));
      total_read += sizeof(double);
      sum += salary;
      ++count;
    }
  }
  
  io_time_ms = Benchmark::ElapsedMs(start);
  bytes_read = total_read;  // Only reads salary for matching rows!
  return count > 0 ? sum / count : 0.0;
}

int main() {
  constexpr size_t N = 1'000'000;
  constexpr uint32_t SEED = 42;
  const std::string BASE = "col_store";  // Generates col_store_salary.bin, etc.
  
  std::cout << "=== Column-Oriented Storage Benchmark ===\n";
  std::cout << "Records: " << N << "\n";
  std::cout << "Query:   SELECT AVG(salary)\n\n";
  
  // Generate data (same as RowsDB for fair comparison)
  std::cout << "[1/4] Generating " << N << " employees...\n";
  std::mt19937 rng(SEED);
  std::vector<Employee> emps;
  emps.reserve(N);
  for (size_t i = 0; i < N; ++i) {
    emps.push_back(Employee::GenerateRandom(rng, static_cast<int32_t>(i)));
  }
  
  // Write column files
  std::cout << "[2/4] Writing column files...\n";
  auto t0 = Benchmark::Now();
  if (!ColumnStore::WriteToFile(BASE, emps)) {
    std::cerr << "Write failed\n";
    return 1;
  }
  std::cout << "  Written: ~" << ColumnStore::QueryBytes(N) 
            << " bytes (salary column only)\n";
  std::cout << "  Time: " << Benchmark::ElapsedMs(t0) << " ms\n\n";
  
  // Query
  std::cout << "[3/4] Querying AVG(salary)...\n";
  uint64_t bytes;
  double time_ms;
  double avg = ColumnStore::QueryAverageSalary(BASE, bytes, time_ms);
  
  Benchmark::PrintResult("Column-Oriented (SoA)", avg, bytes, time_ms, N);
  
  std::cout << "\n[4/4] Viewing and Analyzing Database...\n";
  std::vector<Employee> db_records;
  if (ColumnStore::ReadAll(BASE, db_records)) {
    std::cout << "  Successfully loaded " << db_records.size() << " records by reconstructing rows from columns.\n";
    std::cout << "  --- Sample (First 5 Records) ---\n";
    for (size_t i = 0; i < std::min<size_t>(5, db_records.size()); ++i) {
      const auto& emp = db_records[i];
      std::cout << "  ID: " << emp.id 
                << " | Name: " << emp.name 
                << " | Dept: " << emp.department 
                << " | Salary: $" << emp.salary << "\n";
    }
    
    std::cout << "\n  --- Department Analysis (Group By) ---\n";
    std::unordered_map<std::string, size_t> dept_counts;
    for (const auto& emp : db_records) {
      dept_counts[emp.department]++;
    }
    for (const auto& [dept, count] : dept_counts) {
      std::cout << "  " << dept << ": " << count << " employees\n";
    }
  }

  std::cout << "\n Done. Files: " << ColumnStore::ColFile(BASE, "salary") 
            << ", " << ColumnStore::ColFile(BASE, "id") << " etc.\n";
  return 0;
}