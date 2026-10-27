#include "employee.hpp"
#include "row_store.hpp"
#include "benchmark.cpp"
#include <iostream>
#include <vector>
#include <random>
#include <filesystem>
#include <unordered_map>
#include <algorithm>

namespace fs = std::filesystem;
using namespace rowsdb;

// Query 2: Filtered aggregation (WHERE department = 'Engineering')
double QueryAvgSalaryByDept(const std::string &filename, 
                           const std::string &dept_filter,
                           uint64_t &bytes_read,
                           double &io_time_ms) {
  std::ifstream in(filename, std::ios::binary);
  if (!in) return -1.0;
  
  auto start = Benchmark::Now();
  constexpr size_t CHUNK = 4096;
  std::vector<char> buffer(CHUNK);
  uint64_t total_read = 0;
  double sum = 0.0;
  size_t count = 0;
  
  while (in) {
    in.read(buffer.data(), CHUNK);
    std::streamsize bytes = in.gcount();
    if (bytes <= 0) break;
    total_read += bytes;
    
    for (std::streamsize off = 0; 
         off + static_cast<std::streamsize>(sizeof(Employee)) <= bytes;
         off += sizeof(Employee)) {
      const Employee *emp = reinterpret_cast<const Employee*>(buffer.data() + off);
      // Compare department (null-terminated string comparison)
      if (std::strncmp(emp->department, dept_filter.c_str(), sizeof(emp->department)) == 0) {
        sum += emp->salary;
        ++count;
      }
    }
  }
  
  io_time_ms = Benchmark::ElapsedMs(start);
  bytes_read = total_read;  // Still reads ALL data, even unused rows
  return count > 0 ? sum / count : 0.0;
}

int main() {
  constexpr size_t N = 1'000'000;
  constexpr uint32_t SEED = 42;
  const std::string FILE = "row_store.bin";

  std::cout << "=== Row-Oriented Storage Benchmark ===\n";
  std::cout << "Records: " << N << "\n";
  std::cout << "Query:   SELECT AVG(salary)\n\n";

  std::cout << "[1/3] Generating " << N << " employees...\n";
  std::mt19937 rng(SEED);
  std::vector<Employee> emps;
  emps.reserve(N);
  for (size_t i = 0; i < N; ++i) {
    emps.push_back(Employee::GenerateRandom(rng, static_cast<int32_t>(i)));
  }

  std::cout << "[2/3] Writing row_store.bin...\n";
  auto t0 = rowsdb::Benchmark::Now();
  if (!RowStore::WriteToFile(FILE, emps)) {
    std::cerr << "Write failed\n";
    return 1;
  }
  // Use GetExpectedFileSize
  std::cout << "  Written: " << RowStore::GetExpectedFileSize(N) << " bytes\n";
  std::cout << "  Time: " << rowsdb::Benchmark::ElapsedMs(t0) << " ms\n\n";

  std::cout << "[3/3] Querying AVG(salary)...\n";
  uint64_t bytes;
  double time_ms;
  double avg = RowStore::QueryAverageSalary(FILE, bytes, time_ms);

  rowsdb::Benchmark::PrintResult("Row-Oriented (AoS)", avg, bytes, time_ms, N);

  std::cout << "[4/4] Viewing and Analyzing Database...\n";
  std::vector<Employee> db_records;
  if (RowStore::ReadAll(FILE, db_records)) {
    std::cout << "  Successfully loaded " << db_records.size() << " records into memory.\n";
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

  std::cout << "\n Done. File: " << FILE << " (~"
            << (RowStore::GetExpectedFileSize(N) / (1024*1024)) << " MB)\n";
  return 0;
}