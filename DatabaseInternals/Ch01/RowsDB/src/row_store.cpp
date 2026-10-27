#include "row_store.hpp"

#include <fstream>
#include <chrono>
#include <iostream>

namespace rowsdb {

bool RowStore::WriteToFile(const std::string &filename, const std::vector<Employee> &employees) {
  std::ofstream out(filename, std::ios::binary | std::ios::trunc);
  if (!out) { std::cerr << "Failed to open " << filename << "\n"; return false; }
  out.write(reinterpret_cast<const char*>(employees.data()), employees.size() * sizeof(Employee));
  return out.good();
}

bool RowStore::ReadAll(const std::string &filename, std::vector<Employee> &out_employees) {
  std::ifstream in(filename, std::ios::binary);
  if (!in) return false;

  in.seekg(0, std::ios::end);
  uint64_t file_size = in.tellg();
  in.seekg(0, std::ios::beg);

  out_employees.resize(file_size / sizeof(Employee));
  in.read(reinterpret_cast<char*>(out_employees.data()), file_size);
  return in.good();
}

double RowStore::QueryAverageSalary(const std::string &filename, uint64_t &bytes_read, double &io_time_ms) {
  std::ifstream in(filename, std::ios::binary);
  if (!in) return -1.0;

  in.seekg(0, std::ios::end);
  [[maybe_unused]] uint64_t file_size = in.tellg();
  in.seekg(0, std::ios::beg);

  auto start = std::chrono::high_resolution_clock::now();

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

    for (std::streamsize off = 0; off + static_cast<std::streamsize>(sizeof(Employee)) <= bytes; off += sizeof(Employee)) {
      const Employee *emp = reinterpret_cast<const Employee*>(buffer.data() + off);
      sum += emp->salary;
      ++count;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  io_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
  bytes_read = total_read;
  return count > 0 ? sum / count : 0.0;
}

}  // namespace rowsdb