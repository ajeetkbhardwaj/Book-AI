#pragma once
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <cstdint>

namespace colsdb {

class Benchmark {
 public:
  using Clock = std::chrono::high_resolution_clock;
  
  static auto Now() { return Clock::now(); }
  
  static double ElapsedMs(Clock::time_point start) {
    auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
  }
  
  static void PrintResult(const std::string &label, double avg, 
                         uint64_t bytes, double time_ms, size_t records) {
    double mb = bytes / (1024.0 * 1024.0);
    double throughput = (time_ms > 0) ? (mb / time_ms * 1000) : 0;
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n[" << label << "]\n";
    std::cout << "  Avg Salary:   $" << avg << "\n";
    std::cout << "  Bytes Read:   " << bytes << " (" << mb << " MB)\n";
    std::cout << "  I/O Time:     " << time_ms << " ms\n";
    std::cout << "  Throughput:   " << throughput << " MB/s\n";
    std::cout << "  Bytes/Record: " << (records ? bytes / records : 0) << "\n";
  }
};
}  // namespace colsdb