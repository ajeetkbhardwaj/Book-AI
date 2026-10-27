#!/bin/bash
# Comprehensive benchmark comparison for RowsDB vs ColsDB
# Works on macOS (zsh) and Linux (bash)

set -e  # Exit on error

# Colors for output (macOS compatible)
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Comprehensive Storage Benchmark Comparison ===${NC}\n"

# Function to run benchmark and extract ALL metrics
run_and_parse() {
  local project_dir=$1
  local label=$2
  
  echo -e "${YELLOW}Running $label...${NC}" >&2
  cd "$project_dir/build" || exit 1
  
  # Run and capture full output
  local output
  output=$(./$(basename "$project_dir" | tr '[:upper:]' '[:lower:]') 2>&1)
  
  # Extract metrics using grep/sed (macOS-compatible)
  local bytes=$(echo "$output" | grep "Bytes Read:" | sed 's|.*Bytes Read:[[:space:]]*\([0-9]*\).*|\1|')
  local mb=$(echo "$output" | grep "Bytes Read:" | sed 's|.*(\([0-9.]*\) MB).*|\1|')
  local time_ms=$(echo "$output" | grep "I/O Time:" | sed 's|.*I/O Time:[[:space:]]*\([0-9.]*\) ms.*|\1|')
  local throughput=$(echo "$output" | grep "Throughput:" | sed 's|.*Throughput:[[:space:]]*\([0-9.]*\) MB/s.*|\1|')
  local bytes_per_rec=$(echo "$output" | grep "Bytes/Record:" | sed 's|.*Bytes/Record:[[:space:]]*\([0-9]*\).*|\1|')
  local avg_salary=$(echo "$output" | grep "Avg Salary:" | sed 's|.*Avg Salary:[[:space:]]*\$\([0-9.]*\).*|\1|')
  
  cd - > /dev/null
  
  # Return as space-separated values
  echo "$bytes $mb $time_ms $throughput $bytes_per_rec $avg_salary"
}

# Run both benchmarks
echo "[1/2] Executing Row-Oriented benchmark..."
ROW_METRICS=($(run_and_parse "RowsDB" "Row-Oriented"))
ROW_BYTES=${ROW_METRICS[0]}
ROW_MB=${ROW_METRICS[1]}
ROW_TIME=${ROW_METRICS[2]}
ROW_THROUGHPUT=${ROW_METRICS[3]}
ROW_BPR=${ROW_METRICS[4]}
ROW_AVG=${ROW_METRICS[5]}

echo "[2/2] Executing Column-Oriented benchmark..."
COL_METRICS=($(run_and_parse "ColsDB" "Column-Oriented"))
COL_BYTES=${COL_METRICS[0]}
COL_MB=${COL_METRICS[1]}
COL_TIME=${COL_METRICS[2]}
COL_THROUGHPUT=${COL_METRICS[3]}
COL_BPR=${COL_METRICS[4]}
COL_AVG=${COL_METRICS[5]}

# Print detailed results table
echo -e "\n${GREEN}=== Detailed Metrics Comparison ===${NC}"
printf "%-25s %15s %15s %15s\n" "Metric" "Row-Oriented" "Column-Oriented" "Improvement"
printf "%-25s %15s %15s %15s\n" "-------------------------" "---------------" "---------------" "---------------"
SALARY_MATCH="Diff"
if [[ "$(echo "$ROW_AVG == $COL_AVG" | bc -l 2>/dev/null)" == "1" ]]; then
  SALARY_MATCH="✓ Match"
fi
printf "%-25s %15s %15s %15s\n" "Avg Salary" "\$${ROW_AVG}" "\$${COL_AVG}" "$SALARY_MATCH"
printf "%-25s %15s %15s %15s\n" "Bytes Read" "${ROW_BYTES} (${ROW_MB} MB)" "${COL_BYTES} (${COL_MB} MB)" ""
printf "%-25s %15s %15s %15s\n" "I/O Time" "${ROW_TIME} ms" "${COL_TIME} ms" ""
printf "%-25s %15s %15s %15s\n" "Throughput" "${ROW_THROUGHPUT} MB/s" "${COL_THROUGHPUT} MB/s" ""
printf "%-25s %15s %15s %15s\n" "Bytes/Record" "${ROW_BPR}" "${COL_BPR}" ""

# Compute derived metrics
if [[ -n "$ROW_BYTES" && -n "$COL_BYTES" && -n "$ROW_TIME" && -n "$COL_TIME" ]]; then
  echo -e "\n${GREEN}=== Computed Efficiency Metrics ===${NC}"
  
  # I/O Reduction
  IO_REDUCTION=$(echo "scale=2; (1 - $COL_BYTES / $ROW_BYTES) * 100" | bc)
  printf "%-35s %s%%\n" "I/O Volume Reduction:" "${IO_REDUCTION}"
  
  # Latency Speedup
  LATENCY_SPEEDUP=$(echo "scale=2; $ROW_TIME / $COL_TIME" | bc)
  printf "%-35s %sx\n" "Query Latency Speedup:" "${LATENCY_SPEEDUP}"
  
  # Throughput Ratio (higher is better for column-store in this case)
  # Note: Column-store shows lower MB/s because it reads less data in less time
  # The real win is efficiency: less data moved for same result
  printf "%-35s Row: %s MB/s, Col: %s MB/s\n" "Raw Throughput:" "${ROW_THROUGHPUT}" "${COL_THROUGHPUT}"
  
  # Efficiency: MB processed per ms (higher = more efficient I/O)
  ROW_EFF=$(echo "scale=2; $ROW_MB / $ROW_TIME" | bc)
  COL_EFF=$(echo "scale=2; $COL_MB / $COL_TIME" | bc)
  EFF_RATIO=$(echo "scale=2; $COL_EFF / $ROW_EFF" | bc)
  printf "%-35s Row: %s MB/ms, Col: %s MB/ms (%sx more efficient)\n" "I/O Efficiency:" "${ROW_EFF}" "${COL_EFF}" "${EFF_RATIO}"
  
  # Energy estimate (simplified: proportional to bytes × time)
  ROW_ENERGY=$(echo "scale=0; $ROW_BYTES * ${ROW_TIME%.*}" | bc)
  COL_ENERGY=$(echo "scale=0; $COL_BYTES * ${COL_TIME%.*}" | bc)
  ENERGY_SAVE=$(echo "scale=2; (1 - $COL_ENERGY / $ROW_ENERGY) * 100" | bc)
  printf "%-35s ~%s%% estimated I/O energy saved\n" "Energy Efficiency:" "${ENERGY_SAVE}"
  
  # Statistical confidence (are results consistent?)
  echo -e "\n${YELLOW}=== Interpretation Guide ===${NC}"
  echo "• I/O Reduction >80%: Column-store ideal for analytical queries"
  echo "• Speedup >3x: Significant real-world impact for large datasets"
  echo "• Bytes/Record: Row=${ROW_BPR}B vs Col=${COL_BPR}B explains the gap"
  echo "• Throughput note: Lower MB/s for column-store is GOOD—less data moved!"
  
  # Verdict
  echo -e "\n${GREEN}=== Verdict ===${NC}"
  if (( $(echo "$IO_REDUCTION > 80" | bc -l) )) && (( $(echo "$LATENCY_SPEEDUP > 3" | bc -l) )); then
    echo "Column-oriented storage is HIGHLY recommended for this workload."
    echo "   Workload: Single-column aggregation (OLAP pattern)"
    echo "   Expected benefit: 3-10x faster queries, 80-90% less I/O"
  elif (( $(echo "$IO_REDUCTION > 50" | bc -l) )); then
    echo "Column-store shows moderate benefit. Consider hybrid approaches."
  else
    echo "Row-store may be preferable for this specific query pattern."
  fi
else
  echo -e "${RED}Error: Could not parse benchmark output${NC}"
  exit 1
fi

echo -e "\n${BLUE}=== Raw Output Snippets ===${NC}"
echo "Row-Oriented excerpt:"
(cd RowsDB/build && ./rowsdb 2>&1 | grep -A5 "Row-Oriented")
echo -e "\nColumn-Oriented excerpt:"
(cd ColsDB/build && ./colsdb 2>&1 | grep -A5 "Column-Oriented")