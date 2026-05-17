#include "dram/nearbank/nearbank_pim.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llm_system {

// =============================================================================
// computeGEMVLatency
// =============================================================================
//
// Latency Model (per bank, pipelined):
//
//   Given a GEMV operation M×K @ K×N → M×N:
//
//   1. Total data movement per GEMV:
//      - Read matrix A: M*K elements
//      - Read matrix B: K*N elements
//      - Write result:   M*N elements
//      - Total elements = M*K + K*N + M*N
//      - Total bytes = total_elements * element_size_bytes
//
//   2. Work distribution across banks:
//      - bytes_per_bank = total_bytes / num_pim_banks
//
//   3. Row buffer processing (per bank):
//      - Each row buffer fill reads rowbuffer_size_bytes from DRAM
//      - num_rb_fills = ceil(bytes_per_bank / rowbuffer_size_bytes)
//      - Time per fill: T_fill = rowbuffer_size_bytes / dram_to_rb_bw_per_bank
//      - Total fill time: T_fill_total = num_rb_fills * T_fill
//
//   4. PE computation (per bank):
//      - PE processes pe_width_bytes per cycle
//      - PE cycle time = pe_cycle_time_ns
//      - PE throughput = pe_width_bytes / pe_cycle_time_ns (bytes/ns)
//      - Total PE time: T_pe = bytes_per_bank / pe_throughput
//        = bytes_per_bank * pe_cycle_time_ns / pe_width_bytes
//
//   5. Pipelined latency:
//      - Fill row buffer 0 (T_fill), then PE starts computing on it
//      - While PE computes on rb_i, rb_{i+1} is being filled
//      - latency = T_fill + max((num_rb_fills - 1) * T_fill,
//                                num_rb_fills * T_pe_per_rb)
//      where T_pe_per_rb = rowbuffer_size_bytes * pe_cycle_time_ns
//                           / pe_width_bytes
//
//   6. Special case: if bytes_per_bank <= rowbuffer_size_bytes:
//      - Only one row buffer needed
//      - latency = max(bytes_per_bank / dram_to_rb_bw,
//                       bytes_per_bank * pe_cycle_time_ns / pe_width_bytes)
//      (fill and compute can't fully pipeline with one buffer, but the
//       overlap still exists as data streams through)
//
// For simplicity and consistency, we use the general formula even for the
// single-buffer case, as it provides a reasonable approximation.
// =============================================================================
NearbankGEMVResult NearbankPIMUnit::computeGEMVLatency(
    int M, int K, int N, int element_size_bytes) const {

    NearbankGEMVResult result;

    // --- Step 1: Calculate total data movement ---
    // Elements: read A (M*K) + read B (K*N) + write result (M*N)
    double total_elements = static_cast<double>(M) * K +
                            static_cast<double>(K) * N +
                            static_cast<double>(M) * N;
    result.total_bytes = total_elements * element_size_bytes;

    // --- Step 2: Distribute across banks ---
    int num_banks = config_.getTotalBanks();
    result.bytes_per_bank = result.total_bytes / num_banks;

    // --- Step 3: Row buffer processing ---
    double rb_size = config_.rowbuffer_size_bytes;
    double dram_bw = config_.dram_to_rb_bw_per_bank;

    // Number of row buffer fills needed (ceiling)
    result.num_rowbuffer_fills = static_cast<int>(
        std::ceil(result.bytes_per_bank / rb_size));
    if (result.num_rowbuffer_fills < 1) {
        result.num_rowbuffer_fills = 1;
    }

    // Time to fill one row buffer
    double time_per_rb_fill = rb_size / dram_bw * 1e9;  // Convert to ns

    // Total row buffer fill time (without pipelining)
    result.rowbuffer_time_ns = result.num_rowbuffer_fills * time_per_rb_fill;

    // --- Step 4: PE computation ---
    double pe_width = config_.pe_width_bytes;
    double pe_cycle = config_.pe_cycle_time_ns;

    // Time per row buffer worth of PE computation
    double time_per_rb_pe = rb_size / pe_width * pe_cycle;

    // Total PE time (without pipelining)
    result.pe_compute_time_ns = result.bytes_per_bank / pe_width * pe_cycle;

    // --- Step 5: Pipelined latency ---
    if (result.bytes_per_bank <= rb_size) {
        // Single row buffer case: no pipeline overlap between fills,
        // but compute and fill can partially overlap since PE can start
        // as soon as first data arrives
        // Simplified: max of fill time and compute time
        double fill_time = result.bytes_per_bank / dram_bw * 1e9;
        double compute_time = result.bytes_per_bank / pe_width * pe_cycle;
        result.latency_ns = std::max(fill_time, compute_time);
    } else {
        // Multi-rowbuffer pipelined case:
        // Fill first row buffer, then max(remaining fills, total PE compute)
        double first_fill = time_per_rb_fill;
        double remaining_fill =
            (result.num_rowbuffer_fills - 1) * time_per_rb_fill;
        double total_pe = result.pe_compute_time_ns;

        result.latency_ns = first_fill + std::max(remaining_fill, total_pe);
    }

    // --- Step 6: Build description ---
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "GEMV(" << M << "x" << K << " @ " << K << "x" << N
        << ", elem=" << element_size_bytes << "B)"
        << " total=" << result.total_bytes << "B"
        << " per_bank=" << result.bytes_per_bank << "B"
        << " rb_fills=" << result.num_rowbuffer_fills
        << " latency=" << result.latency_ns << "ns"
        << " (rb=" << result.rowbuffer_time_ns
        << "ns, pe=" << result.pe_compute_time_ns << "ns)";
    result.description = oss.str();

    return result;
}

}  // namespace llm_system
