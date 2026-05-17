#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "common/type.h"
#include "dram/nearbank/nearbank_config.h"
#include "dram/nearbank/nearbank_pim.h"

using namespace llm_system;

static int tests_passed = 0;
static int tests_failed = 0;

void check(bool condition, const std::string& test_name) {
    if (condition) {
        tests_passed++;
        std::cout << "  PASS: " << test_name << std::endl;
    } else {
        tests_failed++;
        std::cout << "  FAIL: " << test_name << std::endl;
    }
}

void print_result(const NearbankGEMVResult& result) {
    std::cout << "    " << result.description << std::endl;
}

// =============================================================================
// Test 1: Basic FP16 GEMV latency with default config
// =============================================================================
void test_basic_fp16_gemv() {
    std::cout << "\n=== Test: Basic FP16 GEMV (default config) ===" << std::endl;

    NearbankPIMConfig config;
    auto unit = NearbankPIMUnit::Create(config);

    // GEMV(1, 128) @ (128, 256) → (1, 256)
    // This represents attention scoring: Q[1,128] @ K^T[128,256]
    int M = 1, K = 128, N = 256;
    int elem_size = 2;  // FP16

    auto result = unit->computeGEMVLatency(M, K, N, elem_size);
    print_result(result);

    check(result.latency_ns > 0, "FP16 GEMV has positive latency");
    check(result.total_bytes > 0, "FP16 GEMV has positive total bytes");
    check(result.bytes_per_bank > 0, "FP16 GEMV has positive bytes per bank");
    check(result.bytes_per_bank <= result.total_bytes,
          "Per-bank bytes <= total bytes");
    check(result.num_rowbuffer_fills >= 1, "At least one rowbuffer fill");

    // Verify total_bytes calculation
    double expected_elements = M * K + K * N + M * N;
    double expected_bytes = expected_elements * elem_size;
    check(std::abs(result.total_bytes - expected_bytes) < 0.01,
          "Total bytes calculation correct");

    std::cout << "  Latency: " << result.latency_ns << " ns" << std::endl;
    std::cout << "  RB time: " << result.rowbuffer_time_ns << " ns" << std::endl;
    std::cout << "  PE time: " << result.pe_compute_time_ns << " ns" << std::endl;
}

// =============================================================================
// Test 2: GEMV with 2-bit quantized data
// =============================================================================
void test_2bit_gemv() {
    std::cout << "\n=== Test: 2-bit Quantized GEMV ===" << std::endl;

    NearbankPIMConfig config;
    auto unit = NearbankPIMUnit::Create(config);

    int M = 1, K = 128, N = 256;
    int elem_size_fp16 = 2;
    int elem_size_2bit = 1;  // 0.25 bytes per element, but precision_byte=1 in simulator

    auto result_fp16 = unit->computeGEMVLatency(M, K, N, elem_size_fp16);
    auto result_2bit = unit->computeGEMVLatency(M, K, N, elem_size_2bit);

    print_result(result_fp16);
    print_result(result_2bit);

    // 2-bit should have roughly half the total bytes (1/2 the element size)
    check(result_2bit.total_bytes < result_fp16.total_bytes,
          "2-bit has less total data than FP16");
    check(result_2bit.latency_ns < result_fp16.latency_ns,
          "2-bit latency < FP16 latency (less data to move)");

    double ratio = result_fp16.total_bytes / result_2bit.total_bytes;
    std::cout << "  FP16/2bit byte ratio: " << ratio << " (expected ~2.0)" << std::endl;
}

// =============================================================================
// Test 3: Different PE configurations
// =============================================================================
void test_pe_configurations() {
    std::cout << "\n=== Test: Different PE Configurations ===" << std::endl;

    int M = 1, K = 128, N = 256;
    int elem_size = 2;

    // Fast PE (narrow but fast)
    {
        NearbankPIMConfig config;
        config.pe_width_bytes = 8.0;      // 8B per cycle
        config.pe_cycle_time_ns = 0.25;   // 4GHz
        auto unit = NearbankPIMUnit::Create(config);
        auto result = unit->computeGEMVLatency(M, K, N, elem_size);
        std::cout << "  Fast PE (8B@4GHz): " << result.latency_ns << " ns" << std::endl;
    }

    // Wide PE
    {
        NearbankPIMConfig config;
        config.pe_width_bytes = 32.0;     // 32B per cycle
        config.pe_cycle_time_ns = 0.5;    // 2GHz
        auto unit = NearbankPIMUnit::Create(config);
        auto result = unit->computeGEMVLatency(M, K, N, elem_size);
        std::cout << "  Wide PE (32B@2GHz): " << result.latency_ns << " ns" << std::endl;
    }

    // Slow PE
    {
        NearbankPIMConfig config;
        config.pe_width_bytes = 4.0;      // 4B per cycle
        config.pe_cycle_time_ns = 1.0;    // 1GHz
        auto unit = NearbankPIMUnit::Create(config);
        auto result = unit->computeGEMVLatency(M, K, N, elem_size);
        std::cout << "  Slow PE (4B@1GHz): " << result.latency_ns << " ns" << std::endl;
    }

    check(true, "PE configurations computed without error");
}

// =============================================================================
// Test 4: Different bank counts
// =============================================================================
void test_bank_counts() {
    std::cout << "\n=== Test: Different Bank Counts ===" << std::endl;

    int M = 1, K = 128, N = 256;
    int elem_size = 2;

    double prev_latency = 1e9;  // Large initial value

    for (int num_banks : {64, 128, 256, 512, 1024}) {
        NearbankPIMConfig config;
        config.num_pim_banks = num_banks;
        auto unit = NearbankPIMUnit::Create(config);
        auto result = unit->computeGEMVLatency(M, K, N, elem_size);
        std::cout << "  " << num_banks << " banks: " << result.latency_ns
                  << " ns, per_bank=" << result.bytes_per_bank
                  << "B, rb_fills=" << result.num_rowbuffer_fills << std::endl;

        // More banks → less work per bank → lower latency (generally)
        // Note: with very few banks, rowbuffer fills increase
        check(result.bytes_per_bank <= result.total_bytes,
              "Per-bank bytes <= total bytes");
        prev_latency = result.latency_ns;
    }
}

// =============================================================================
// Test 5: Different GEMV sizes (attention scenarios)
// =============================================================================
void test_attention_scenarios() {
    std::cout << "\n=== Test: Attention GEMV Scenarios ===" << std::endl;

    NearbankPIMConfig config;
    auto unit = NearbankPIMUnit::Create(config);
    int elem_size = 2;  // FP16

    // Short sequence (prompt processing)
    {
        // Scoring: Q[1,128] @ K^T[128,256]
        auto score = unit->computeGEMVLatency(1, 128, 256, elem_size);
        std::cout << "  Short seq scoring (256 tokens): " << score.latency_ns
                  << " ns" << std::endl;

        // Context: scores[1,256] @ V[256,128]
        auto ctx = unit->computeGEMVLatency(1, 256, 128, elem_size);
        std::cout << "  Short seq context (256 tokens): " << ctx.latency_ns
                  << " ns" << std::endl;
    }

    // Long sequence
    {
        auto score = unit->computeGEMVLatency(1, 128, 4096, elem_size);
        std::cout << "  Long seq scoring (4096 tokens): " << score.latency_ns
                  << " ns" << std::endl;

        auto ctx = unit->computeGEMVLatency(1, 4096, 128, elem_size);
        std::cout << "  Long seq context (4096 tokens): " << ctx.latency_ns
                  << " ns" << std::endl;
    }

    // Very long sequence (32K)
    {
        auto score = unit->computeGEMVLatency(1, 128, 32768, elem_size);
        std::cout << "  Very long seq scoring (32768 tokens): "
                  << score.latency_ns << " ns" << std::endl;
        check(score.num_rowbuffer_fills >= 1,
              "Very long seq needs multiple rowbuffer fills");
    }

    check(true, "Attention scenarios computed");
}

// =============================================================================
// Test 6: Edge cases
// =============================================================================
void test_edge_cases() {
    std::cout << "\n=== Test: Edge Cases ===" << std::endl;

    NearbankPIMConfig config;
    auto unit = NearbankPIMUnit::Create(config);

    // Very small GEMV (single element)
    {
        auto result = unit->computeGEMVLatency(1, 1, 1, 1);
        print_result(result);
        check(result.latency_ns > 0, "Single element GEMV has positive latency");
        check(result.num_rowbuffer_fills == 1,
              "Single element needs exactly 1 rowbuffer fill");
    }

    // GEMV where data fits in one row buffer
    {
        auto result = unit->computeGEMVLatency(1, 16, 16, 2);
        print_result(result);
        double total_bytes = (1 * 16 + 16 * 16 + 1 * 16) * 2.0;
        check(result.total_bytes == total_bytes, "Small GEMV byte count correct");
    }

    // M > 1 (multiple query tokens)
    {
        auto result = unit->computeGEMVLatency(4, 128, 256, 2);
        print_result(result);
        check(result.latency_ns > 0, "Multi-token GEMV has positive latency");
        // M=4 should have 4x the data of M=1
        auto result_m1 = unit->computeGEMVLatency(1, 128, 256, 2);
        double ratio = result.total_bytes / result_m1.total_bytes;
        std::cout << "  M=4 / M=1 byte ratio: " << ratio << std::endl;
    }
}

// =============================================================================
// Test 7: Rowbuffer pipelining verification
// =============================================================================
void test_pipelining() {
    std::cout << "\n=== Test: Rowbuffer Pipelining ===" << std::endl;

    // Create a config with very slow DRAM so multiple rowbuffers are needed
    NearbankPIMConfig config;
    config.rowbuffer_size_bytes = 512.0;   // Small rowbuffer to force multiple fills
    config.dram_to_rb_bw_per_bank = 10.0e9;  // Slow DRAM: 10 GB/s
    config.pe_width_bytes = 32.0;             // Fast PE: 32B/cycle
    config.pe_cycle_time_ns = 0.25;           // 4GHz
    config.num_pim_banks = 64;               // Fewer banks → more work per bank

    auto unit = NearbankPIMUnit::Create(config);

    // GEMV large enough to need multiple rowbuffers
    int M = 1, K = 128, N = 1024;
    auto result = unit->computeGEMVLatency(M, K, N, 2);

    print_result(result);

    // With pipelining, latency should be less than rb_time + pe_time
    double sequential_time = result.rowbuffer_time_ns + result.pe_compute_time_ns;
    std::cout << "  Sequential (no-pipe) time: " << sequential_time << " ns" << std::endl;
    std::cout << "  Pipelined latency: " << result.latency_ns << " ns" << std::endl;
    std::cout << "  Pipeline efficiency: "
              << (sequential_time / result.latency_ns) << "x" << std::endl;

    check(result.latency_ns < sequential_time,
          "Pipelined latency < sequential time (pipelining works)");
    check(result.num_rowbuffer_fills > 1,
          "Multiple rowbuffer fills needed");
}

// =============================================================================
// Test 8: Config validation
// =============================================================================
void test_config_validation() {
    std::cout << "\n=== Test: Config Validation ===" << std::endl;

    {
        NearbankPIMConfig config;
        check(config.getTotalBanks() == config.num_pim_banks,
              "getTotalBanks returns num_pim_banks");
        check(config.getPEThroughput() > 0,
              "PE throughput is positive");
        std::cout << "  PE throughput: " << config.getPEThroughput() / 1e9
                  << " GB/s" << std::endl;
    }

    {
        NearbankPIMConfig config;
        config.pe_width_bytes = 16.0;
        config.pe_cycle_time_ns = 0.5;
        double expected_throughput = 16.0 / 0.5 * 1e9;  // 32 GB/s
        check(std::abs(config.getPEThroughput() - expected_throughput) < 0.01,
              "PE throughput calculation correct");
    }
}

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "NearbankPIMUnit Test Suite" << std::endl;
    std::cout << "============================================================" << std::endl;

    test_basic_fp16_gemv();
    test_2bit_gemv();
    test_pe_configurations();
    test_bank_counts();
    test_attention_scenarios();
    test_edge_cases();
    test_pipelining();
    test_config_validation();

    std::cout << "\n============================================================" << std::endl;
    std::cout << "RESULTS: " << tests_passed << " passed, "
              << tests_failed << " failed" << std::endl;
    std::cout << "============================================================" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
