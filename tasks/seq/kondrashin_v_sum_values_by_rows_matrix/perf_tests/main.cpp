#include <gtest/gtest.h>
#include <chrono>
#include <memory>
#include <vector>
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/kondrashin_v_sum_values_by_rows_matrix/include/ops_seq.hpp"

TEST(kondrashin_v_sum_values_by_rows_matrix_seq, test_run_pipeline) {
  const int rows = 12000, cols = 12000;
  std::vector<int> in(rows * cols, 1), out(rows, 0), expect(rows, cols);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->inputs_count.emplace_back(rows);
  task_data->inputs_count.emplace_back(cols);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto task = std::make_shared<kondrashin_v_sum_values_by_rows_matrix_seq::SumValByRowsMatrix>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(expect, out);
}

TEST(kondrashin_v_sum_values_by_rows_matrix_seq, test_task_run) {
  const int rows = 12000, cols = 12000;
  std::vector<int> in(rows * cols, 1), out(rows, 0), expect(rows, cols);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->inputs_count.emplace_back(rows);
  task_data->inputs_count.emplace_back(cols);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto task = std::make_shared<kondrashin_v_sum_values_by_rows_matrix_seq::SumValByRowsMatrix>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(expect, out);
}