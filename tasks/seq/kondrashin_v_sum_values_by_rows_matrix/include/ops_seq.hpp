#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kondrashin_v_sum_values_by_rows_matrix_seq {

class SumValByRowsMatrix : public ppc::core::Task {
 public:
  explicit SumValByRowsMatrix(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, sum_;
  unsigned int rows_, cols_;
};

}  // namespace kondrashin_v_sum_values_by_rows_matrix