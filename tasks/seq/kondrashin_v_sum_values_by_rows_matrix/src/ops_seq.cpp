#include <algorithm>
#include <vector>

#include "seq/kondrashin_v_sum_values_by_rows_matrix/include/ops_seq.hpp"

// using namespace std::chrono_literals;
namespace kondrashin_v_sum_values_by_rows_matrix_seq {
  bool SumValByRowsMatrix::PreProcessingImpl() {
    auto *tmp = reinterpret_cast<int *>(task_data->inputs[0]);
    input_.resize(task_data->inputs_count[0]);
    std::copy(tmp, tmp + task_data->inputs_count[0], input_.begin());
    rows_ = task_data->inputs_count[1];
    cols_ = task_data->inputs_count[2];
    sum_.resize(rows_, 0);
    return true;
  }
  
  bool SumValByRowsMatrix::ValidationImpl() {
    return (task_data->inputs_count[1] == task_data->outputs_count[0]);
  }
  
  bool SumValByRowsMatrix::RunImpl() {
    for (size_t i = 0; i < rows_; i++) {
      int row_sum = 0;
      for (size_t j = 0; j < cols_; j++) {
        row_sum += input_[i * cols_ + j];
      }
      sum_[i] = row_sum;
    }
    return true;
  }
  
  bool SumValByRowsMatrix::PostProcessingImpl() {
    auto *output = reinterpret_cast<int *>(task_data->outputs[0]);
    std::copy(sum_.begin(), sum_.end(), output);
    return true;
  }
} 
