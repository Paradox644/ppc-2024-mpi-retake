#include "mpi/kondrashin_v_sum_values_by_rows_matrix/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <vector>

namespace kondrashin_v_sum_values_by_rows_matrix_mpi {

bool SumValByRowsMatrix::PreProcessingImpl() {
  if (mpi_world_.rank() == 0) {
    input_.resize(task_data->inputs_count[0]);
    auto *data_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
    std::copy(data_ptr, data_ptr + task_data->inputs_count[0], input_.begin());
    rows_ = task_data->inputs_count[1];
    cols_ = task_data->inputs_count[2];
    sums_.resize(rows_, 0);
  }
  return true;
}

bool SumValByRowsMatrix::ValidationImpl() {
  if (mpi_world_.rank() == 0) {
    return task_data->inputs_count[1] == task_data->outputs_count[0];
  }
  return true;
}

bool SumValByRowsMatrix::RunImpl() {
  broadcast(mpi_world_, rows_, 0);
  broadcast(mpi_world_, cols_, 0);

  int rows_per_process = rows_ / mpi_world_.size();
  int remaining_rows = rows_ % mpi_world_.size();
  int local_rows = (mpi_world_.rank() == mpi_world_.size() - 1) ? rows_per_process + remaining_rows : rows_per_process;

  local_input_.resize(local_rows * cols_);
  std::vector<int> send_counts(mpi_world_.size());
  std::vector<int> recv_counts(mpi_world_.size());
  for (int i = 0; i < mpi_world_.size(); ++i) {
    send_counts[i] = (i == mpi_world_.size() - 1) ? rows_per_process + remaining_rows : rows_per_process;
    send_counts[i] *= cols_;
    recv_counts[i] = (i == mpi_world_.size() - 1) ? rows_per_process + remaining_rows : rows_per_process;
  }

  boost::mpi::scatterv(mpi_world_, input_.data(), send_counts, local_input_.data(), 0);

  std::vector<int> local_sums(local_rows, 0);
  for (int i = 0; i < local_rows; ++i) {
    for (unsigned int j = 0; j < cols_; ++j) {
      local_sums[i] += local_input_[i * cols_ + j];
    }
  }

  boost::mpi::gatherv(mpi_world_, local_sums.data(), local_sums.size(), sums_.data(), recv_counts, 0);

  return true;
}

bool SumValByRowsMatrix::PostProcessingImpl() {
  if (mpi_world_.rank() == 0) {
    auto *output_ptr = reinterpret_cast<int *>(task_data->outputs[0]);
    std::copy(sums_.begin(), sums_.end(), output_ptr);
  }
  return true;
}

}  // namespace kondrashin_v_sum_values_by_rows_matrix_mpi