#include <gtest/gtest.h>
#include <boost/mpi/communicator.hpp>
#include <vector>
#include <random>
#include <memory>
#include "core/task/include/task.hpp"
#include "mpi/kondrashin_v_sum_values_by_rows_matrix/include/ops_mpi.hpp"

namespace kondrashin_v_sum_values_by_rows_matrix_mpi {

std::vector<int> GenerateRandomMatrix(int size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 99);
  std::vector<int> matrix(size);
  for (int &elem : matrix) {
    elem = dis(gen);
  }
  return matrix;
}

TEST(RowSumCalculatorTest, EmptyMatrixTest) {
  boost::mpi::communicator world;
  int rows = 0, cols = 0;
  std::vector<int> in, out, expect;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->inputs_count.emplace_back(rows);
    task_data->inputs_count.emplace_back(cols);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  RowSumCalculator task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();
}

TEST(RowSumCalculatorTest, SingleElementMatrixTest) {
  boost::mpi::communicator world;
  int rows = 1, cols = 1;
  std::vector<int> in = {5}, out(rows, 0), expect = {5};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->inputs_count.emplace_back(rows);
    task_data->inputs_count.emplace_back(cols);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  RowSumCalculator task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(out, expect);
  }
}

TEST(RowSumCalculatorTest, LargeMatrixTest) {
  boost::mpi::communicator world;
  int rows = 100, cols = 100;
  auto in = GenerateRandomMatrix(rows * cols);
  std::vector<int> out(rows, 0), expect(rows, 0);

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      expect[i] += in[i * cols + j];
    }
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->inputs_count.emplace_back(rows);
    task_data->inputs_count.emplace_back(cols);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  RowSumCalculator task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(out, expect);
  }
}

}  // namespace kondrashin_v_sum_values_by_rows_matrix_mpi