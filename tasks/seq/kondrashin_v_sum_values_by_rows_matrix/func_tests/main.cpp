#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <memory>
#include "core/task/include/task.hpp"
#include "seq/kondrashin_v_sum_values_by_rows_matrix/include/ops_seq.hpp"
#include "core/task/include/task.hpp"
namespace kondrashin_v_sum_values_by_rows_matrix_seq {
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
      
      TEST(kondrashin_v_sum_values_by_rows_matrix_seq, test_validation) {
        const int rows = 1, cols = 1;
        std::vector<int> in(rows * cols, 0), out(rows, 0);
        auto task_data = std::make_shared<ppc::core::TaskData>();
        task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
        task_data->inputs_count.emplace_back(in.size());
        task_data->inputs_count.emplace_back(rows);
        task_data->inputs_count.emplace_back(cols);
        task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
        task_data->outputs_count.emplace_back(out.size());
      
        SumValByRowsMatrix task(task_data);
        ASSERT_TRUE(task.ValidationImpl());
      }
      
      TEST(kondrashin_v_sum_values_by_rows_matrix_seq, test_empty_martix) {
        const int rows = 0, cols = 0;
        std::vector<int> in, out, expect;
        auto task_data = std::make_shared<ppc::core::TaskData>();
        task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
        task_data->inputs_count.emplace_back(in.size());
        task_data->inputs_count.emplace_back(rows);
        task_data->inputs_count.emplace_back(cols);
        task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
        task_data->outputs_count.emplace_back(out.size());
      
        SumValByRowsMatrix task(task_data);
        ASSERT_TRUE(task.ValidationImpl());
        task.PreProcessingImpl();
        task.RunImpl();
        task.PostProcessingImpl();
        ASSERT_EQ(expect, out);
      }
      
      TEST(kondrashin_v_sum_values_by_rows_matrix_seq, test_one_element_matrix) {
        const int rows = 1, cols = 1;
        std::vector<int> in = {5}, out(rows, 0), expect = {5};
        auto task_data = std::make_shared<ppc::core::TaskData>();
        task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
        task_data->inputs_count.emplace_back(in.size());
        task_data->inputs_count.emplace_back(rows);
        task_data->inputs_count.emplace_back(cols);
        task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
        task_data->outputs_count.emplace_back(out.size());
      
        SumValByRowsMatrix task(task_data);
        ASSERT_TRUE(task.ValidationImpl());
        task.PreProcessingImpl();
        task.RunImpl();
        task.PostProcessingImpl();
        ASSERT_EQ(expect, out);
      }
      
      TEST(kondrashin_v_sum_values_by_rows_matrix_seq, test_big_matrix) {
        const int rows = 100, cols = 100;
        auto in = GenerateRandomMatrix(rows * cols);
        std::vector<int> out(rows, 0), expect(rows, 0);
        for (int i = 0; i < rows; ++i) {
          for (int j = 0; j < cols; ++j) {
            expect[i] += in[i * cols + j];
          }
        }
      
        auto task_data = std::make_shared<ppc::core::TaskData>();
        task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
        task_data->inputs_count.emplace_back(in.size());
        task_data->inputs_count.emplace_back(rows);
        task_data->inputs_count.emplace_back(cols);
        task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
        task_data->outputs_count.emplace_back(out.size());
      
        SumValByRowsMatrix task(task_data);
        ASSERT_TRUE(task.ValidationImpl());
        task.PreProcessingImpl();
        task.RunImpl();
        task.PostProcessingImpl();
        ASSERT_EQ(expect, out);
      }
} //namespace kondrashin_v_sum_values_by_rows_matrix_seq