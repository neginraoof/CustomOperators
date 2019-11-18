#include <iostream>
#include <gtest/gtest.h>
#include "custom_op.h"
#include "onnxruntime_cxx_api.h"

#define TSTR(X) (X)
typedef const char* PATH_TYPE;

// LOGGING TEST
template <bool use_customer_logger>
class CApiTestImpl : public ::testing::Test {
 protected:
  Ort::Env env_{nullptr};
  void SetUp() override {
    env_ = Ort::Env(ORT_LOGGING_LEVEL_INFO, "Default");

  }
};

typedef CApiTestImpl<false> CApiTest;
// END LOGGING

template <typename T>
void TestInference(Ort::Env& env, T model_uri,
                   const std::vector<Input>& inputs,
                   const char* output_name,
                   const std::vector<int64_t>& expected_dims_y,
                   const std::vector<float>& expected_values_y,
                   int provider_type, OrtCustomOpDomain* custom_op_domain_ptr) {
  Ort::SessionOptions session_options;
  std::cout << "Running simple inference with default provider" << std::endl;

  if (custom_op_domain_ptr) {
    session_options.Add(custom_op_domain_ptr);
  }

  Ort::Session session(env, model_uri, session_options);

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::vector<Ort::Value> input_tensors;
  std::vector<const char*> input_names;
  Ort::Value output_tensor{nullptr};
  output_tensor = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(expected_values_y.data()), expected_values_y.size(), expected_dims_y.data(), expected_dims_y.size());

  for (size_t i = 0; i < inputs.size(); i++) {
    input_names.emplace_back(inputs[i].name);
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(inputs[i].values.data()), inputs[i].values.size(), inputs[i].dims.data(), inputs[i].dims.size()));
	}

  std::vector<Ort::Value> ort_outputs;
  ort_outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), &output_name, 1);

}

static constexpr PATH_TYPE MODEL_URI = TSTR("../../pytorch_custom_op/model.onnx");

class CApiTestWithProvider : public CApiTest,
                             public ::testing::WithParamInterface<int> {
};

// Tests that the Foo::Bar() method does Abc.
TEST_P(CApiTestWithProvider, simple) {
  // simple inference test
  // prepare inputs

  std::vector<Input> inputs(4);
  auto input = inputs.begin();
  input->name = "X";
  input->dims = {3, 2, 1, 2};
  input->values = { 1.5410f, -0.2934f, -2.1788f,  0.5684f, -1.0845f, -1.3986f , 0.4033f,  0.8380f, -0.7193f, -0.4033f ,-0.5966f,  0.1820f};

	input = std::next(input, 1);
  input->name = "num_groups";
  input->dims = {1};
  input->values = {2.f};

	input = std::next(input, 1);
  input->name = "scale";
  input->dims = {2};
  input->values = {2.0f, 1.0f};

	input = std::next(input, 1);
  input->name = "bias";
  input->dims = {2};
  input->values = {1.f, 0.f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2, 1, 2};
  std::vector<float> expected_values_y = { 3.0000f, -1.0000f, -1.0000f,  1.0000f, 2.9996f, -0.9996f, -0.9999f,  0.9999f,  -0.9996f,  2.9996f, -1.0000f,  1.0000f};

  GroupNormCustomOp custom_op;
  Ort::CustomOpDomain custom_op_domain("org.pytorch.mydomain");
  custom_op_domain.Add(&custom_op);

  TestInference<PATH_TYPE>(env_, MODEL_URI, inputs, "Y", expected_dims_y, expected_values_y, GetParam(), custom_op_domain);
}

INSTANTIATE_TEST_CASE_P(CApiTestWithProviders,
                        CApiTestWithProvider,
                        ::testing::Values(0, 1, 2, 3, 4));


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;

}