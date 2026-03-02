#include "inference/RKNNEngine.hpp"
#include <stdexcept>
#include <iostream>
#include <cstring>

// Include real RKNN headers; guard for non-ARM build environments
#if __has_include(<rknn_api.h>)
  #include <rknn_api.h>
  #define HAVE_RKNN 1
#else
  #define HAVE_RKNN 0
  // Minimal stubs so the TU compiles on x86 CI
  typedef uint64_t rknn_context;
  struct rknn_input  { uint32_t index; void* buf; uint32_t size; int pass_through; int type; int fmt; };
  struct rknn_output { uint8_t want_float; uint8_t is_prealloc; uint32_t index; void* buf; uint32_t size; };
  inline int rknn_init(rknn_context* c,void*,uint32_t,uint32_t,void*){(void)c;return -1;}
  inline int rknn_run(rknn_context,void*){return -1;}
  inline int rknn_inputs_set(rknn_context,uint32_t,rknn_input*){return -1;}
  inline int rknn_outputs_get(rknn_context,uint32_t,rknn_output*,void*){return -1;}
  inline int rknn_outputs_release(rknn_context,uint32_t,rknn_output*){return -1;}
  inline int rknn_destroy(rknn_context){return 0;}
  inline int rknn_query(rknn_context,int,void*,uint32_t){return -1;}
  #define RKNN_CORE_AUTO 0
  #define RKNN_CORE_MASK_ALL 7
  #define RKNN_QUERY_IN_OUT_NUM 3
  struct rknn_input_output_num { uint32_t n_input; uint32_t n_output; };
  #define RKNN_TENSOR_UINT8 1
  #define RKNN_TENSOR_NHWC  1
#endif

namespace el {

RKNNEngine::RKNNEngine(const std::string& modelPath, int numCores)
  : modelPath_(modelPath), numCores_(numCores) {}

RKNNEngine::~RKNNEngine() {
  release();
}

int RKNNEngine::coreMask() const {
#if HAVE_RKNN
  switch (numCores_) {
    case 1: return RKNN_NPU_CORE_0;
    case 2: return RKNN_NPU_CORE_0_1;
    default: return RKNN_NPU_CORE_ALL;
  }
#else
  return 0;
#endif
}

bool RKNNEngine::load() {
#if !HAVE_RKNN
  std::cerr << "[RKNNEngine] RKNN runtime not available (non-ARM build). Using mock mode.\n";
  loaded_ = true;
  numOutputs_ = 2;
  return true;
#else
  // Read model file
  FILE* fp = fopen(modelPath_.c_str(), "rb");
  if (!fp) {
    std::cerr << "[RKNNEngine] Cannot open model: " << modelPath_ << "\n";
    return false;
  }
  fseek(fp, 0, SEEK_END);
  long size = ftell(fp);
  rewind(fp);
  std::vector<uint8_t> model(size);
  fread(model.data(), 1, size, fp);
  fclose(fp);

  int ret = rknn_init(&ctx_, model.data(), (uint32_t)size, 0, nullptr);
  if (ret != 0) {
    std::cerr << "[RKNNEngine] rknn_init failed: " << ret << "\n";
    return false;
  }

  // Set core mask
  rknn_core_mask mask = (rknn_core_mask)coreMask();
  rknn_set_core_mask(ctx_, mask);

  // Query output count
  rknn_input_output_num io_num{};
  ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret == 0) numOutputs_ = io_num.n_output;

  loaded_ = true;
  std::cout << "[RKNNEngine] Model loaded: " << modelPath_
            << "  outputs=" << numOutputs_ << "\n";
  return true;
#endif
}

void RKNNEngine::release() {
#if HAVE_RKNN
  if (ctx_) {
    rknn_destroy(ctx_);
    ctx_ = 0;
  }
#endif
  loaded_    = false;
  numOutputs_ = 0;
}

InferResult RKNNEngine::infer(const uint8_t* data, int width, int height) {
  InferResult res{};
  res.ok = false;
  if (!loaded_) return res;

#if !HAVE_RKNN
  // Mock: return zeroed output
  res.outputData.assign(8400 * 85, 0.0f);
  res.width  = width;
  res.height = height;
  res.ok     = true;
  return res;
#else
  rknn_input inputs[1]{};
  inputs[0].index      = 0;
  inputs[0].type       = RKNN_TENSOR_UINT8;
  inputs[0].fmt        = RKNN_TENSOR_NHWC;
  inputs[0].buf        = const_cast<uint8_t*>(data);
  inputs[0].size       = (uint32_t)(width * height * 3);
  inputs[0].pass_through = 0;

  int ret = rknn_inputs_set(ctx_, 1, inputs);
  if (ret != 0) { std::cerr << "[RKNNEngine] inputs_set failed: " << ret << "\n"; return res; }

  ret = rknn_run(ctx_, nullptr);
  if (ret != 0) { std::cerr << "[RKNNEngine] run failed: " << ret << "\n"; return res; }

  std::vector<rknn_output> outputs(numOutputs_);
  for (auto& o : outputs) { o.want_float = 1; o.is_prealloc = 0; }
  ret = rknn_outputs_get(ctx_, numOutputs_, outputs.data(), nullptr);
  if (ret != 0) { std::cerr << "[RKNNEngine] outputs_get failed: " << ret << "\n"; return res; }

  for (auto& o : outputs) {
    float* p = reinterpret_cast<float*>(o.buf);
    uint32_t n = o.size / sizeof(float);
    res.outputData.insert(res.outputData.end(), p, p + n);
  }
  rknn_outputs_release(ctx_, numOutputs_, outputs.data());

  res.width  = width;
  res.height = height;
  res.ok     = true;
  return res;
#endif
}

bool RKNNEngine::swapModel(const std::string& newModelPath) {
  release();
  modelPath_ = newModelPath;
  return load();
}

} // namespace el
