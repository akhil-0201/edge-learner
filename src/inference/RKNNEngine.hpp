#pragma once
#include <string>
#include <vector>
#include <cstdint>

// Forward-declare RKNNLite handle to avoid pulling in rknn headers in every TU
typedef void* rknn_context;

namespace el {

struct InferResult {
  std::vector<float> outputData;  // raw flattened output tensor(s)
  int                width;
  int                height;
  bool               ok;
};

class RKNNEngine {
public:
  explicit RKNNEngine(const std::string& modelPath, int numCores = 3);
  ~RKNNEngine();

  // Non-copyable, moveable
  RKNNEngine(const RKNNEngine&)            = delete;
  RKNNEngine& operator=(const RKNNEngine&) = delete;
  RKNNEngine(RKNNEngine&&)                 = default;
  RKNNEngine& operator=(RKNNEngine&&)      = default;

  bool load();
  void release();
  bool isLoaded() const { return loaded_; }

  // Run inference on a pre-resized BGR frame (H x W x 3, uint8)
  InferResult infer(const uint8_t* data, int width, int height);

  // Hot-swap: release current model and load a new one atomically
  bool swapModel(const std::string& newModelPath);

private:
  std::string  modelPath_;
  int          numCores_;
  rknn_context ctx_  = nullptr;
  bool         loaded_ = false;

  // Number of output tensors (detected at load time)
  uint32_t     numOutputs_ = 0;

  int coreMask() const;
};

} // namespace el
