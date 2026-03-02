#pragma once
#include <vector>
#include <string>
#include "inference/RKNNEngine.hpp"

namespace el {

struct Detection {
  int   classId;
  float confidence;
  float cx, cy, w, h;   // normalized [0,1] in input frame coords
  std::string className;
};

class FramePipeline {
public:
  FramePipeline(
    RKNNEngine&                    engine,
    const std::vector<std::string>& classNames,
    float                          confThreshold = 0.5f,
    float                          iouThreshold  = 0.45f,
    int                            inputSize     = 640
  );

  // Process a BGR frame (from OpenCV), returns detections after NMS
  std::vector<Detection> process(const uint8_t* bgrData, int frameW, int frameH);

private:
  RKNNEngine&                     engine_;
  std::vector<std::string>        classNames_;
  float                           confThreshold_;
  float                           iouThreshold_;
  int                             inputSize_;

  // Letterbox resize helper
  std::vector<uint8_t> letterbox(const uint8_t* src, int srcW, int srcH,
                                  float& scale, int& padW, int& padH);

  // Decode YOLOv8 raw output [1 x 84 x 8400] from output tensor
  std::vector<Detection> decode(const InferResult& res, float scale,
                                 int padW, int padH, int origW, int origH);

  // Non-maximum suppression
  std::vector<Detection> nms(std::vector<Detection>& dets);
};

} // namespace el
