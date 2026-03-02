#include "FramePipeline.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>

FramePipeline::FramePipeline(
    RKNNEngine& engine,
    const std::vector<std::string>& classNames,
    float confThreshold,
    float iouThreshold,
    int inputSize)
    : engine_(engine)
    , classNames_(classNames)
    , confThreshold_(confThreshold)
    , iouThreshold_(iouThreshold)
    , inputSize_(inputSize)
{}

std::vector<uint8_t> FramePipeline::letterbox(
    const uint8_t* src, int srcW, int srcH,
    float& scale, int& padW, int& padH)
{
    float scaleX = static_cast<float>(inputSize_) / srcW;
    float scaleY = static_cast<float>(inputSize_) / srcH;
    scale = std::min(scaleX, scaleY);

    int newW = static_cast<int>(srcW * scale);
    int newH = static_cast<int>(srcH * scale);
    padW = (inputSize_ - newW) / 2;
    padH = (inputSize_ - newH) / 2;

    std::vector<uint8_t> dst(inputSize_ * inputSize_ * 3, 114);

    // Simple nearest-neighbour resize + paste
    for (int y = 0; y < newH; ++y) {
        for (int x = 0; x < newW; ++x) {
            int srcX = static_cast<int>(x / scale);
            int srcY = static_cast<int>(y / scale);
            srcX = std::min(srcX, srcW - 1);
            srcY = std::min(srcY, srcH - 1);
            const uint8_t* s = src + (srcY * srcW + srcX) * 3;
            uint8_t* d = dst.data() + ((y + padH) * inputSize_ + (x + padW)) * 3;
            d[0] = s[0]; d[1] = s[1]; d[2] = s[2];
        }
    }
    return dst;
}

std::vector<Detection> FramePipeline::decode(
    const InferResult& res, float scale,
    int padW, int padH, int origW, int origH)
{
    // Output tensor: float32 [1 x 84 x 8400]
    const float* data = reinterpret_cast<const float*>(res.data.data());
    const int numAnchors = 8400;
    const int numAttribs = 84; // 4 box + 80 classes

    std::vector<Detection> dets;
    for (int i = 0; i < numAnchors; ++i) {
        // Find best class
        float maxConf = 0.0f;
        int bestCls = -1;
        for (int c = 4; c < numAttribs; ++c) {
            float v = data[c * numAnchors + i];
            if (v > maxConf) { maxConf = v; bestCls = c - 4; }
        }
        if (maxConf < confThreshold_) continue;

        float cx = data[0 * numAnchors + i];
        float cy = data[1 * numAnchors + i];
        float w  = data[2 * numAnchors + i];
        float h  = data[3 * numAnchors + i];

        // Convert from letterboxed coords to original frame coords
        cx = (cx - padW) / scale;
        cy = (cy - padH) / scale;
        w  /= scale;
        h  /= scale;

        Detection d;
        d.classId   = bestCls;
        d.confidence = maxConf;
        d.cx = cx / origW;
        d.cy = cy / origH;
        d.w  = w  / origW;
        d.h  = h  / origH;
        d.className = (bestCls < static_cast<int>(classNames_.size()))
                      ? classNames_[bestCls] : "unknown";
        dets.push_back(d);
    }
    return dets;
}

std::vector<Detection> FramePipeline::nms(
    std::vector<Detection>& dets)
{
    std::sort(dets.begin(), dets.end(),
        [](const Detection& a, const Detection& b){
            return a.confidence > b.confidence;
        });

    std::vector<bool> suppressed(dets.size(), false);
    std::vector<Detection> result;

    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(dets[i]);
        const auto& a = dets[i];
        float ax1 = a.cx - a.w/2, ay1 = a.cy - a.h/2;
        float ax2 = a.cx + a.w/2, ay2 = a.cy + a.h/2;
        for (size_t j = i+1; j < dets.size(); ++j) {
            if (suppressed[j]) continue;
            if (dets[j].classId != a.classId) continue;
            const auto& b = dets[j];
            float bx1 = b.cx - b.w/2, by1 = b.cy - b.h/2;
            float bx2 = b.cx + b.w/2, by2 = b.cy + b.h/2;
            float ix1 = std::max(ax1, bx1), iy1 = std::max(ay1, by1);
            float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
            float inter = std::max(0.0f, ix2-ix1) * std::max(0.0f, iy2-iy1);
            float unionA = a.w*a.h + b.w*b.h - inter;
            if (inter / unionA > iouThreshold_) suppressed[j] = true;
        }
    }
    return result;
}

std::vector<Detection> FramePipeline::process(
    const uint8_t* bgrData, int frameW, int frameH)
{
    float scale; int padW, padH;
    auto lb = letterbox(bgrData, frameW, frameH, scale, padW, padH);

    InferResult res = engine_.infer(lb.data(), lb.size());

    auto dets = decode(res, scale, padW, padH, frameW, frameH);
    return nms(dets);
}
