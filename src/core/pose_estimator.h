#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include "core/types.h"

/**
 * AlphaPose SPPE FastPose ONNX inference wrapper.
 *
 * Reproduces the reference runtime flow for Part 4:
 *   - expand detection boxes,
 *   - affine crop to (224x160), BGR/[0,1], channel mean shift,
 *   - run ONNX -> heatmaps,
 *   - keep channels [0] + [5:], argmax decode,
 *   - map keypoints back to original-frame coordinates,
 *   - append neck node (midpoint of shoulders).
 */
class PoseEstimator
{
 public:
    PoseEstimator();
    ~PoseEstimator();

    PoseEstimator(PoseEstimator const&)            = delete;
    PoseEstimator& operator=(PoseEstimator const&) = delete;

    bool load(std::string const& onnxPath);
    bool isLoaded() const { return session_ != nullptr; }

    /** Returns one pose per detection (empty pose when decode is unavailable). */
    Poses estimate(cv::Mat const& bgr, Detections const& detections);

    void setInputShape(std::int32_t inputHeight, std::int32_t inputWidth)
    {
        input_height_ = inputHeight;
        input_width_  = inputWidth;
    }

 private:
    struct CropRegion
    {
        float x1{0.0F};
        float y1{0.0F};
        float x2{0.0F};
        float y2{0.0F};
    };

    bool buildInput(cv::Mat const& bgr,
                    Detections const& detections,
                    std::vector<float>& input,
                    std::vector<CropRegion>& cropRegions,
                    std::vector<std::size_t>& detectionIndices) const;

    Pose decodeOne(float const* heatmaps,
                   CropRegion const& cropRegion,
                   std::int64_t heatmapChannels,
                   std::int64_t heatmapHeight,
                   std::int64_t heatmapWidth) const;

    PoseKeypoint transformBoxInvert(float x,
                                    float y,
                                    float score,
                                    CropRegion const& cropRegion,
                                    std::int64_t heatmapHeight,
                                    std::int64_t heatmapWidth) const;

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;

    std::string input_name_;
    std::string output_name_;

    std::int32_t input_height_{224};
    std::int32_t input_width_{160};
    std::vector<std::int32_t> keep_channels_{0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
};
