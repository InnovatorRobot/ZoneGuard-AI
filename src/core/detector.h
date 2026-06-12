#ifndef ZoneGuardAI_CORE_DETECTOR_H_
#define ZoneGuardAI_CORE_DETECTOR_H_

#include <cstdint>
#include <memory>
#include <string>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include "core/types.h"

namespace ZoneGuardAI
{
namespace Core
{
/**
 * Tiny-YOLOv3 one-class person detector running on ONNX Runtime.
 *
 * Ports the pre/post-processing of DetectorLoader.TinyYOLOv3_onecls and
 * Detection/Utils.py from the reference project:
 *   - ResizePadding the BGR frame to a square `inputSize`,
 *   - BGR->RGB, scale to [0,1], NCHW float input,
 *   - run the ONNX graph -> raw [cx,cy,w,h,obj,cls] candidates,
 *   - confidence filter + Non-Maximum-Suppression (with weighted box merge),
 *   - rescale boxes back to the original frame and expand slightly.
 *
 * The Detector is intentionally Qt-free so the vision core stays reusable.
 */
class Detector
{
 public:
    Detector();
    ~Detector();

    Detector(Detector const&)            = delete;
    Detector& operator=(Detector const&) = delete;

    /** Load an ONNX model. Returns false on failure. */
    bool load(std::string const& onnxPath);

    bool isLoaded() const { return session_ != nullptr; }

    /** Detect persons in a BGR frame; boxes are in original-frame coords. */
    Detections detect(cv::Mat const& bgr);

    // --- Tunable parameters (defaults match the reference / manifest) ---------
    void setInputSize(std::int32_t inputSize) { input_size_ = inputSize; }
    void setConfThreshold(float threshold) { conf_threshold_ = threshold; }
    void setNmsThreshold(float threshold) { nms_threshold_ = threshold; }
    void setExpand(std::int32_t expandPixels) { expand_pixels_ = expandPixels; }

    std::int32_t inputSize() const { return input_size_; }

 private:
    cv::Mat resizePadding(cv::Mat const& bgr, float& scale) const;

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;

    std::string input_name_;
    std::string output_name_;

    std::int32_t input_size_{384};
    float conf_threshold_{0.45F};
    float nms_threshold_{0.20F};
    std::int32_t expand_pixels_{10};
};

}  // namespace Core
}  // namespace ZoneGuardAI
#endif  // ZoneGuardAI_CORE_DETECTOR_H_
