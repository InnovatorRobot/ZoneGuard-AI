#include "core/detector.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>

namespace
{
// Candidate box used internally during post-processing. Keeps the objectness
// confidence separately so the weighted NMS merge matches the reference.
struct Candidate
{
    float x1;
    float y1;
    float x2;
    float y2;
    float obj;
    float score;
};

// IoU with the +1 pixel convention used by Detection/Utils.bbox_iou.
float boxIou(Candidate const& lhs, Candidate const& rhs)
{
    float const ix1{std::max(lhs.x1, rhs.x1)};
    float const iy1{std::max(lhs.y1, rhs.y1)};
    float const ix2{std::min(lhs.x2, rhs.x2)};
    float const iy2{std::min(lhs.y2, rhs.y2)};

    float const intersection_w{std::max(0.0F, ix2 - ix1 + 1.0F)};
    float const intersection_h{std::max(0.0F, iy2 - iy1 + 1.0F)};
    float const intersection{intersection_w * intersection_h};

    float const lhs_area{(lhs.x2 - lhs.x1 + 1.0F) * (lhs.y2 - lhs.y1 + 1.0F)};
    float const rhs_area{(rhs.x2 - rhs.x1 + 1.0F) * (rhs.y2 - rhs.y1 + 1.0F)};

    return intersection / (lhs_area + rhs_area - intersection + 1e-16F);
}

// Non-Maximum-Suppression with the reference's confidence-weighted box merge.
// Single class, so detection labels always match.
Detections nonMaxSuppression(std::vector<Candidate> const& candidates_in, float nms_threshold)
{
    std::vector<Candidate> detections{candidates_in};
    std::sort(detections.begin(), detections.end(), [](Candidate const& lhs, Candidate const& rhs) {
        return lhs.score > rhs.score;
    });

    Detections output{};
    while (!detections.empty())
    {
        Candidate const top{detections.front()};

        float weighted_x1{0.0F};
        float weighted_y1{0.0F};
        float weighted_x2{0.0F};
        float weighted_y2{0.0F};
        float weight_sum{0.0F};
        std::vector<Candidate> remaining{};
        remaining.reserve(detections.size());
        for (Candidate const& detection : detections)
        {
            if (boxIou(top, detection) > nms_threshold)
            {
                weighted_x1 += detection.obj * detection.x1;
                weighted_y1 += detection.obj * detection.y1;
                weighted_x2 += detection.obj * detection.x2;
                weighted_y2 += detection.obj * detection.y2;
                weight_sum += detection.obj;
            }
            else
            {
                remaining.push_back(detection);
            }
        }

        Detection merged{};
        merged.x1    = weighted_x1 / weight_sum;
        merged.y1    = weighted_y1 / weight_sum;
        merged.x2    = weighted_x2 / weight_sum;
        merged.y2    = weighted_y2 / weight_sum;
        merged.score = top.score;
        output.push_back(merged);

        detections.swap(remaining);
    }
    return output;
}
}  // namespace

Detector::Detector() : env_(ORT_LOGGING_LEVEL_WARNING, "zoneguard-detector")
{
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

Detector::~Detector() = default;

bool Detector::load(std::string const& onnxPath)
{
    try
    {
        session_ = std::make_unique<Ort::Session>(env_, onnxPath.c_str(), session_options_);

        Ort::AllocatorWithDefaultOptions allocator{};
        auto inputName  = session_->GetInputNameAllocated(0, allocator);
        auto outputName = session_->GetOutputNameAllocated(0, allocator);
        input_name_     = inputName.get();
        output_name_    = outputName.get();
    }
    catch (Ort::Exception const& error)
    {
        std::cerr << "[Detector] Failed to load '" << onnxPath << "': " << error.what()
                  << std::endl;
        session_.reset();
        return false;
    }
    return true;
}

cv::Mat Detector::resizePadding(cv::Mat const& bgr, float& scale) const
{
    std::int32_t const height{bgr.rows};
    std::int32_t const width{bgr.cols};
    float const resizeRatio{static_cast<float>(input_size_) /
                            static_cast<float>(std::max(height, width))};

    std::int32_t const newWidth{static_cast<std::int32_t>(static_cast<float>(width) * resizeRatio)};
    std::int32_t const newHeight{
        static_cast<std::int32_t>(static_cast<float>(height) * resizeRatio)};

    cv::Mat resized{};
    cv::resize(bgr, resized, cv::Size(newWidth, newHeight));

    std::int32_t const deltaWidth{input_size_ - newWidth};
    std::int32_t const deltaHeight{input_size_ - newHeight};
    std::int32_t const top{deltaHeight / 2};
    std::int32_t const bottom{deltaHeight - top};
    std::int32_t const left{deltaWidth / 2};
    std::int32_t const right{deltaWidth - left};

    cv::Mat padded{};
    cv::copyMakeBorder(resized,
                       padded,
                       top,
                       bottom,
                       left,
                       right,
                       cv::BORDER_CONSTANT,
                       cv::Scalar(0, 0, 0));

    scale = resizeRatio;
    return padded;
}

Detections Detector::detect(cv::Mat const& bgr)
{
    if (!isLoaded() || bgr.empty())
    {
        return {};
    }

    std::int32_t const originalHeight{bgr.rows};
    std::int32_t const originalWidth{bgr.cols};

    // --- Preprocess: resize+pad -> RGB -> [0,1] -> NCHW ----------------------
    float resizeScale{1.0F};
    cv::Mat padded{resizePadding(bgr, resizeScale)};

    std::int32_t const side{input_size_};
    std::size_t const channelStride{static_cast<std::size_t>(side) *
                                    static_cast<std::size_t>(side)};
    std::vector<float> input(static_cast<std::size_t>(3) * channelStride);
    // Fill in CHW order with R, G, B channels normalized to [0,1].
    for (std::int32_t y{0}; y < side; ++y)
    {
        cv::Vec3b const* const row{padded.ptr<cv::Vec3b>(y)};
        for (std::int32_t x{0}; x < side; ++x)
        {
            cv::Vec3b const& pixel{row[x]};
            std::size_t const index{static_cast<std::size_t>(y) * static_cast<std::size_t>(side) +
                                    static_cast<std::size_t>(x)};
            input[0U * channelStride + index] = pixel[2] / 255.0F;
            input[1U * channelStride + index] = pixel[1] / 255.0F;
            input[2U * channelStride + index] = pixel[0] / 255.0F;
        }
    }

    std::array<std::int64_t, 4> const inputShape{1,
                                                 3,
                                                 static_cast<std::int64_t>(side),
                                                 static_cast<std::int64_t>(side)};

    Ort::MemoryInfo memory_info{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    Ort::Value input_tensor{Ort::Value::CreateTensor<float>(memory_info,
                                                            input.data(),
                                                            input.size(),
                                                            inputShape.data(),
                                                            inputShape.size())};

    // --- Inference -----------------------------------------------------------
    char const* const inputNames[]{input_name_.c_str()};
    char const* const outputNames[]{output_name_.c_str()};

    std::vector<Ort::Value> outputs{};
    try
    {
        outputs =
            session_->Run(Ort::RunOptions{nullptr}, inputNames, &input_tensor, 1, outputNames, 1);
    }
    catch (Ort::Exception const& error)
    {
        std::cerr << "[Detector] Inference failed: " << error.what() << std::endl;
        return {};
    }

    float const* const outputData{outputs.front().GetTensorData<float>()};
    std::vector<std::int64_t> const outputShape{
        outputs.front().GetTensorTypeAndShapeInfo().GetShape()};
    // Expected [1, N, 6] = cx, cy, w, h, obj_conf, class_conf.
    if (outputShape.size() != 3U || outputShape[2] < 6)
    {
        return {};
    }
    std::int64_t const numBoxes{outputShape[1]};
    std::int64_t const stride{outputShape[2]};

    // --- Confidence filter + xywh->xyxy --------------------------------------
    std::vector<Candidate> candidates{};
    candidates.reserve(static_cast<std::size_t>(numBoxes));
    for (std::int64_t boxIndex{0}; boxIndex < numBoxes; ++boxIndex)
    {
        float const* const box{outputData + boxIndex * stride};
        float const objectness{box[4]};
        if (objectness < conf_threshold_)
        {
            continue;
        }
        float const classConfidence{box[5]};
        float const centerX{box[0]};
        float const centerY{box[1]};
        float const boxWidth{box[2]};
        float const boxHeight{box[3]};
        candidates.push_back(Candidate{centerX - boxWidth / 2.0F,
                                       centerY - boxHeight / 2.0F,
                                       centerX + boxWidth / 2.0F,
                                       centerY + boxHeight / 2.0F,
                                       objectness,
                                       objectness * classConfidence});
    }
    if (candidates.empty())
    {
        return {};
    }

    Detections kept{nonMaxSuppression(candidates, nms_threshold_)};

    // --- Rescale boxes back to original frame + expand + clamp ---------------
    float const padX{(static_cast<float>(side) - resizeScale * static_cast<float>(originalWidth)) /
                     2.0F};
    float const padY{(static_cast<float>(side) - resizeScale * static_cast<float>(originalHeight)) /
                     2.0F};
    for (Detection& detection : kept)
    {
        detection.x1 = (detection.x1 - padX) / resizeScale;
        detection.x2 = (detection.x2 - padX) / resizeScale;
        detection.y1 = (detection.y1 - padY) / resizeScale;
        detection.y2 = (detection.y2 - padY) / resizeScale;

        detection.x1 = std::max(0.0F, detection.x1 - static_cast<float>(expand_pixels_));
        detection.y1 = std::max(0.0F, detection.y1 - static_cast<float>(expand_pixels_));
        detection.x2 = std::min(static_cast<float>(originalWidth),
                                detection.x2 + static_cast<float>(expand_pixels_));
        detection.y2 = std::min(static_cast<float>(originalHeight),
                                detection.y2 + static_cast<float>(expand_pixels_));
    }
    return kept;
}
