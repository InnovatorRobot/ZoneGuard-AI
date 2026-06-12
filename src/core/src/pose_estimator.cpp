#include "core/pose_estimator.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>

namespace ZoneGuardAI
{
namespace Core
{
PoseEstimator::PoseEstimator() : env_(ORT_LOGGING_LEVEL_WARNING, "zoneguard-pose")
{
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

PoseEstimator::~PoseEstimator() = default;

bool PoseEstimator::load(std::string const& onnxPath)
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
        std::cerr << "[PoseEstimator] Failed to load '" << onnxPath << "': " << error.what()
                  << std::endl;
        session_.reset();
        return false;
    }
    return true;
}

bool PoseEstimator::buildInput(cv::Mat const& bgr,
                               Detections const& detections,
                               std::vector<float>& input,
                               std::vector<CropRegion>& cropRegions,
                               std::vector<std::size_t>& detectionIndices) const
{
    input.clear();
    cropRegions.clear();
    detectionIndices.clear();

    if (bgr.empty() || detections.empty())
    {
        return false;
    }

    std::int32_t const frameHeight{bgr.rows};
    std::int32_t const frameWidth{bgr.cols};
    if (frameHeight <= 1 || frameWidth <= 1)
    {
        return false;
    }

    std::size_t const channelStride{static_cast<std::size_t>(input_height_) *
                                    static_cast<std::size_t>(input_width_)};

    for (std::size_t index{0}; index < detections.size(); ++index)
    {
        Detection const& detection{detections[index]};

        float const width{std::max(1.0F, detection.width())};
        float const height{std::max(1.0F, detection.height())};
        float const scaleRate{(width > 100.0F) ? 0.2F : 0.3F};

        float const halfScale{scaleRate * 0.5F};
        float const frameWidthMax{static_cast<float>(frameWidth - 1)};
        float const frameHeightMax{static_cast<float>(frameHeight - 1)};

        float upLeftX{std::max(0.0F, detection.x1 - width * halfScale)};
        float upLeftY{std::max(0.0F, detection.y1 - height * halfScale)};

        float bottomRightX{
            std::max(std::min(frameWidthMax, detection.x2 + width * halfScale), upLeftX + 5.0F)};
        float bottomRightY{
            std::max(std::min(frameHeightMax, detection.y2 + height * halfScale), upLeftY + 5.0F)};

        if (bottomRightX <= upLeftX || bottomRightY <= upLeftY)
        {
            continue;
        }

        CropRegion const cropRegion{upLeftX, upLeftY, bottomRightX, bottomRightY};
        cropRegions.push_back(cropRegion);
        detectionIndices.push_back(index);

        float const cropWidth{cropRegion.x2 - cropRegion.x1};
        float const cropHeight{cropRegion.y2 - cropRegion.y1};

        float const lenH{std::max(
            cropHeight,
            cropWidth * static_cast<float>(input_height_) / static_cast<float>(input_width_))};
        float const lenW{lenH * static_cast<float>(input_width_) /
                         static_cast<float>(input_height_)};

        float const padX{std::max(0.0F, (lenW - cropWidth) * 0.5F)};
        float const padY{std::max(0.0F, (lenH - cropHeight) * 0.5F)};

        std::array<cv::Point2f, 3> const src{
            {cv::Point2f(cropRegion.x1 - padX, cropRegion.y1 - padY),
             cv::Point2f(cropRegion.x2 + padX, cropRegion.y2 + padY),
             cv::Point2f(cropRegion.x1 - padX, cropRegion.y2 + padY)}};
        std::array<cv::Point2f, 3> const dst{
            {cv::Point2f(0.0F, 0.0F),
             cv::Point2f(static_cast<float>(input_width_ - 1),
                         static_cast<float>(input_height_ - 1)),
             cv::Point2f(0.0F, static_cast<float>(input_height_ - 1))}};

        cv::Mat const affine{cv::getAffineTransform(src.data(), dst.data())};
        cv::Mat crop{};
        cv::warpAffine(bgr,
                       crop,
                       affine,
                       cv::Size(input_width_, input_height_),
                       cv::INTER_LINEAR,
                       cv::BORDER_CONSTANT,
                       cv::Scalar(0, 0, 0));

        if (crop.type() != CV_8UC3)
        {
            crop.convertTo(crop, CV_8UC3);
        }

        std::size_t const personIndex{cropRegions.size() - 1U};
        input.resize((personIndex + 1U) * 3U * channelStride);
        std::size_t const base{personIndex * 3U * channelStride};

        // Keep BGR channel ordering to match the original crop_dets pipeline.
        for (std::int32_t y{0}; y < input_height_; ++y)
        {
            cv::Vec3b const* const row{crop.ptr<cv::Vec3b>(y)};
            for (std::int32_t x{0}; x < input_width_; ++x)
            {
                cv::Vec3b const& pixel{row[x]};
                std::size_t const offset{static_cast<std::size_t>(y) *
                                             static_cast<std::size_t>(input_width_) +
                                         static_cast<std::size_t>(x)};

                input[base + 0U * channelStride + offset] = pixel[0] / 255.0F - 0.406F;
                input[base + 1U * channelStride + offset] = pixel[1] / 255.0F - 0.457F;
                input[base + 2U * channelStride + offset] = pixel[2] / 255.0F - 0.480F;
            }
        }
    }

    return !cropRegions.empty();
}

PoseKeypoint PoseEstimator::transformBoxInvert(float x,
                                               float y,
                                               float score,
                                               CropRegion const& cropRegion,
                                               std::int64_t heatmapHeight,
                                               std::int64_t heatmapWidth) const
{
    if (score <= 0.0F)
    {
        return {};
    }

    float const centerX{(cropRegion.x2 - 1.0F - cropRegion.x1) * 0.5F};
    float const centerY{(cropRegion.y2 - 1.0F - cropRegion.y1) * 0.5F};

    float const cropWidth{cropRegion.x2 - cropRegion.x1};
    float const cropHeight{cropRegion.y2 - cropRegion.y1};

    float const lenH{
        std::max(cropHeight,
                 cropWidth * static_cast<float>(input_height_) / static_cast<float>(input_width_))};
    float const lenW{lenH * static_cast<float>(input_width_) / static_cast<float>(input_height_)};

    float px{x * lenH / static_cast<float>(heatmapHeight)};
    float py{y * lenH / static_cast<float>(heatmapHeight)};

    px -= std::max(0.0F, (lenW - 1.0F) * 0.5F - centerX);
    py -= std::max(0.0F, (lenH - 1.0F) * 0.5F - centerY);

    px += cropRegion.x1;
    py += cropRegion.y1;

    return PoseKeypoint{px, py, score};
}

Pose PoseEstimator::decodeOne(float const* heatmaps,
                              CropRegion const& cropRegion,
                              std::int64_t heatmapChannels,
                              std::int64_t heatmapHeight,
                              std::int64_t heatmapWidth) const
{
    Pose pose{};
    pose.keypoints.reserve(14U);

    if (heatmapChannels <= 0 || heatmapHeight <= 0 || heatmapWidth <= 0)
    {
        return pose;
    }

    std::size_t const heatmapStride{static_cast<std::size_t>(heatmapHeight) *
                                    static_cast<std::size_t>(heatmapWidth)};

    float scoreSum{0.0F};
    for (std::int32_t channel : keep_channels_)
    {
        if (channel < 0 || channel >= heatmapChannels)
        {
            pose.keypoints.push_back(PoseKeypoint{});
            continue;
        }

        float const* const plane{heatmaps + static_cast<std::size_t>(channel) * heatmapStride};

        std::size_t argmax{0U};
        float maxValue{std::numeric_limits<float>::lowest()};
        for (std::size_t i{0}; i < heatmapStride; ++i)
        {
            float const value{plane[i]};
            if (value > maxValue)
            {
                maxValue = value;
                argmax   = i;
            }
        }

        if (maxValue <= 0.0F)
        {
            pose.keypoints.push_back(PoseKeypoint{});
            continue;
        }

        float const x{static_cast<float>(argmax % static_cast<std::size_t>(heatmapWidth))};
        float const y{static_cast<float>(argmax / static_cast<std::size_t>(heatmapWidth))};

        PoseKeypoint const keypoint{
            transformBoxInvert(x, y, maxValue, cropRegion, heatmapHeight, heatmapWidth)};
        scoreSum += keypoint.score;
        pose.keypoints.push_back(keypoint);
    }

    if (pose.keypoints.size() >= 3U)
    {
        PoseKeypoint const& leftShoulder{pose.keypoints[1]};
        PoseKeypoint const& rightShoulder{pose.keypoints[2]};
        pose.keypoints.push_back(PoseKeypoint{(leftShoulder.x + rightShoulder.x) * 0.5F,
                                              (leftShoulder.y + rightShoulder.y) * 0.5F,
                                              (leftShoulder.score + rightShoulder.score) * 0.5F});
    }
    else
    {
        pose.keypoints.push_back(PoseKeypoint{});
    }

    if (!keep_channels_.empty())
    {
        pose.meanScore = scoreSum / static_cast<float>(keep_channels_.size());
    }
    return pose;
}

Poses PoseEstimator::estimate(cv::Mat const& bgr, Detections const& detections)
{
    Poses poses(detections.size());
    if (!isLoaded() || bgr.empty() || detections.empty())
    {
        return poses;
    }

    std::vector<float> input{};
    std::vector<CropRegion> cropRegions{};
    std::vector<std::size_t> detectionIndices{};
    if (!buildInput(bgr, detections, input, cropRegions, detectionIndices))
    {
        return poses;
    }

    std::array<std::int64_t, 4> const inputShape{static_cast<std::int64_t>(cropRegions.size()),
                                                 3,
                                                 static_cast<std::int64_t>(input_height_),
                                                 static_cast<std::int64_t>(input_width_)};

    Ort::MemoryInfo memoryInfo{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    Ort::Value inputTensor{Ort::Value::CreateTensor<float>(memoryInfo,
                                                           input.data(),
                                                           input.size(),
                                                           inputShape.data(),
                                                           inputShape.size())};

    char const* const inputNames[]{input_name_.c_str()};
    char const* const outputNames[]{output_name_.c_str()};

    std::vector<Ort::Value> outputs{};
    try
    {
        outputs =
            session_->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);
    }
    catch (Ort::Exception const& error)
    {
        std::cerr << "[PoseEstimator] Inference failed: " << error.what() << std::endl;
        return poses;
    }

    if (outputs.empty())
    {
        return poses;
    }

    std::vector<std::int64_t> const outputShape{
        outputs.front().GetTensorTypeAndShapeInfo().GetShape()};
    if (outputShape.size() != 4U)
    {
        return poses;
    }

    std::int64_t const numPersons{outputShape[0]};
    std::int64_t const heatmapChannels{outputShape[1]};
    std::int64_t const heatmapHeight{outputShape[2]};
    std::int64_t const heatmapWidth{outputShape[3]};
    if (numPersons <= 0 || heatmapChannels <= 0 || heatmapHeight <= 0 || heatmapWidth <= 0)
    {
        return poses;
    }

    float const* const outputData{outputs.front().GetTensorData<float>()};
    std::size_t const personStride{static_cast<std::size_t>(heatmapChannels) *
                                   static_cast<std::size_t>(heatmapHeight) *
                                   static_cast<std::size_t>(heatmapWidth)};

    std::size_t const decodedCount{std::min(static_cast<std::size_t>(numPersons),
                                            std::min(cropRegions.size(), detectionIndices.size()))};

    for (std::size_t i{0}; i < decodedCount; ++i)
    {
        float const* const personHeatmaps{outputData + i * personStride};
        Pose const pose{decodeOne(personHeatmaps,
                                  cropRegions[i],
                                  heatmapChannels,
                                  heatmapHeight,
                                  heatmapWidth)};
        poses[detectionIndices[i]] = pose;
    }

    return poses;
}

}  // namespace Core
}  // namespace ZoneGuardAI
