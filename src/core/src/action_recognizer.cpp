#include "core/action_recognizer.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <limits>

namespace ZoneGuardAI
{
namespace Core
{
namespace
{
constexpr float kScaleEpsilon{1e-6F};

std::string const kEmptyName{};
}  // namespace

ActionRecognizer::ActionRecognizer() : env_(ORT_LOGGING_LEVEL_WARNING, "zoneguard-action")
{
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

ActionRecognizer::~ActionRecognizer() = default;

bool ActionRecognizer::load(std::string const& onnxPath)
{
    try
    {
        session_ = std::make_unique<Ort::Session>(env_, onnxPath.c_str(), session_options_);

        std::size_t const inputCount{session_->GetInputCount()};
        if (inputCount < 2U)
        {
            std::cerr << "[ActionRecognizer] Expected two inputs (points, motion), got "
                      << inputCount << std::endl;
            session_.reset();
            return false;
        }

        Ort::AllocatorWithDefaultOptions allocator{};
        auto pointsName    = session_->GetInputNameAllocated(0, allocator);
        auto motionName    = session_->GetInputNameAllocated(1, allocator);
        auto outputName    = session_->GetOutputNameAllocated(0, allocator);
        points_input_name_ = pointsName.get();
        motion_input_name_ = motionName.get();
        output_name_       = outputName.get();
    }
    catch (Ort::Exception const& error)
    {
        std::cerr << "[ActionRecognizer] Failed to load '" << onnxPath << "': " << error.what()
                  << std::endl;
        session_.reset();
        return false;
    }
    return true;
}

std::string const& ActionRecognizer::className(std::int32_t classIndex) const
{
    if (classIndex < 0 || static_cast<std::size_t>(classIndex) >= class_names_.size())
    {
        return kEmptyName;
    }
    return class_names_[static_cast<std::size_t>(classIndex)];
}

ActionRecognizer::Result ActionRecognizer::predict(std::deque<Pose> const& keypointsList,
                                                   cv::Size imageSize) const
{
    Result result{};
    if (session_ == nullptr)
    {
        return result;
    }
    if (static_cast<std::int32_t>(keypointsList.size()) != time_steps_)
    {
        return result;
    }
    if (imageSize.width <= 0 || imageSize.height <= 0)
    {
        return result;
    }

    std::int32_t const timeSteps{time_steps_};
    std::int32_t const numNodes{num_nodes_};
    auto const t{static_cast<std::size_t>(timeSteps)};
    auto const v{static_cast<std::size_t>(numNodes)};

    // Points stream, layout N,C,T,V with C = (x, y, score). Normalize by frame
    // size, then scale each pose to [-1, 1] (matching pose_utils.scale_pose).
    std::vector<float> points(3U * t * v, 0.0F);
    auto const pointAt =
        [&points, t, v](std::size_t channel, std::size_t frame, std::size_t node) -> float& {
        return points[channel * t * v + frame * v + node];
    };

    float const width{static_cast<float>(imageSize.width)};
    float const height{static_cast<float>(imageSize.height)};

    for (std::size_t frame{0}; frame < t; ++frame)
    {
        Pose const& pose{keypointsList[frame]};
        if (static_cast<std::int32_t>(pose.keypoints.size()) < numNodes)
        {
            return result;
        }

        float minX{std::numeric_limits<float>::max()};
        float minY{std::numeric_limits<float>::max()};
        float maxX{std::numeric_limits<float>::lowest()};
        float maxY{std::numeric_limits<float>::lowest()};

        for (std::size_t node{0}; node < v; ++node)
        {
            PoseKeypoint const& keypoint{pose.keypoints[node]};
            float const normX{keypoint.x / width};
            float const normY{keypoint.y / height};
            pointAt(0U, frame, node) = normX;
            pointAt(1U, frame, node) = normY;
            pointAt(2U, frame, node) = keypoint.score;

            minX = std::min(minX, normX);
            minY = std::min(minY, normY);
            maxX = std::max(maxX, normX);
            maxY = std::max(maxY, normY);
        }

        float const rangeX{std::max(maxX - minX, kScaleEpsilon)};
        float const rangeY{std::max(maxY - minY, kScaleEpsilon)};
        for (std::size_t node{0}; node < v; ++node)
        {
            pointAt(0U, frame, node) = ((pointAt(0U, frame, node) - minX) / rangeX) * 2.0F - 1.0F;
            pointAt(1U, frame, node) = ((pointAt(1U, frame, node) - minY) / rangeY) * 2.0F - 1.0F;
        }
    }

    // Motion stream, layout N,C,T-1,V with C = (dx, dy): consecutive-frame
    // deltas of the scaled (x, y) coordinates.
    std::size_t const motionSteps{t - 1U};
    std::vector<float> motion(2U * motionSteps * v, 0.0F);
    for (std::size_t channel{0}; channel < 2U; ++channel)
    {
        for (std::size_t frame{0}; frame < motionSteps; ++frame)
        {
            for (std::size_t node{0}; node < v; ++node)
            {
                float const next{pointAt(channel, frame + 1U, node)};
                float const curr{pointAt(channel, frame, node)};
                motion[channel * motionSteps * v + frame * v + node] = next - curr;
            }
        }
    }

    std::array<std::int64_t, 4> const pointsShape{1,
                                                  3,
                                                  static_cast<std::int64_t>(timeSteps),
                                                  static_cast<std::int64_t>(numNodes)};
    std::array<std::int64_t, 4> const motionShape{1,
                                                  2,
                                                  static_cast<std::int64_t>(motionSteps),
                                                  static_cast<std::int64_t>(numNodes)};

    Ort::MemoryInfo memoryInfo{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    std::array<Ort::Value, 2> inputTensors{Ort::Value::CreateTensor<float>(memoryInfo,
                                                                           points.data(),
                                                                           points.size(),
                                                                           pointsShape.data(),
                                                                           pointsShape.size()),
                                           Ort::Value::CreateTensor<float>(memoryInfo,
                                                                           motion.data(),
                                                                           motion.size(),
                                                                           motionShape.data(),
                                                                           motionShape.size())};

    char const* const inputNames[]{points_input_name_.c_str(), motion_input_name_.c_str()};
    char const* const outputNames[]{output_name_.c_str()};

    std::vector<Ort::Value> outputs{};
    try
    {
        outputs = session_->Run(Ort::RunOptions{nullptr},
                                inputNames,
                                inputTensors.data(),
                                inputTensors.size(),
                                outputNames,
                                1);
    }
    catch (Ort::Exception const& error)
    {
        std::cerr << "[ActionRecognizer] Inference failed: " << error.what() << std::endl;
        return result;
    }

    if (outputs.empty())
    {
        return result;
    }

    std::vector<std::int64_t> const outputShape{
        outputs.front().GetTensorTypeAndShapeInfo().GetShape()};
    std::int64_t classCount{0};
    for (std::int64_t const dim : outputShape)
    {
        classCount = std::max<std::int64_t>(classCount, dim);
    }
    if (classCount <= 0)
    {
        return result;
    }

    float const* const scores{outputs.front().GetTensorData<float>()};
    std::int64_t bestIndex{0};
    float bestScore{scores[0]};
    for (std::int64_t i{1}; i < classCount; ++i)
    {
        if (scores[i] > bestScore)
        {
            bestScore = scores[i];
            bestIndex = i;
        }
    }

    result.classIndex = static_cast<std::int32_t>(bestIndex);
    result.confidence = bestScore;
    result.name       = className(result.classIndex);
    result.valid      = true;
    return result;
}

}  // namespace Core
}  // namespace ZoneGuardAI
