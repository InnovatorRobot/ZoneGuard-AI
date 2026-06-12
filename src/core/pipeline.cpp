#include "core/pipeline.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <utility>
#include <vector>

namespace
{
constexpr float kPoseDrawThreshold{0.05F};

// 14-node graph (13 kept COCO joints + appended neck at index 13).
std::array<std::pair<std::int32_t, std::int32_t>, 12> const kPoseEdges{{{0, 13},
                                                                        {1, 2},
                                                                        {1, 3},
                                                                        {3, 5},
                                                                        {2, 4},
                                                                        {4, 6},
                                                                        {13, 7},
                                                                        {13, 8},
                                                                        {7, 9},
                                                                        {8, 10},
                                                                        {9, 11},
                                                                        {10, 12}}};

std::array<cv::Scalar, 14> const kPointColors{{cv::Scalar(0, 255, 255),
                                               cv::Scalar(77, 255, 255),
                                               cv::Scalar(77, 255, 204),
                                               cv::Scalar(77, 204, 255),
                                               cv::Scalar(191, 255, 77),
                                               cv::Scalar(77, 191, 255),
                                               cv::Scalar(191, 255, 77),
                                               cv::Scalar(204, 77, 255),
                                               cv::Scalar(77, 255, 204),
                                               cv::Scalar(191, 77, 255),
                                               cv::Scalar(77, 255, 191),
                                               cv::Scalar(127, 77, 255),
                                               cv::Scalar(77, 255, 127),
                                               cv::Scalar(0, 255, 255)}};

std::array<cv::Scalar, 12> const kLineColors{{cv::Scalar(0, 215, 255),
                                              cv::Scalar(0, 255, 204),
                                              cv::Scalar(0, 134, 255),
                                              cv::Scalar(0, 255, 50),
                                              cv::Scalar(77, 255, 222),
                                              cv::Scalar(77, 196, 255),
                                              cv::Scalar(77, 135, 255),
                                              cv::Scalar(191, 255, 77),
                                              cv::Scalar(77, 255, 77),
                                              cv::Scalar(77, 222, 255),
                                              cv::Scalar(255, 156, 127),
                                              cv::Scalar(0, 127, 255)}};

void drawPose(cv::Mat& canvas, Pose const& pose)
{
    if (pose.keypoints.size() < kPointColors.size())
    {
        return;
    }

    std::array<bool, 14> visible{};
    std::array<cv::Point, 14> points{};

    for (std::size_t i{0}; i < kPointColors.size(); ++i)
    {
        PoseKeypoint const& keypoint{pose.keypoints[i]};
        if (keypoint.score <= kPoseDrawThreshold)
        {
            continue;
        }

        cv::Point const point{static_cast<std::int32_t>(keypoint.x),
                              static_cast<std::int32_t>(keypoint.y)};
        points[i]  = point;
        visible[i] = true;

        cv::circle(canvas, point, 3, kPointColors[i], -1, cv::LINE_AA);
    }

    for (std::size_t i{0}; i < kPoseEdges.size(); ++i)
    {
        auto const [startIndex, endIndex]{kPoseEdges[i]};
        if (!visible[startIndex] || !visible[endIndex])
        {
            continue;
        }

        cv::line(canvas, points[startIndex], points[endIndex], kLineColors[i], 2, cv::LINE_AA);
    }
}

// Bounding box that holds all valid keypoints, expanded by `expand` pixels.
// Ports main.kpt2bbox. Falls back to the detector box when no keypoint is valid.
TrackInput toTrackInput(Detection const& detection, Pose const& pose, float expand = 20.0F)
{
    TrackInput input{};
    input.keypoints = pose;

    float minX{std::numeric_limits<float>::max()};
    float minY{std::numeric_limits<float>::max()};
    float maxX{std::numeric_limits<float>::lowest()};
    float maxY{std::numeric_limits<float>::lowest()};

    float scoreSum{0.0F};
    std::int32_t validCount{0};
    for (PoseKeypoint const& keypoint : pose.keypoints)
    {
        scoreSum += keypoint.score;
        if (keypoint.score <= kPoseDrawThreshold)
        {
            continue;
        }
        minX = std::min(minX, keypoint.x);
        minY = std::min(minY, keypoint.y);
        maxX = std::max(maxX, keypoint.x);
        maxY = std::max(maxY, keypoint.y);
        ++validCount;
    }

    if (validCount > 0)
    {
        input.x1 = minX - expand;
        input.y1 = minY - expand;
        input.x2 = maxX + expand;
        input.y2 = maxY + expand;
        input.confidence =
            scoreSum / static_cast<float>(std::max<std::size_t>(1U, pose.keypoints.size()));
    }
    else
    {
        input.x1         = detection.x1;
        input.y1         = detection.y1;
        input.x2         = detection.x2;
        input.y2         = detection.y2;
        input.confidence = detection.score;
    }

    return input;
}
}  // namespace

Pipeline::Pipeline() = default;

Pipeline::~Pipeline()
{
    stop();
}

bool Pipeline::loadModels(std::string const& modelsDir)
{
    std::filesystem::path const detectorPath{std::filesystem::path{modelsDir} / "detector.onnx"};
    std::filesystem::path const posePath{std::filesystem::path{modelsDir} / "pose.onnx"};

    detector_loaded_ = detector_.load(detectorPath.string());
    pose_loaded_     = pose_estimator_.load(posePath.string());
    return detector_loaded_ && pose_loaded_;
}

void Pipeline::start()
{
    if (!stopped_.load())
    {
        return;
    }
    stopped_.store(false);
    worker_ = std::thread(&Pipeline::run, this);
}

void Pipeline::stop()
{
    if (stopped_.exchange(true))
    {
        return;
    }
    cv_.notify_all();
    if (worker_.joinable())
    {
        worker_.join();
    }
}

void Pipeline::submit(cv::Mat const& bgrFrame)
{
    {
        std::lock_guard<std::mutex> lock{mutex_};
        pending_     = bgrFrame;  // latest wins; older un-processed frame is dropped
        has_pending_ = true;
    }
    cv_.notify_one();
}

void Pipeline::setFrameCallback(FrameCallback callback)
{
    std::lock_guard<std::mutex> lock{mutex_};
    frame_callback_ = std::move(callback);
}

void Pipeline::setStatsCallback(StatsCallback callback)
{
    std::lock_guard<std::mutex> lock{mutex_};
    stats_callback_ = std::move(callback);
}

void Pipeline::run()
{
    while (!stopped_.load())
    {
        cv::Mat frame{};
        {
            std::unique_lock<std::mutex> lock{mutex_};
            cv_.wait(lock, [this] { return has_pending_ || stopped_.load(); });
            if (stopped_.load())
            {
                break;
            }
            frame        = pending_;
            has_pending_ = false;
        }

        if (frame.empty())
        {
            continue;
        }

        std::int32_t numDetections{0};
        double inferenceMs{0.0};
        cv::Mat const annotated{process(frame, numDetections, inferenceMs)};

        FrameCallback frameCallback;
        StatsCallback statsCallback;
        {
            std::lock_guard<std::mutex> lock{mutex_};
            frameCallback = frame_callback_;
            statsCallback = stats_callback_;
        }

        if (frameCallback)
        {
            frameCallback(annotated);
        }
        if (statsCallback)
        {
            statsCallback(numDetections, inferenceMs);
        }
    }
}

cv::Mat Pipeline::process(cv::Mat const& bgr, std::int32_t& numDetections, double& inferenceMs)
{
    cv::Mat canvas{bgr};  // shared until we draw; clone only if we annotate

    numDetections = 0;
    inferenceMs   = 0.0;

    if (detector_loaded_)
    {
        auto const t0{std::chrono::steady_clock::now()};
        Detections const detections{detector_.detect(bgr)};
        Poses poses{};
        if (pose_loaded_ && !detections.empty())
        {
            poses = pose_estimator_.estimate(bgr, detections);
        }
        auto const t1{std::chrono::steady_clock::now()};

        inferenceMs   = std::chrono::duration<double, std::milli>(t1 - t0).count();
        numDetections = static_cast<std::int32_t>(detections.size());

        canvas = bgr.clone();

        if (pose_loaded_)
        {
            // Build tracker measurements from the detected poses, then advance
            // the tracker one step.
            std::vector<TrackInput> trackInputs{};
            std::size_t const count{std::min(detections.size(), poses.size())};
            trackInputs.reserve(count);
            for (std::size_t i{0}; i < count; ++i)
            {
                trackInputs.push_back(toTrackInput(detections[i], poses[i]));
            }

            tracker_.predict();
            tracker_.update(trackInputs);

            for (Track const& track : tracker_.tracks())
            {
                if (!track.isConfirmed() || track.timeSinceUpdate() > 0)
                {
                    continue;
                }

                cv::Vec4f const box{track.toTlbr()};
                cv::rectangle(
                    canvas,
                    cv::Point(static_cast<std::int32_t>(box[0]), static_cast<std::int32_t>(box[1])),
                    cv::Point(static_cast<std::int32_t>(box[2]), static_cast<std::int32_t>(box[3])),
                    cv::Scalar(0, 255, 0),
                    2);

                cv::Point2f const trackCenter{track.center()};
                cv::putText(canvas,
                            cv::format("ID %d", track.id()),
                            cv::Point(static_cast<std::int32_t>(trackCenter.x),
                                      static_cast<std::int32_t>(trackCenter.y)),
                            cv::FONT_HERSHEY_COMPLEX,
                            0.5,
                            cv::Scalar(255, 0, 0),
                            2);

                if (!track.keypointsList().empty())
                {
                    drawPose(canvas, track.keypointsList().back());
                }
            }
        }
        else
        {
            for (Detection const& detection : detections)
            {
                cv::rectangle(canvas,
                              cv::Point(static_cast<std::int32_t>(detection.x1),
                                        static_cast<std::int32_t>(detection.y1)),
                              cv::Point(static_cast<std::int32_t>(detection.x2),
                                        static_cast<std::int32_t>(detection.y2)),
                              cv::Scalar(0, 255, 0),
                              2);
                cv::putText(canvas,
                            cv::format("person %.0f%%", detection.score * 100.0F),
                            cv::Point(static_cast<std::int32_t>(detection.x1) + 3,
                                      static_cast<std::int32_t>(detection.y1) - 5),
                            cv::FONT_HERSHEY_SIMPLEX,
                            0.5,
                            cv::Scalar(0, 255, 0),
                            1);
            }
        }
    }

    return canvas;
}
