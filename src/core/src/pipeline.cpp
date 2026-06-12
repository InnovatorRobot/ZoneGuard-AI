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

namespace ZoneGuardAI
{
namespace Core
{
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

// Draw a monitoring zone polygon (and its name) onto the frame.
void drawZone(cv::Mat& canvas,
              Zone const& zone,
              cv::Size frameSize,
              cv::Scalar const& color,
              std::int32_t thickness)
{
    std::vector<cv::Point> const points{zone.toPixelPolygon(frameSize)};
    if (points.size() < 2U)
    {
        return;
    }

    std::vector<std::vector<cv::Point>> const polygons{points};
    cv::polylines(canvas, polygons, /*isClosed=*/true, color, thickness, cv::LINE_AA);

    if (!zone.name.empty())
    {
        cv::putText(canvas,
                    zone.name,
                    points.front() + cv::Point(4, 16),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv::LINE_AA);
    }
}
}  // namespace

Pipeline::Pipeline()
{
    // Default monitoring zone: a centered rectangle so the ROI feature is
    // visible out of the box. Replaced by the zone editor in a later milestone.
    Zone defaultZone{};
    defaultZone.name    = "Zone 1";
    defaultZone.enabled = true;
    defaultZone.polygon = {{0.2F, 0.2F}, {0.8F, 0.2F}, {0.8F, 0.8F}, {0.2F, 0.8F}};
    zone_monitor_.setZones({defaultZone});

    // Forward de-duplicated alerts to the registered UI callback, if any.
    notifier_.addSink([this](Alert const& alert) {
        AlertCallback callback;
        {
            std::lock_guard<std::mutex> lock{mutex_};
            callback = alert_callback_;
        }
        if (callback)
        {
            callback(alert);
        }
    });
}

Pipeline::~Pipeline()
{
    stop();
}

bool Pipeline::loadModels(std::string const& modelsDir)
{
    std::filesystem::path const detectorPath{std::filesystem::path{modelsDir} / "detector.onnx"};
    std::filesystem::path const posePath{std::filesystem::path{modelsDir} / "pose.onnx"};
    std::filesystem::path const actionPath{std::filesystem::path{modelsDir} / "action.onnx"};

    detector_loaded_ = detector_.load(detectorPath.string());
    pose_loaded_     = pose_estimator_.load(posePath.string());
    action_loaded_   = action_recognizer_.load(actionPath.string());
    return detector_loaded_ && pose_loaded_ && action_loaded_;
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

void Pipeline::setZones(std::vector<Zone> zones)
{
    std::lock_guard<std::mutex> lock{zones_mutex_};
    zone_monitor_.setZones(std::move(zones));
}

std::vector<Zone> Pipeline::zones() const
{
    std::lock_guard<std::mutex> lock{zones_mutex_};
    return zone_monitor_.zones();
}

void Pipeline::setAlertCallback(AlertCallback callback)
{
    std::lock_guard<std::mutex> lock{mutex_};
    alert_callback_ = std::move(callback);
}

void Pipeline::addAlertSink(NotificationClient::Sink sink)
{
    notifier_.addSink(std::move(sink));
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

    // Snapshot the monitoring zones once so the worker evaluates them lock-free.
    ZoneMonitor monitor{};
    {
        std::lock_guard<std::mutex> lock{zones_mutex_};
        monitor = zone_monitor_;
    }

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

        // Draw the monitoring zones underneath the detections/skeletons.
        for (Zone const& zone : monitor.zones())
        {
            if (zone.enabled)
            {
                drawZone(canvas, zone, bgr.size(), cv::Scalar(255, 255, 0), 2);
            }
        }

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

                // Recognize the action once a full pose window is buffered.
                // Ports main.py's per-track action label (default "pending..").
                std::string actionLabel{"pending.."};
                cv::Scalar actionColor{0, 255, 0};
                std::string fallAction{};
                float fallConfidence{0.0F};
                bool isFall{false};
                if (action_loaded_ &&
                    static_cast<std::int32_t>(track.keypointsList().size()) ==
                        action_recognizer_.timeSteps())
                {
                    ActionRecognizer::Result const action{
                        action_recognizer_.predict(track.keypointsList(), bgr.size())};
                    if (action.valid)
                    {
                        actionLabel = cv::format("%s: %.2f%%",
                                                 action.name.c_str(),
                                                 action.confidence * 100.0F);
                        if (action.name == "Fall Down")
                        {
                            actionColor    = cv::Scalar(255, 0, 0);
                            isFall         = true;
                            fallAction     = action.name;
                            fallConfidence = action.confidence;
                        }
                        else if (action.name == "Lying Down")
                        {
                            actionColor    = cv::Scalar(255, 200, 0);
                            isFall         = true;
                            fallAction     = action.name;
                            fallConfidence = action.confidence;
                        }
                    }
                }

                // A fall inside an enabled monitoring zone raises an alert. The
                // person's ground position is the bottom-center of the box.
                cv::Point2f const footPoint{(box[0] + box[2]) * 0.5F, box[3]};
                std::int32_t const zoneIndex{
                    monitor.empty() ? -1 : monitor.zoneAt(footPoint, bgr.size())};
                bool const isAlert{isFall && zoneIndex >= 0};

                cv::Scalar const boxColor{isAlert ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0)};
                cv::rectangle(
                    canvas,
                    cv::Point(static_cast<std::int32_t>(box[0]), static_cast<std::int32_t>(box[1])),
                    cv::Point(static_cast<std::int32_t>(box[2]), static_cast<std::int32_t>(box[3])),
                    boxColor,
                    2);

                if (isAlert)
                {
                    Zone const& zone{monitor.zones()[static_cast<std::size_t>(zoneIndex)]};
                    drawZone(canvas, zone, bgr.size(), cv::Scalar(0, 0, 255), 3);
                    cv::putText(canvas,
                                cv::format("ALERT: %s", zone.name.c_str()),
                                cv::Point(static_cast<std::int32_t>(box[0]),
                                          static_cast<std::int32_t>(box[1]) - 8),
                                cv::FONT_HERSHEY_COMPLEX,
                                0.5,
                                cv::Scalar(0, 0, 255),
                                2);

                    // Raise a (debounced) notification for this fall-in-zone.
                    Alert alert{};
                    alert.trackId     = track.id();
                    alert.zoneName    = zone.name;
                    alert.action      = fallAction;
                    alert.confidence  = fallConfidence;
                    alert.timestampMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                            std::chrono::system_clock::now().time_since_epoch())
                                            .count();
                    notifier_.notify(alert);
                }

                cv::Point2f const trackCenter{track.center()};
                cv::putText(canvas,
                            cv::format("ID %d", track.id()),
                            cv::Point(static_cast<std::int32_t>(trackCenter.x),
                                      static_cast<std::int32_t>(trackCenter.y)),
                            cv::FONT_HERSHEY_COMPLEX,
                            0.5,
                            cv::Scalar(255, 0, 0),
                            2);

                cv::putText(canvas,
                            actionLabel,
                            cv::Point(static_cast<std::int32_t>(box[0]) + 5,
                                      static_cast<std::int32_t>(box[1]) + 15),
                            cv::FONT_HERSHEY_COMPLEX,
                            0.4,
                            actionColor,
                            1);

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

}  // namespace Core
}  // namespace ZoneGuardAI
