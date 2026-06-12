#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

#include <opencv2/opencv.hpp>

#include "core/detector.h"
#include "core/pose_estimator.h"

/**
 * Vision processing worker.
 *
 * Runs on its own thread, consuming frames from a single-slot "latest wins"
 * buffer so that when inference is slower than capture, intermediate frames are
 * dropped instead of piling up. For each processed frame it runs the person
 * detector + pose estimator, draws the results, and publishes outputs via
 * callbacks.
 *
 * Later milestones extend `process` with tracking, action
 * recognition and zone checks.
 */
class Pipeline
{
 public:
    using FrameCallback = std::function<void(cv::Mat const&)>;
    using StatsCallback = std::function<void(std::int32_t, double)>;

    Pipeline();
    ~Pipeline();

    Pipeline(Pipeline const&)            = delete;
    Pipeline& operator=(Pipeline const&) = delete;

    /** Load models. `modelsDir` holds detector.onnx etc. Safe before start(). */
    bool loadModels(std::string const& modelsDir);

    void start();
    void stop();

    /** Submit a BGR frame (deep-copied by the caller). Thread-safe; drops old. */
    void submit(cv::Mat const& bgrFrame);

    void setFrameCallback(FrameCallback callback);
    void setStatsCallback(StatsCallback callback);

 private:
    void run();
    cv::Mat process(cv::Mat const& bgr, std::int32_t& numDetections, double& inferenceMs);

    Detector detector_;
    bool detector_loaded_{false};
    PoseEstimator pose_estimator_;
    bool pose_loaded_{false};

    std::thread worker_;
    std::atomic<bool> stopped_{true};

    std::mutex mutex_;
    std::condition_variable cv_;
    cv::Mat pending_;
    bool has_pending_{false};

    FrameCallback frame_callback_;
    StatsCallback stats_callback_;
};
