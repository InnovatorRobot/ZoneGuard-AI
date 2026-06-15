#ifndef ZoneGuardAI_CORE_PIPELINE_H_
#define ZoneGuardAI_CORE_PIPELINE_H_

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "core/action_recognizer.h"
#include "core/detector.h"
#include "core/notification.h"
#include "core/pose_estimator.h"
#include "core/tracker.h"
#include "core/zone.h"

namespace ZoneGuardAI
{
namespace Core
{
/**
 * Vision processing worker.
 *
 * Runs on its own thread, consuming frames from a single-slot "latest wins"
 * buffer so that when inference is slower than capture, intermediate frames are
 * dropped instead of piling up. For each processed frame it runs the person
 * detector + pose estimator + tracker, draws the results, and publishes outputs
 * via callbacks.
 *
 * Later milestones extend `process` with action recognition and zone checks.
 */
class Pipeline
{
 public:
    using FrameCallback = std::function<void(cv::Mat const&)>;
    using StatsCallback = std::function<void(std::int32_t, double)>;
    using AlertCallback = std::function<void(Alert const&)>;

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

    /**
     * Frame buffering policy. When `drop` is true (default) the pipeline keeps
     * only the latest frame and drops anything the worker cannot keep up with
     * (correct for live cameras). When false it applies backpressure so every
     * submitted frame is processed (no drop) - used for video files so the
     * tracker sees small inter-frame motion, matching the reference's
     * CamLoader_Q which processes every frame.
     */
    void setDropOldFrames(bool drop);

    void setFrameCallback(FrameCallback callback);
    void setStatsCallback(StatsCallback callback);

    /** Replace the monitoring zones. Thread-safe; safe to call while running. */
    void setZones(std::vector<Zone> zones);

    /** Snapshot of the current monitoring zones. Thread-safe. */
    std::vector<Zone> zones() const;

    /** Invoked (on the worker thread) for each dispatched alert. */
    void setAlertCallback(AlertCallback callback);

    /** Register an extra alert sink (e.g. a webhook). Thread-safe. */
    void addAlertSink(NotificationClient::Sink sink);

    /** Cooldown between repeated alerts for the same track+zone (ms). */
    void setAlertCooldownMs(std::int64_t cooldownMs);

 private:
    void run();
    cv::Mat process(cv::Mat const& bgr, std::int32_t& numDetections, double& inferenceMs);

    Detector detector_;
    bool detector_loaded_{false};
    PoseEstimator pose_estimator_;
    bool pose_loaded_{false};
    Tracker tracker_{0.7F, 30, 3, 30};
    ActionRecognizer action_recognizer_;
    bool action_loaded_{false};

    mutable std::mutex zones_mutex_;
    ZoneMonitor zone_monitor_;

    NotificationClient notifier_;

    std::thread worker_;
    std::atomic<bool> stopped_{true};

    std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable consumed_cv_;
    std::atomic<bool> drop_old_frames_{true};
    cv::Mat pending_;
    bool has_pending_{false};

    FrameCallback frame_callback_;
    StatsCallback stats_callback_;
    AlertCallback alert_callback_;
};

}  // namespace Core
}  // namespace ZoneGuardAI
#endif  // ZoneGuardAI_CORE_PIPELINE_H_
