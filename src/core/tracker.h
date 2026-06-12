#ifndef ZoneGuardAI_CORE_TRACKER_H_
#define ZoneGuardAI_CORE_TRACKER_H_

#include <cstdint>
#include <deque>
#include <vector>

#include <opencv2/opencv.hpp>

#include "core/kalman_filter.h"
#include "core/types.h"

namespace ZoneGuardAI
{
namespace Core
{
/**
 * A measurement fed to the tracker for a single frame: the person's bounding
 * box (x1, y1, x2, y2), its pose keypoints and an overall confidence.
 */
struct TrackInput
{
    float x1{0.0F};
    float y1{0.0F};
    float x2{0.0F};
    float y2{0.0F};
    Pose keypoints{};
    float confidence{0.0F};
};

enum class TrackState
{
    Tentative = 1,
    Confirmed = 2,
    Deleted   = 3
};

/**
 * A single tracked person. Holds the Kalman state plus a bounded history of
 * pose keypoints (used by the later action-recognition stage).
 *
 * Ports Track/Tracker.py's `Track`.
 */
class Track
{
 public:
    Track(cv::Mat mean,
          cv::Mat covariance,
          std::int32_t trackId,
          std::int32_t nInit,
          std::int32_t maxAge,
          std::int32_t bufferSize);

    void predict(KalmanFilter const& kf);
    void update(KalmanFilter const& kf, TrackInput const& detection);
    void markMissed();

    bool isTentative() const { return state_ == TrackState::Tentative; }
    bool isConfirmed() const { return state_ == TrackState::Confirmed; }
    bool isDeleted() const { return state_ == TrackState::Deleted; }

    /** Current box as (x1, y1, x2, y2) in image coordinates. */
    cv::Vec4f toTlbr() const;
    cv::Point2f center() const;

    std::int32_t id() const { return track_id_; }
    std::int32_t timeSinceUpdate() const { return time_since_update_; }

    cv::Mat const& mean() const { return mean_; }
    cv::Mat const& covariance() const { return covariance_; }

    std::deque<Pose> const& keypointsList() const { return keypoints_list_; }

 private:
    cv::Mat mean_;
    cv::Mat covariance_;

    std::int32_t track_id_{0};
    std::int32_t hits_{1};
    std::int32_t age_{1};
    std::int32_t time_since_update_{0};
    std::int32_t n_init_{0};
    std::int32_t max_age_{0};
    std::size_t buffer_size_{0};

    std::deque<Pose> keypoints_list_{};
    TrackState state_{TrackState::Tentative};
};

/**
 * Multi-person tracker: Kalman prediction + IoU matching cascade.
 *
 * Ports Track/Tracker.py's `Tracker`, Track/iou_matching.py and
 * Track/linear_assignment.py (the Hungarian solver replaces SciPy's
 * `linear_sum_assignment`). Qt-free, OpenCV-only.
 */
class Tracker
{
 public:
    explicit Tracker(float maxIouDistance    = 0.7F,
                     std::int32_t maxAge     = 30,
                     std::int32_t nInit      = 5,
                     std::int32_t bufferSize = 30);

    /** Propagate all track states one step forward. Call before `update`. */
    void predict();

    /** Associate detections to tracks and manage the track set. */
    void update(std::vector<TrackInput> const& detections);

    std::vector<Track> const& tracks() const { return tracks_; }

 private:
    struct MatchResult
    {
        std::vector<std::pair<std::int32_t, std::int32_t>> matches{};
        std::vector<std::int32_t> unmatchedTracks{};
        std::vector<std::int32_t> unmatchedDetections{};
    };

    MatchResult match(std::vector<TrackInput> const& detections) const;
    void initiateTrack(TrackInput const& detection);

    float max_iou_distance_{0.7F};
    std::int32_t max_age_{30};
    std::int32_t n_init_{5};
    std::int32_t buffer_size_{30};
    std::int32_t next_id_{1};

    KalmanFilter kf_{};
    std::vector<Track> tracks_{};
};

}  // namespace Core
}  // namespace ZoneGuardAI
#endif  // ZoneGuardAI_CORE_TRACKER_H_
