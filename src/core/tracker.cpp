#include "core/tracker.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <utility>

namespace
{
constexpr float kInitConfidence{0.4F};

// Convert a (x1, y1, x2, y2) box to the Kalman measurement (cx, cy, a, h).
KalmanFilter::Measurement toXyah(float x1, float y1, float x2, float y2)
{
    double const width{static_cast<double>(x2 - x1)};
    double const height{static_cast<double>(y2 - y1)};
    double const centerX{static_cast<double>(x1) + width / 2.0};
    double const centerY{static_cast<double>(y1) + height / 2.0};
    double const aspect{(height != 0.0) ? width / height : 0.0};
    return {centerX, centerY, aspect, height};
}

// IoU between a (x1,y1,x2,y2) box and a candidate box.
double iou(cv::Vec4f const& box, TrackInput const& candidate)
{
    float const ix1{std::max(box[0], candidate.x1)};
    float const iy1{std::max(box[1], candidate.y1)};
    float const ix2{std::min(box[2], candidate.x2)};
    float const iy2{std::min(box[3], candidate.y2)};

    float const interW{std::max(0.0F, ix2 - ix1)};
    float const interH{std::max(0.0F, iy2 - iy1)};
    float const intersection{interW * interH};

    float const boxArea{(box[2] - box[0]) * (box[3] - box[1])};
    float const candidateArea{(candidate.x2 - candidate.x1) * (candidate.y2 - candidate.y1)};

    float const denominator{boxArea + candidateArea - intersection};
    if (denominator <= 0.0F)
    {
        return 0.0;
    }
    return static_cast<double>(intersection / denominator);
}

// Rectangular linear-sum-assignment via the O(n^3) Hungarian algorithm.
// Returns, for each row, the assigned column index (or -1 if unassigned).
std::vector<std::int32_t> solveAssignment(std::vector<std::vector<double>> const& cost)
{
    std::size_t const rows{cost.size()};
    std::size_t const cols{(rows == 0) ? 0U : cost.front().size()};
    if (rows == 0 || cols == 0)
    {
        return std::vector<std::int32_t>(rows, -1);
    }

    std::size_t const n{std::max(rows, cols)};
    constexpr double kLarge{1e12};

    // 1-indexed square cost matrix padded with a large constant.
    std::vector<std::vector<double>> a(n + 1, std::vector<double>(n + 1, 0.0));
    for (std::size_t i{1}; i <= n; ++i)
    {
        for (std::size_t j{1}; j <= n; ++j)
        {
            a[i][j] = (i <= rows && j <= cols) ? cost[i - 1][j - 1] : kLarge;
        }
    }

    std::vector<double> u(n + 1, 0.0);
    std::vector<double> v(n + 1, 0.0);
    std::vector<std::int32_t> p(n + 1, 0);
    std::vector<std::int32_t> way(n + 1, 0);

    for (std::size_t i{1}; i <= n; ++i)
    {
        p[0] = static_cast<std::int32_t>(i);
        std::size_t j0{0};
        std::vector<double> minv(n + 1, std::numeric_limits<double>::infinity());
        std::vector<char> used(n + 1, 0);

        do
        {
            used[j0] = 1;
            std::int32_t const i0{p[j0]};
            double delta{std::numeric_limits<double>::infinity()};
            std::int32_t j1{-1};

            for (std::size_t j{1}; j <= n; ++j)
            {
                if (used[j] != 0)
                {
                    continue;
                }
                double const cur{a[static_cast<std::size_t>(i0)][j] -
                                 u[static_cast<std::size_t>(i0)] - v[j]};
                if (cur < minv[j])
                {
                    minv[j] = cur;
                    way[j]  = static_cast<std::int32_t>(j0);
                }
                if (minv[j] < delta)
                {
                    delta = minv[j];
                    j1    = static_cast<std::int32_t>(j);
                }
            }

            for (std::size_t j{0}; j <= n; ++j)
            {
                if (used[j] != 0)
                {
                    u[static_cast<std::size_t>(p[j])] += delta;
                    v[j] -= delta;
                }
                else
                {
                    minv[j] -= delta;
                }
            }
            j0 = static_cast<std::size_t>(j1);
        } while (p[j0] != 0);

        do
        {
            std::int32_t const j1{way[j0]};
            p[j0] = p[static_cast<std::size_t>(j1)];
            j0    = static_cast<std::size_t>(j1);
        } while (j0 != 0);
    }

    std::vector<std::int32_t> rowToCol(rows, -1);
    for (std::size_t j{1}; j <= n; ++j)
    {
        std::int32_t const row{p[j]};
        if (row >= 1 && static_cast<std::size_t>(row) <= rows && j <= cols)
        {
            rowToCol[static_cast<std::size_t>(row) - 1] = static_cast<std::int32_t>(j) - 1;
        }
    }
    return rowToCol;
}
}  // namespace

// ---------------------------------------------------------------------------
// Track
// ---------------------------------------------------------------------------
Track::Track(cv::Mat mean,
             cv::Mat covariance,
             std::int32_t trackId,
             std::int32_t nInit,
             std::int32_t maxAge,
             std::int32_t bufferSize) :
    mean_{std::move(mean)},
    covariance_{std::move(covariance)},
    track_id_{trackId},
    n_init_{nInit},
    max_age_{maxAge},
    buffer_size_{static_cast<std::size_t>(std::max(1, bufferSize))}
{}

void Track::predict(KalmanFilter const& kf)
{
    kf.predict(mean_, covariance_);
    ++age_;
    ++time_since_update_;
}

void Track::update(KalmanFilter const& kf, TrackInput const& detection)
{
    kf.update(mean_, covariance_, toXyah(detection.x1, detection.y1, detection.x2, detection.y2));

    keypoints_list_.push_back(detection.keypoints);
    while (keypoints_list_.size() > buffer_size_)
    {
        keypoints_list_.pop_front();
    }

    ++hits_;
    time_since_update_ = 0;
    if (state_ == TrackState::Tentative && hits_ >= n_init_)
    {
        state_ = TrackState::Confirmed;
    }
}

void Track::markMissed()
{
    if (state_ == TrackState::Tentative)
    {
        state_ = TrackState::Deleted;
    }
    else if (time_since_update_ > max_age_)
    {
        state_ = TrackState::Deleted;
    }
}

cv::Vec4f Track::toTlbr() const
{
    double const centerX{mean_.at<double>(0, 0)};
    double const centerY{mean_.at<double>(1, 0)};
    double const aspect{mean_.at<double>(2, 0)};
    double const height{mean_.at<double>(3, 0)};
    double const width{aspect * height};

    float const x1{static_cast<float>(centerX - width / 2.0)};
    float const y1{static_cast<float>(centerY - height / 2.0)};
    return cv::Vec4f{x1, y1, static_cast<float>(x1 + width), static_cast<float>(y1 + height)};
}

cv::Point2f Track::center() const
{
    return cv::Point2f{static_cast<float>(mean_.at<double>(0, 0)),
                       static_cast<float>(mean_.at<double>(1, 0))};
}

// ---------------------------------------------------------------------------
// Tracker
// ---------------------------------------------------------------------------
Tracker::Tracker(float maxIouDistance,
                 std::int32_t maxAge,
                 std::int32_t nInit,
                 std::int32_t bufferSize) :
    max_iou_distance_{maxIouDistance},
    max_age_{maxAge},
    n_init_{nInit},
    buffer_size_{bufferSize}
{}

void Tracker::predict()
{
    for (Track& track : tracks_)
    {
        track.predict(kf_);
    }
}

void Tracker::update(std::vector<TrackInput> const& detections)
{
    MatchResult const result{match(detections)};

    for (auto const& [trackIndex, detectionIndex] : result.matches)
    {
        tracks_[static_cast<std::size_t>(trackIndex)].update(
            kf_,
            detections[static_cast<std::size_t>(detectionIndex)]);
    }
    for (std::int32_t const trackIndex : result.unmatchedTracks)
    {
        tracks_[static_cast<std::size_t>(trackIndex)].markMissed();
    }
    for (std::int32_t const detectionIndex : result.unmatchedDetections)
    {
        initiateTrack(detections[static_cast<std::size_t>(detectionIndex)]);
    }

    tracks_.erase(std::remove_if(tracks_.begin(),
                                 tracks_.end(),
                                 [](Track const& track) { return track.isDeleted(); }),
                  tracks_.end());
}

namespace
{
// Min-cost matching over the given track/detection index subsets using the IoU
// cost metric, replicating Track/linear_assignment.min_cost_matching.
struct LocalMatch
{
    std::vector<std::pair<std::int32_t, std::int32_t>> matches{};
    std::vector<std::int32_t> unmatchedTracks{};
    std::vector<std::int32_t> unmatchedDetections{};
};

LocalMatch minCostMatching(float maxDistance,
                           std::vector<Track> const& tracks,
                           std::vector<TrackInput> const& detections,
                           std::vector<std::int32_t> const& trackIndices,
                           std::vector<std::int32_t> const& detectionIndices)
{
    LocalMatch result{};
    if (trackIndices.empty() || detectionIndices.empty())
    {
        result.unmatchedTracks     = trackIndices;
        result.unmatchedDetections = detectionIndices;
        return result;
    }

    std::vector<std::vector<double>> cost(trackIndices.size(),
                                          std::vector<double>(detectionIndices.size(), 0.0));
    for (std::size_t row{0}; row < trackIndices.size(); ++row)
    {
        cv::Vec4f const box{tracks[static_cast<std::size_t>(trackIndices[row])].toTlbr()};
        for (std::size_t col{0}; col < detectionIndices.size(); ++col)
        {
            double value{1.0 -
                         iou(box, detections[static_cast<std::size_t>(detectionIndices[col])])};
            if (value > static_cast<double>(maxDistance))
            {
                value = static_cast<double>(maxDistance) + 1e-5;
            }
            cost[row][col] = value;
        }
    }

    std::vector<std::int32_t> const rowToCol{solveAssignment(cost)};

    std::set<std::int32_t> assignedDetections{};
    for (std::size_t row{0}; row < trackIndices.size(); ++row)
    {
        std::int32_t const col{rowToCol[row]};
        std::int32_t const trackIndex{trackIndices[row]};
        if (col < 0)
        {
            result.unmatchedTracks.push_back(trackIndex);
            continue;
        }

        std::int32_t const detectionIndex{detectionIndices[static_cast<std::size_t>(col)]};
        if (cost[row][static_cast<std::size_t>(col)] > static_cast<double>(maxDistance))
        {
            result.unmatchedTracks.push_back(trackIndex);
        }
        else
        {
            result.matches.emplace_back(trackIndex, detectionIndex);
            assignedDetections.insert(detectionIndex);
        }
    }

    for (std::int32_t const detectionIndex : detectionIndices)
    {
        if (assignedDetections.find(detectionIndex) == assignedDetections.end())
        {
            result.unmatchedDetections.push_back(detectionIndex);
        }
    }

    return result;
}

// Matching cascade prioritising recently-seen tracks
// (Track/linear_assignment.matching_cascade).
LocalMatch matchingCascade(float maxDistance,
                           std::int32_t cascadeDepth,
                           std::vector<Track> const& tracks,
                           std::vector<TrackInput> const& detections,
                           std::vector<std::int32_t> const& trackIndices)
{
    LocalMatch result{};
    std::vector<std::int32_t> unmatchedDetections(detections.size());
    for (std::size_t i{0}; i < detections.size(); ++i)
    {
        unmatchedDetections[i] = static_cast<std::int32_t>(i);
    }

    for (std::int32_t level{0}; level < cascadeDepth; ++level)
    {
        if (unmatchedDetections.empty())
        {
            break;
        }

        std::vector<std::int32_t> trackIndicesLevel{};
        for (std::int32_t const trackIndex : trackIndices)
        {
            if (tracks[static_cast<std::size_t>(trackIndex)].timeSinceUpdate() == 1 + level)
            {
                trackIndicesLevel.push_back(trackIndex);
            }
        }
        if (trackIndicesLevel.empty())
        {
            continue;
        }

        LocalMatch const levelMatch{minCostMatching(maxDistance,
                                                    tracks,
                                                    detections,
                                                    trackIndicesLevel,
                                                    unmatchedDetections)};
        result.matches.insert(result.matches.end(),
                              levelMatch.matches.begin(),
                              levelMatch.matches.end());
        unmatchedDetections = levelMatch.unmatchedDetections;
    }

    std::set<std::int32_t> matchedTracks{};
    for (auto const& [trackIndex, detectionIndex] : result.matches)
    {
        (void)detectionIndex;
        matchedTracks.insert(trackIndex);
    }
    for (std::int32_t const trackIndex : trackIndices)
    {
        if (matchedTracks.find(trackIndex) == matchedTracks.end())
        {
            result.unmatchedTracks.push_back(trackIndex);
        }
    }
    result.unmatchedDetections = unmatchedDetections;
    return result;
}
}  // namespace

Tracker::MatchResult Tracker::match(std::vector<TrackInput> const& detections) const
{
    std::vector<std::int32_t> confirmedTracks{};
    std::vector<std::int32_t> unconfirmedTracks{};
    for (std::size_t i{0}; i < tracks_.size(); ++i)
    {
        if (tracks_[i].isConfirmed())
        {
            confirmedTracks.push_back(static_cast<std::int32_t>(i));
        }
        else
        {
            unconfirmedTracks.push_back(static_cast<std::int32_t>(i));
        }
    }

    LocalMatch matchA{
        matchingCascade(max_iou_distance_, max_age_, tracks_, detections, confirmedTracks)};

    // Tracks that just went unmatched (age 1) get another chance alongside the
    // unconfirmed tracks; older unmatched tracks are kept as unmatched.
    std::vector<std::int32_t> trackCandidates{unconfirmedTracks};
    std::vector<std::int32_t> remainingUnmatched{};
    for (std::int32_t const trackIndex : matchA.unmatchedTracks)
    {
        if (tracks_[static_cast<std::size_t>(trackIndex)].timeSinceUpdate() == 1)
        {
            trackCandidates.push_back(trackIndex);
        }
        else
        {
            remainingUnmatched.push_back(trackIndex);
        }
    }

    LocalMatch matchB{minCostMatching(max_iou_distance_,
                                      tracks_,
                                      detections,
                                      trackCandidates,
                                      matchA.unmatchedDetections)};

    MatchResult result{};
    result.matches = matchA.matches;
    result.matches.insert(result.matches.end(), matchB.matches.begin(), matchB.matches.end());

    std::set<std::int32_t> unmatchedTrackSet{};
    for (std::int32_t const trackIndex : remainingUnmatched)
    {
        unmatchedTrackSet.insert(trackIndex);
    }
    for (std::int32_t const trackIndex : matchB.unmatchedTracks)
    {
        unmatchedTrackSet.insert(trackIndex);
    }
    result.unmatchedTracks.assign(unmatchedTrackSet.begin(), unmatchedTrackSet.end());
    result.unmatchedDetections = matchB.unmatchedDetections;
    return result;
}

void Tracker::initiateTrack(TrackInput const& detection)
{
    if (detection.confidence < kInitConfidence)
    {
        return;
    }

    cv::Mat mean{};
    cv::Mat covariance{};
    kf_.initiate(toXyah(detection.x1, detection.y1, detection.x2, detection.y2), mean, covariance);

    tracks_.emplace_back(std::move(mean),
                         std::move(covariance),
                         next_id_,
                         n_init_,
                         max_age_,
                         buffer_size_);
    ++next_id_;
}
