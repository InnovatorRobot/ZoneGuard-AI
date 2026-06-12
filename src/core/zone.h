#ifndef ZoneGuardAI_CORE_ZONE_H_
#define ZoneGuardAI_CORE_ZONE_H_

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace ZoneGuardAI
{
namespace Core
{
/**
 * A monitoring region of interest (ROI).
 *
 * The polygon is stored in normalized coordinates (each component in [0, 1])
 * so a zone is resolution-independent and follows the video when the frame
 * size changes. Helpers convert to pixel space and answer containment queries.
 */
struct Zone
{
    std::string name{};
    std::vector<cv::Point2f> polygon{};  // normalized [0, 1] vertices
    bool enabled{true};

    /** Polygon mapped to pixel coordinates for a given frame size. */
    std::vector<cv::Point> toPixelPolygon(cv::Size frameSize) const;

    /** True when a normalized point lies inside (or on) the polygon. */
    bool containsNormalized(cv::Point2f normalizedPoint) const;
};

/**
 * Holds the configured monitoring zones and answers containment queries.
 *
 * Qt-free; geometry uses OpenCV so the vision core stays reusable. The UI owns
 * the zone definitions and pushes them to the pipeline; the pipeline copies a
 * `ZoneMonitor` snapshot per frame for lock-free evaluation.
 */
class ZoneMonitor
{
 public:
    void setZones(std::vector<Zone> zones);
    std::vector<Zone> const& zones() const { return zones_; }

    bool empty() const { return zones_.empty(); }

    /**
     * Index of the first enabled zone containing the pixel point, or -1 when
     * the point is outside every enabled zone.
     */
    std::int32_t zoneAt(cv::Point2f pixelPoint, cv::Size frameSize) const;

 private:
    std::vector<Zone> zones_{};
};

}  // namespace Core
}  // namespace ZoneGuardAI
#endif  // ZoneGuardAI_CORE_ZONE_H_
