#include "core/zone.h"

#include <cstddef>
#include <utility>

namespace ZoneGuardAI
{
namespace Core
{
std::vector<cv::Point> Zone::toPixelPolygon(cv::Size frameSize) const
{
    std::vector<cv::Point> pixels{};
    pixels.reserve(polygon.size());

    float const width{static_cast<float>(frameSize.width)};
    float const height{static_cast<float>(frameSize.height)};
    for (cv::Point2f const& vertex : polygon)
    {
        pixels.emplace_back(static_cast<std::int32_t>(vertex.x * width),
                            static_cast<std::int32_t>(vertex.y * height));
    }
    return pixels;
}

bool Zone::containsNormalized(cv::Point2f normalizedPoint) const
{
    if (polygon.size() < 3U)
    {
        return false;
    }
    return cv::pointPolygonTest(polygon, normalizedPoint, false) >= 0.0;
}

void ZoneMonitor::setZones(std::vector<Zone> zones)
{
    zones_ = std::move(zones);
}

std::int32_t ZoneMonitor::zoneAt(cv::Point2f pixelPoint, cv::Size frameSize) const
{
    if (frameSize.width <= 0 || frameSize.height <= 0)
    {
        return -1;
    }

    cv::Point2f const normalizedPoint{pixelPoint.x / static_cast<float>(frameSize.width),
                                      pixelPoint.y / static_cast<float>(frameSize.height)};

    for (std::size_t i{0}; i < zones_.size(); ++i)
    {
        if (zones_[i].enabled && zones_[i].containsNormalized(normalizedPoint))
        {
            return static_cast<std::int32_t>(i);
        }
    }
    return -1;
}

}  // namespace Core
}  // namespace ZoneGuardAI
