#ifndef ZoneGuardAI_CORE_TYPES_H_
#define ZoneGuardAI_CORE_TYPES_H_

#include <vector>

namespace ZoneGuardAI
{
namespace Core
{
/**
 * A single person detection in original-frame pixel coordinates.
 *
 * Box is top-left / bottom-right (tlbr). `score` is the combined
 * objectness * class confidence used for ranking and thresholding.
 */
struct Detection
{
    float x1{0.0F};
    float y1{0.0F};
    float x2{0.0F};
    float y2{0.0F};
    float score{0.0F};

    float width() const { return x2 - x1; }
    float height() const { return y2 - y1; }
};

using Detections = std::vector<Detection>;

struct PoseKeypoint
{
    float x{0.0F};
    float y{0.0F};
    float score{0.0F};
};

using PoseKeypoints = std::vector<PoseKeypoint>;

struct Pose
{
    PoseKeypoints keypoints{};
    float meanScore{0.0F};
};

using Poses = std::vector<Pose>;

}  // namespace Core
}  // namespace ZoneGuardAI
#endif  // ZoneGuardAI_CORE_TYPES_H_
