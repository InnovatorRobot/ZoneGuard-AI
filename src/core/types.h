#pragma once

#include <vector>

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
