#ifndef ZoneGuardAI_CORE_ACTION_RECOGNIZER_H_
#define ZoneGuardAI_CORE_ACTION_RECOGNIZER_H_

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include "core/types.h"

namespace ZoneGuardAI
{
namespace Core
{
/**
 * Two-Stream Spatial-Temporal Graph (TSSTG) action recognizer on ONNX Runtime.
 *
 * Ports ActionsEstLoader.TSSTG and pose_utils.py from the reference project:
 *   - take a window of `timeSteps` poses (14 graph nodes, neck appended),
 *   - normalize keypoints by frame size, then scale each pose to [-1, 1],
 *   - feed two streams: points (x, y, score) and motion (frame-to-frame
 *     deltas of x, y),
 *   - run the ONNX graph -> per-class scores, argmax -> action label.
 *
 * Qt-free so the vision core stays reusable.
 */
class ActionRecognizer
{
 public:
    struct Result
    {
        std::string name{};
        float confidence{0.0F};
        std::int32_t classIndex{-1};
        bool valid{false};
    };

    ActionRecognizer();
    ~ActionRecognizer();

    ActionRecognizer(ActionRecognizer const&)            = delete;
    ActionRecognizer& operator=(ActionRecognizer const&) = delete;

    /** Load an ONNX model. Returns false on failure. */
    bool load(std::string const& onnxPath);

    bool isLoaded() const { return session_ != nullptr; }

    /** Number of pose frames required for a prediction. */
    std::int32_t timeSteps() const { return time_steps_; }

    /** Class label for an index, or an empty string when out of range. */
    std::string const& className(std::int32_t classIndex) const;

    /**
     * Predict the action for a single track from its pose history.
     *
     * `keypointsList` must hold exactly `timeSteps()` poses, each with at least
     * `num_nodes_` keypoints. `imageSize` is the source frame size used for
     * normalization. Returns an invalid result when prerequisites are unmet.
     */
    Result predict(std::deque<Pose> const& keypointsList, cv::Size imageSize) const;

 private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;

    std::string points_input_name_;
    std::string motion_input_name_;
    std::string output_name_;

    std::int32_t time_steps_{30};
    std::int32_t num_nodes_{14};
    std::vector<std::string> class_names_{"Standing",
                                          "Walking",
                                          "Sitting",
                                          "Lying Down",
                                          "Stand up",
                                          "Sit down",
                                          "Fall Down"};
};

}  // namespace Core
}  // namespace ZoneGuardAI
#endif  // ZoneGuardAI_CORE_ACTION_RECOGNIZER_H_
