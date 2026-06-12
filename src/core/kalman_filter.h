#ifndef ZoneGuardAI_CORE_KALMAN_FILTER_H_
#define ZoneGuardAI_CORE_KALMAN_FILTER_H_

#include <array>

#include <opencv2/opencv.hpp>

namespace ZoneGuardAI
{
namespace Core
{
/**
 * Constant-velocity Kalman filter for tracking bounding boxes in image space.
 *
 * Ports Track/kalman_filter.py from the reference project. The 8-dimensional
 * state is (x, y, a, h, vx, vy, va, vh): box center, aspect ratio, height and
 * their velocities. The measurement is (x, y, a, h). Matrices use double
 * precision (CV_64F) to match the reference's NumPy float64 maths.
 */
class KalmanFilter
{
 public:
    using Measurement = std::array<double, 4>;

    KalmanFilter();

    /** Create a track distribution from an unassociated (x, y, a, h) measurement. */
    void initiate(Measurement const& measurement, cv::Mat& mean, cv::Mat& covariance) const;

    /** Constant-velocity prediction step (updates mean/covariance in place). */
    void predict(cv::Mat& mean, cv::Mat& covariance) const;

    /** Project a state distribution into measurement space. */
    void project(cv::Mat const& mean,
                 cv::Mat const& covariance,
                 cv::Mat& projectedMean,
                 cv::Mat& projectedCovariance) const;

    /** Measurement correction step (updates mean/covariance in place). */
    void update(cv::Mat& mean, cv::Mat& covariance, Measurement const& measurement) const;

 private:
    cv::Mat motion_mat_;  // 8x8
    cv::Mat update_mat_;  // 4x8

    double std_weight_position_{1.0 / 20.0};
    double std_weight_velocity_{1.0 / 160.0};
};

}  // namespace Core
}  // namespace ZoneGuardAI
#endif  // ZoneGuardAI_CORE_KALMAN_FILTER_H_
