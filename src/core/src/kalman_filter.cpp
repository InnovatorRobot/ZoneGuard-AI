#include "core/kalman_filter.h"

#include <array>

namespace ZoneGuardAI
{
namespace Core
{
namespace
{
// Build a diagonal covariance matrix from per-element standard deviations
// (entries are squared, matching np.diag(np.square(std))).
cv::Mat diagonalFromStd(std::array<double, 8> const& std)
{
    cv::Mat covariance{cv::Mat::zeros(8, 8, CV_64F)};
    for (int i{0}; i < 8; ++i)
    {
        covariance.at<double>(i, i) = std[i] * std[i];
    }
    return covariance;
}
}  // namespace

KalmanFilter::KalmanFilter()
{
    constexpr int kNdim{4};
    constexpr double kDt{1.0};

    motion_mat_ = cv::Mat::eye(8, 8, CV_64F);
    for (int i{0}; i < kNdim; ++i)
    {
        motion_mat_.at<double>(i, kNdim + i) = kDt;
    }

    update_mat_ = cv::Mat::eye(kNdim, 8, CV_64F);
}

void KalmanFilter::initiate(Measurement const& measurement,
                            cv::Mat& mean,
                            cv::Mat& covariance) const
{
    mean = cv::Mat::zeros(8, 1, CV_64F);
    for (int i{0}; i < 4; ++i)
    {
        mean.at<double>(i, 0) = measurement[i];  // velocities stay zero
    }

    double const height{measurement[3]};
    std::array<double, 8> const std{2.0 * std_weight_position_ * height,
                                    2.0 * std_weight_position_ * height,
                                    1e-2,
                                    2.0 * std_weight_position_ * height,
                                    10.0 * std_weight_velocity_ * height,
                                    10.0 * std_weight_velocity_ * height,
                                    1e-5,
                                    10.0 * std_weight_velocity_ * height};
    covariance = diagonalFromStd(std);
}

void KalmanFilter::predict(cv::Mat& mean, cv::Mat& covariance) const
{
    double const height{mean.at<double>(3, 0)};
    std::array<double, 8> const std{std_weight_position_ * height,
                                    std_weight_position_ * height,
                                    1e-2,
                                    std_weight_position_ * height,
                                    std_weight_velocity_ * height,
                                    std_weight_velocity_ * height,
                                    1e-5,
                                    std_weight_velocity_ * height};
    cv::Mat const motionCovariance{diagonalFromStd(std)};

    mean       = motion_mat_ * mean;
    covariance = motion_mat_ * covariance * motion_mat_.t() + motionCovariance;
}

void KalmanFilter::project(cv::Mat const& mean,
                           cv::Mat const& covariance,
                           cv::Mat& projectedMean,
                           cv::Mat& projectedCovariance) const
{
    double const height{mean.at<double>(3, 0)};
    std::array<double, 4> const std{std_weight_position_ * height,
                                    std_weight_position_ * height,
                                    1e-1,
                                    std_weight_position_ * height};

    cv::Mat innovationCovariance{cv::Mat::zeros(4, 4, CV_64F)};
    for (int i{0}; i < 4; ++i)
    {
        innovationCovariance.at<double>(i, i) = std[i] * std[i];
    }

    projectedMean       = update_mat_ * mean;
    projectedCovariance = update_mat_ * covariance * update_mat_.t() + innovationCovariance;
}

void KalmanFilter::update(cv::Mat& mean, cv::Mat& covariance, Measurement const& measurement) const
{
    cv::Mat projectedMean{};
    cv::Mat projectedCovariance{};
    project(mean, covariance, projectedMean, projectedCovariance);

    // kalman_gain = covariance * update_mat^T * projected_cov^-1, computed via a
    // symmetric solve of projected_cov * gain^T = (covariance * update_mat^T)^T.
    cv::Mat const crossCovariance{covariance * update_mat_.t()};  // 8x4
    cv::Mat gainTransposed{};
    cv::solve(projectedCovariance, crossCovariance.t(), gainTransposed, cv::DECOMP_CHOLESKY);
    cv::Mat const kalmanGain{gainTransposed.t()};  // 8x4

    cv::Mat measurementVector{cv::Mat::zeros(4, 1, CV_64F)};
    for (int i{0}; i < 4; ++i)
    {
        measurementVector.at<double>(i, 0) = measurement[i];
    }

    cv::Mat const innovation{measurementVector - projectedMean};  // 4x1

    mean       = mean + kalmanGain * innovation;
    covariance = covariance - kalmanGain * projectedCovariance * kalmanGain.t();
}

}  // namespace Core
}  // namespace ZoneGuardAI
