#include "vision/frame_source.h"

#include <chrono>
#include <cstdint>
#include <thread>

#include <QElapsedTimer>

namespace ZoneGuardAI
{
namespace Vision
{
FrameSource::FrameSource(QObject* parent) : QObject(parent)
{}

FrameSource::~FrameSource()
{
    stop();
}

bool FrameSource::start(QString const& source)
{
    stop();

    // A purely numeric source string is treated as a camera index; anything
    // else (file path or RTSP/HTTP URL) is opened as-is.
    bool is_index{false};
    std::int32_t const index{source.toInt(&is_index)};
    if (is_index)
    {
        cap_.open(index);
    }
    else
    {
        cap_.open(source.toStdString());
    }

    if (!cap_.isOpened())
    {
        emit errorOccurred(QStringLiteral("Cannot open source: %1").arg(source));
        return false;
    }

    fps_        = cap_.get(cv::CAP_PROP_FPS);
    frame_size_ = cv::Size(static_cast<std::int32_t>(cap_.get(cv::CAP_PROP_FRAME_WIDTH)),
                           static_cast<std::int32_t>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT)));

    // Seekable video files report a positive frame count; pace those to the
    // source FPS. Cameras and live streams are left unthrottled.
    double const frame_count{cap_.get(cv::CAP_PROP_FRAME_COUNT)};
    throttle_to_fps_ = (!is_index) && (frame_count > 0.0) && (fps_ > 0.0);

    running_.store(true);
    worker_ = std::thread(&FrameSource::run, this);
    return true;
}

void FrameSource::stop()
{
    running_.store(false);
    if (worker_.joinable())
    {
        worker_.join();
    }
    if (cap_.isOpened())
    {
        cap_.release();
    }
}

void FrameSource::run()
{
    cv::Mat frame{};
    QElapsedTimer timer{};
    timer.start();
    std::int32_t num_frames{0};

    // Target wall-clock interval between frames for paced file playback.
    double const frame_interval_ms{throttle_to_fps_ ? (1000.0 / fps_) : 0.0};
    QElapsedTimer pace_timer{};
    pace_timer.start();

    while (running_.load())
    {
        if (!cap_.read(frame) || frame.empty())
        {
            // End of file or device disconnected.
            running_.store(false);
            emit sourceEnded();
            break;
        }

        // Pace file playback to the source FPS so a video does not run at
        // decode speed. Sleep the remainder of this frame's time budget.
        if (throttle_to_fps_)
        {
            double const sleep_ms{frame_interval_ms -
                                  static_cast<double>(pace_timer.nsecsElapsed()) / 1.0e6};
            if (sleep_ms > 0.0)
            {
                std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(sleep_ms));
            }
            pace_timer.restart();
        }

        // Deep-copy: cv::VideoCapture reuses its internal buffer, so a clone
        // keeps the emitted frame valid for the consumer.
        emit frameReady(frame.clone());

        // Report measured FPS roughly once per second.
        ++num_frames;
        std::int64_t const elapsed_ms{timer.elapsed()};
        if (elapsed_ms >= 1000)
        {
            emit fpsUpdated(static_cast<double>(num_frames) * 1000.0 /
                            static_cast<double>(elapsed_ms));
            num_frames = 0;
            timer.restart();
        }
    }
}

}  // namespace Vision
}  // namespace ZoneGuardAI
