#include "vision/frame_source.h"

#include <cstdint>

#include <QElapsedTimer>

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

    while (running_.load())
    {
        if (!cap_.read(frame) || frame.empty())
        {
            // End of file or device disconnected.
            running_.store(false);
            emit sourceEnded();
            break;
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
