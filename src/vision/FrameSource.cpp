#include "vision/FrameSource.h"

#include <chrono>

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
    bool isIndex    = false;
    int const index = source.toInt(&isIndex);
    if (isIndex)
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

    fps_       = cap_.get(cv::CAP_PROP_FPS);
    frameSize_ = cv::Size(static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH)),
                          static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT)));

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
    cv::Mat frame;
    QElapsedTimer timer;
    timer.start();
    int frames = 0;

    while (running_.load())
    {
        if (!cap_.read(frame) || frame.empty())
        {
            // End of file or device disconnected.
            running_.store(false);
            emit sourceEnded();
            break;
        }

        emit frameReady(matToQImage(frame));

        // Report measured FPS roughly once per second.
        ++frames;
        qint64 const elapsed = timer.elapsed();
        if (elapsed >= 1000)
        {
            emit fpsUpdated(frames * 1000.0 / static_cast<double>(elapsed));
            frames = 0;
            timer.restart();
        }
    }
}

QImage FrameSource::matToQImage(cv::Mat const& bgr)
{
    if (bgr.empty())
    {
        return {};
    }

    // OpenCV is BGR; Qt wants RGB. Convert then deep-copy so the QImage owns
    // its pixels independently of the (reused) cv::Mat buffer.
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    QImage img(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
    return img.copy();
}
