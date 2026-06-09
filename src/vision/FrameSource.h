#pragma once

#include <atomic>
#include <mutex>
#include <thread>

#include <QImage>
#include <QObject>
#include <QString>
#include <opencv2/opencv.hpp>

/**
 * Threaded video capture source.
 *
 * Wraps cv::VideoCapture (camera index, video file, or RTSP/HTTP URL) and grabs
 * frames on a dedicated worker thread so the UI thread is never blocked. Each
 * decoded frame is delivered via the `frameReady` signal as a deep-copied
 * QImage (safe to use directly on the GUI thread).
 *
 * This mirrors CameraLoader.py from the reference project, but uses Qt signals
 * instead of a Python queue/lock.
 */
class FrameSource : public QObject
{
    Q_OBJECT

 public:
    explicit FrameSource(QObject* parent = nullptr);
    ~FrameSource() override;

    /** Open a camera index (as string, e.g. "0") or a file/URL path. */
    bool start(QString const& source);

    /** Stop the worker thread and release the capture device. */
    void stop();

    bool isRunning() const { return running_.load(); }

    double sourceFps() const { return fps_; }
    cv::Size frameSize() const { return frameSize_; }

 signals:
    /** Emitted (queued) for every successfully decoded frame. */
    void frameReady(QImage const& frame);

    /** Measured throughput of the capture loop, in frames per second. */
    void fpsUpdated(double fps);

    /** Emitted when the source ends or cannot be read further. */
    void sourceEnded();

    /** Emitted on open/read errors with a human-readable message. */
    void errorOccurred(QString const& message);

 private:
    void run();
    static QImage matToQImage(cv::Mat const& bgr);

    cv::VideoCapture cap_;
    std::thread worker_;
    std::atomic<bool> running_{false};

    double fps_ = 0.0;
    cv::Size frameSize_{0, 0};
};
