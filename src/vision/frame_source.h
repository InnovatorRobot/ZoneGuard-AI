#ifndef ZoneGuardAI_VISION_FRAME_SOURCE_H_
#define ZoneGuardAI_VISION_FRAME_SOURCE_H_

#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>

#include <QObject>
#include <QString>
#include <opencv2/opencv.hpp>

namespace ZoneGuardAI
{
namespace Vision
{
/**
 * Threaded video capture source.
 *
 * Wraps cv::VideoCapture (camera index, video file, or RTSP/HTTP URL) and grabs
 * frames on a dedicated worker thread so the UI thread is never blocked. Each
 * decoded frame is delivered via the `frameReady` signal as a deep-copied BGR
 * cv::Mat, ready for the processing Pipeline.
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
    cv::Size frameSize() const { return frame_size_; }

 signals:
    /** Emitted for every successfully decoded frame (deep-copied BGR). */
    void frameReady(cv::Mat const& frame);

    /** Measured throughput of the capture loop, in frames per second. */
    void fpsUpdated(double fps);

    /** Emitted when the source ends or cannot be read further. */
    void sourceEnded();

    /** Emitted on open/read errors with a human-readable message. */
    void errorOccurred(QString const& message);

 private:
    void run();

    cv::VideoCapture cap_;
    std::thread worker_;
    std::atomic<bool> running_{false};

    double fps_ = 0.0;
    cv::Size frame_size_{0, 0};

    // True for seekable video files: the capture loop is paced to the source
    // FPS so playback runs at natural speed instead of as fast as it decodes.
    bool throttle_to_fps_{false};
};

}  // namespace Vision
}  // namespace ZoneGuardAI
#endif  // ZoneGuardAI_VISION_FRAME_SOURCE_H_
