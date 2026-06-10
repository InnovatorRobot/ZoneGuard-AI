#include "core/pipeline.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <utility>

Pipeline::Pipeline() = default;

Pipeline::~Pipeline()
{
    stop();
}

bool Pipeline::loadModels(std::string const& models_dir)
{
    std::filesystem::path const detector_path{std::filesystem::path{models_dir} / "detector.onnx"};
    detector_loaded_ = detector_.load(detector_path.string());
    return detector_loaded_;
}

void Pipeline::start()
{
    if (!stopped_.load())
    {
        return;
    }
    stopped_.store(false);
    worker_ = std::thread(&Pipeline::run, this);
}

void Pipeline::stop()
{
    if (stopped_.exchange(true))
    {
        return;
    }
    cv_.notify_all();
    if (worker_.joinable())
    {
        worker_.join();
    }
}

void Pipeline::submit(cv::Mat const& bgr_frame)
{
    {
        std::lock_guard<std::mutex> lock{mutex_};
        pending_     = bgr_frame;  // latest wins; older un-processed frame is dropped
        has_pending_ = true;
    }
    cv_.notify_one();
}

void Pipeline::setFrameCallback(FrameCallback callback)
{
    std::lock_guard<std::mutex> lock{mutex_};
    frame_callback_ = std::move(callback);
}

void Pipeline::setStatsCallback(StatsCallback callback)
{
    std::lock_guard<std::mutex> lock{mutex_};
    stats_callback_ = std::move(callback);
}

void Pipeline::run()
{
    while (!stopped_.load())
    {
        cv::Mat frame{};
        {
            std::unique_lock<std::mutex> lock{mutex_};
            cv_.wait(lock, [this] { return has_pending_ || stopped_.load(); });
            if (stopped_.load())
            {
                break;
            }
            frame        = pending_;
            has_pending_ = false;
        }

        if (frame.empty())
        {
            continue;
        }

        std::int32_t num_detections{0};
        double inference_ms{0.0};
        cv::Mat const annotated{process(frame, num_detections, inference_ms)};

        FrameCallback frame_callback;
        StatsCallback stats_callback;
        {
            std::lock_guard<std::mutex> lock{mutex_};
            frame_callback = frame_callback_;
            stats_callback = stats_callback_;
        }

        if (frame_callback)
        {
            frame_callback(annotated);
        }
        if (stats_callback)
        {
            stats_callback(num_detections, inference_ms);
        }
    }
}

cv::Mat Pipeline::process(cv::Mat const& bgr, std::int32_t& num_detections, double& inference_ms)
{
    cv::Mat canvas{bgr};  // shared until we draw; clone only if we annotate

    num_detections = 0;
    inference_ms   = 0.0;

    if (detector_loaded_)
    {
        auto const t0{std::chrono::steady_clock::now()};
        Detections const detections{detector_.detect(bgr)};
        auto const t1{std::chrono::steady_clock::now()};
        inference_ms   = std::chrono::duration<double, std::milli>(t1 - t0).count();
        num_detections = static_cast<std::int32_t>(detections.size());

        canvas = bgr.clone();
        for (Detection const& detection : detections)
        {
            cv::rectangle(canvas,
                          cv::Point(static_cast<std::int32_t>(detection.x1),
                                    static_cast<std::int32_t>(detection.y1)),
                          cv::Point(static_cast<std::int32_t>(detection.x2),
                                    static_cast<std::int32_t>(detection.y2)),
                          cv::Scalar(0, 255, 0),
                          2);
            cv::putText(canvas,
                        cv::format("person %.0f%%", detection.score * 100.0F),
                        cv::Point(static_cast<std::int32_t>(detection.x1) + 3,
                                  static_cast<std::int32_t>(detection.y1) - 5),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.5,
                        cv::Scalar(0, 255, 0),
                        1);
        }
    }

    return canvas;
}
