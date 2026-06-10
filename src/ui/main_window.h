#pragma once

#include <cstdint>
#include <memory>

#include <QMainWindow>

class FrameSource;
class Pipeline;
class VideoWidget;
class QLineEdit;
class QPushButton;
class QLabel;

/**
 * Main application window for ZoneGuard-AI.
 *
 * Milestone 1: pick a source (camera index / file / RTSP URL), start/stop a
 * threaded capture, and show the live feed with an FPS readout. Detection,
 * pose, tracking, zones, and alerts are layered on in later milestones.
 */
class MainWindow : public QMainWindow
{
    Q_OBJECT

 public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

 private slots:
    void onStartStop();
    void onSourceEnded();
    void onError(QString const& message);
    void onFps(double fps);
    void onStats(std::int32_t numDetections, double inferenceMs);

 private:
    void setRunningState(bool running);

    std::unique_ptr<FrameSource> source_;
    std::unique_ptr<Pipeline> pipeline_;
    std::unique_ptr<VideoWidget> video_;
    std::unique_ptr<QLineEdit> source_edit_;
    std::unique_ptr<QPushButton> start_stop_button_;
    std::unique_ptr<QLabel> status_label_;
    std::unique_ptr<QLabel> fps_label_;

    bool running_{false};
};
