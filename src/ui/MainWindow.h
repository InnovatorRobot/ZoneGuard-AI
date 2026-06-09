#pragma once

#include <QMainWindow>

class FrameSource;
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

 private:
    void setRunningState(bool running);

    FrameSource* source_          = nullptr;
    VideoWidget* video_           = nullptr;
    QLineEdit* sourceEdit_        = nullptr;
    QPushButton* startStopButton_ = nullptr;
    QLabel* statusLabel_          = nullptr;
    QLabel* fpsLabel_             = nullptr;

    bool running_ = false;
};
