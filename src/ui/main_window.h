#ifndef ZoneGuardAI_UI_MAIN_WINDOW_H_
#define ZoneGuardAI_UI_MAIN_WINDOW_H_

#include <cstdint>
#include <memory>

#include <QMainWindow>

namespace ZoneGuardAI
{
namespace Vision
{
class FrameSource;
}  // namespace Vision
namespace Core
{
class Pipeline;
}  // namespace Core
}  // namespace ZoneGuardAI

class QLineEdit;
class QPushButton;
class QLabel;

namespace ZoneGuardAI
{
namespace UI
{
class VideoWidget;

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

    std::unique_ptr<Vision::FrameSource> source_;
    std::unique_ptr<Core::Pipeline> pipeline_;
    std::unique_ptr<VideoWidget> video_;
    std::unique_ptr<QLineEdit> source_edit_;
    std::unique_ptr<QPushButton> start_stop_button_;
    std::unique_ptr<QLabel> status_label_;
    std::unique_ptr<QLabel> fps_label_;

    bool running_{false};
};

}  // namespace UI
}  // namespace ZoneGuardAI

#endif  // ZoneGuardAI_UI_MAIN_WINDOW_H_
