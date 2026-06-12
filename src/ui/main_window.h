#ifndef ZoneGuardAI_UI_MAIN_WINDOW_H_
#define ZoneGuardAI_UI_MAIN_WINDOW_H_

#include <cstdint>
#include <memory>
#include <vector>

#include <QMainWindow>

#include "core/zone.h"

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
class QListWidget;
class QListWidgetItem;
class QSpinBox;

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
    void onAlert(QString const& zoneName, QString const& action, std::int32_t trackId);
    void onDrawZone();
    void onFinishZone();
    void onCancelZone();
    void onDeleteZone();
    void onZoneItemChanged(QListWidgetItem* item);
    void onClearAlerts();
    void onCooldownChanged(int seconds);

 private:
    void setRunningState(bool running);
    void setEditingZones(bool editing);
    void refreshZoneList();

    std::unique_ptr<Vision::FrameSource> source_;
    std::unique_ptr<Core::Pipeline> pipeline_;
    std::unique_ptr<VideoWidget> video_;
    std::unique_ptr<QLineEdit> source_edit_;
    std::unique_ptr<QPushButton> start_stop_button_;
    std::unique_ptr<QLabel> status_label_;
    std::unique_ptr<QLabel> fps_label_;

    // Zone editor + alerts panel + settings (Part 9).
    QListWidget* zone_list_{nullptr};
    QPushButton* draw_zone_button_{nullptr};
    QPushButton* finish_zone_button_{nullptr};
    QPushButton* cancel_zone_button_{nullptr};
    QPushButton* delete_zone_button_{nullptr};
    QListWidget* alerts_list_{nullptr};
    QPushButton* clear_alerts_button_{nullptr};
    QSpinBox* cooldown_spin_{nullptr};

    std::vector<Core::Zone> zones_{};
    bool running_{false};
    bool editing_zones_{false};
};

}  // namespace UI
}  // namespace ZoneGuardAI

#endif  // ZoneGuardAI_UI_MAIN_WINDOW_H_
