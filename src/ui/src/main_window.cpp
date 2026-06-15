#include "ui/main_window.h"

#include <QDateTime>
#include <QDockWidget>
#include <QFileDialog>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMetaObject>
#include <QProcessEnvironment>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QStatusBar>
#include <QVBoxLayout>
#include <QWidget>
#include <opencv2/opencv.hpp>

#include "core/pipeline.h"
#include "ui/video_widget.h"
#include "vision/frame_source.h"

namespace ZoneGuardAI
{
namespace UI
{
using Core::Pipeline;
using Vision::FrameSource;

namespace
{
/** Resolve the ONNX models directory: env override, else compile-time default. */
QString resolveModelsDir()
{
    QString const env_value =
        QProcessEnvironment::systemEnvironment().value(QStringLiteral("ZONEGUARD_MODELS_DIR"));
    if (!env_value.isEmpty())
    {
        return env_value;
    }
#ifdef ZONEGUARD_MODELS_DIR
    return QStringLiteral(ZONEGUARD_MODELS_DIR);
#else
    return QStringLiteral("models/onnx");
#endif
}

QImage toQImage(cv::Mat const& bgr)
{
    if (bgr.empty())
    {
        return {};
    }

    cv::Mat rgb{};
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    QImage const image(rgb.data,
                       rgb.cols,
                       rgb.rows,
                       static_cast<std::int32_t>(rgb.step),
                       QImage::Format_RGB888);
    return image.copy();
}
}  // namespace

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent)
{
    setWindowTitle(tr("ZoneGuard-AI"));

    auto* central_widget{new QWidget(this)};
    auto* main_layout{new QVBoxLayout(central_widget)};

    // --- Source control row ---
    auto* controls_layout{new QHBoxLayout{}};
    auto* source_label{new QLabel(tr("Source:"), central_widget)};
    source_edit_ = std::make_unique<QLineEdit>();
    source_edit_->setText(QStringLiteral("0"));
    source_edit_->setToolTip(tr("Camera index (e.g. 0), a video file path, or an RTSP/HTTP URL"));
    browse_button_ = std::make_unique<QPushButton>(tr("Browse..."));
    browse_button_->setToolTip(tr("Pick a video file"));
    start_stop_button_ = std::make_unique<QPushButton>(tr("Start"));

    controls_layout->addWidget(source_label);
    controls_layout->addWidget(source_edit_.get(), /*stretch=*/1);
    controls_layout->addWidget(browse_button_.get());
    controls_layout->addWidget(start_stop_button_.get());

    // --- Video display ---
    video_ = std::make_unique<VideoWidget>();

    main_layout->addLayout(controls_layout);
    main_layout->addWidget(video_.get(), /*stretch=*/1);
    setCentralWidget(central_widget);

    // --- Side panel: zone editor + settings + alerts (Part 9) ---
    auto* dock{new QDockWidget(tr("Monitoring"), this)};
    dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    auto* panel{new QWidget(dock)};
    auto* panel_layout{new QVBoxLayout(panel)};

    auto* zones_group{new QGroupBox(tr("Zones"), panel)};
    auto* zones_group_layout{new QVBoxLayout(zones_group)};
    zone_list_ = new QListWidget(zones_group);
    zone_list_->setToolTip(tr("Toggle the checkbox to enable/disable a zone"));
    auto* zone_buttons{new QHBoxLayout{}};
    draw_zone_button_   = new QPushButton(tr("Draw"), zones_group);
    finish_zone_button_ = new QPushButton(tr("Finish"), zones_group);
    cancel_zone_button_ = new QPushButton(tr("Cancel"), zones_group);
    delete_zone_button_ = new QPushButton(tr("Delete"), zones_group);
    zone_buttons->addWidget(draw_zone_button_);
    zone_buttons->addWidget(finish_zone_button_);
    zone_buttons->addWidget(cancel_zone_button_);
    zone_buttons->addWidget(delete_zone_button_);
    zones_group_layout->addWidget(zone_list_);
    zones_group_layout->addLayout(zone_buttons);

    auto* settings_group{new QGroupBox(tr("Settings"), panel)};
    auto* settings_layout{new QHBoxLayout(settings_group)};
    auto* cooldown_label{new QLabel(tr("Alert cooldown (s):"), settings_group)};
    cooldown_spin_ = new QSpinBox(settings_group);
    cooldown_spin_->setRange(0, 600);
    cooldown_spin_->setValue(5);
    settings_layout->addWidget(cooldown_label);
    settings_layout->addWidget(cooldown_spin_, /*stretch=*/1);

    auto* alerts_group{new QGroupBox(tr("Alerts"), panel)};
    auto* alerts_layout{new QVBoxLayout(alerts_group)};
    alerts_list_         = new QListWidget(alerts_group);
    clear_alerts_button_ = new QPushButton(tr("Clear"), alerts_group);
    alerts_layout->addWidget(alerts_list_, /*stretch=*/1);
    alerts_layout->addWidget(clear_alerts_button_);

    panel_layout->addWidget(zones_group);
    panel_layout->addWidget(settings_group);
    panel_layout->addWidget(alerts_group, /*stretch=*/1);
    dock->setWidget(panel);
    addDockWidget(Qt::RightDockWidgetArea, dock);

    // --- Status bar ---
    status_label_ = std::make_unique<QLabel>(tr("Idle"));
    fps_label_    = std::make_unique<QLabel>(tr("FPS: --"));
    statusBar()->addWidget(status_label_.get(), /*stretch=*/1);
    statusBar()->addPermanentWidget(fps_label_.get());

    // --- Processing pipeline (own worker thread) ---
    pipeline_ = std::make_unique<Pipeline>();
    if (pipeline_->loadModels(resolveModelsDir().toStdString()))
    {
        status_label_->setText(tr("Models loaded"));
    }
    else
    {
        status_label_->setText(tr("Warning: detector/pose model not fully loaded (degraded)"));
    }

    pipeline_->setFrameCallback([this](cv::Mat const& bgrFrame) {
        cv::Mat const frameCopy{bgrFrame.clone()};
        QMetaObject::invokeMethod(
            this,
            [this, frameCopy] { video_->setFrame(toQImage(frameCopy)); },
            Qt::QueuedConnection);
    });
    pipeline_->setStatsCallback([this](std::int32_t numDetections, double inferenceMs) {
        QMetaObject::invokeMethod(
            this,
            [this, numDetections, inferenceMs] { onStats(numDetections, inferenceMs); },
            Qt::QueuedConnection);
    });
    pipeline_->setAlertCallback([this](Core::Alert const& alert) {
        QString const zoneName{QString::fromStdString(alert.zoneName)};
        QString const action{QString::fromStdString(alert.action)};
        std::int32_t const trackId{alert.trackId};
        QMetaObject::invokeMethod(
            this,
            [this, zoneName, action, trackId] { onAlert(zoneName, action, trackId); },
            Qt::QueuedConnection);
    });
    pipeline_->start();

    // --- Capture source + wiring ---
    // Capture thread -> pipeline buffer (direct, so it just drops/queues a frame).
    source_ = std::make_unique<FrameSource>(this);
    connect(
        source_.get(),
        &FrameSource::frameReady,
        this,
        [this](cv::Mat const& frame) { pipeline_->submit(frame); },
        Qt::DirectConnection);

    connect(source_.get(), &FrameSource::fpsUpdated, this, &MainWindow::onFps);
    connect(source_.get(), &FrameSource::sourceEnded, this, &MainWindow::onSourceEnded);
    connect(source_.get(), &FrameSource::errorOccurred, this, &MainWindow::onError);

    connect(start_stop_button_.get(), &QPushButton::clicked, this, &MainWindow::onStartStop);
    connect(browse_button_.get(), &QPushButton::clicked, this, &MainWindow::onBrowse);
    connect(source_edit_.get(), &QLineEdit::returnPressed, this, &MainWindow::onStartStop);

    // --- Zone editor + settings + alerts wiring (Part 9) ---
    connect(draw_zone_button_, &QPushButton::clicked, this, &MainWindow::onDrawZone);
    connect(finish_zone_button_, &QPushButton::clicked, this, &MainWindow::onFinishZone);
    connect(cancel_zone_button_, &QPushButton::clicked, this, &MainWindow::onCancelZone);
    connect(delete_zone_button_, &QPushButton::clicked, this, &MainWindow::onDeleteZone);
    connect(zone_list_, &QListWidget::itemChanged, this, &MainWindow::onZoneItemChanged);
    connect(clear_alerts_button_, &QPushButton::clicked, this, &MainWindow::onClearAlerts);
    connect(cooldown_spin_,
            QOverload<int>::of(&QSpinBox::valueChanged),
            this,
            &MainWindow::onCooldownChanged);
    connect(video_.get(), &VideoWidget::editPointsChanged, this, [this](int count) {
        finish_zone_button_->setEnabled(editing_zones_ && count >= 3);
    });

    // Seed the editor from the pipeline's default zones.
    zones_ = pipeline_->zones();
    refreshZoneList();
    setEditingZones(false);

    resize(1100, 700);
}

MainWindow::~MainWindow()
{
    if (source_)
    {
        source_->stop();
    }
    if (pipeline_)
    {
        pipeline_->stop();
    }
}

void MainWindow::onStartStop()
{
    if (running_)
    {
        source_->stop();
        setRunningState(false);
        status_label_->setText(tr("Stopped"));
        video_->clear();
        return;
    }

    QString const source_text{source_edit_->text().trimmed()};
    if (source_text.isEmpty())
    {
        onError(tr("Please enter a source."));
        return;
    }

    status_label_->setText(tr("Opening %1...").arg(source_text));
    if (source_->start(source_text))
    {
        // Video files: process every frame (no drop) so the tracker sees small
        // inter-frame motion and pose windows fill - matching the reference.
        // Live cameras/streams keep latest-wins so we never lag behind.
        pipeline_->setDropOldFrames(!source_->isSeekableFile());
        setRunningState(true);
        status_label_->setText(tr("Running: %1").arg(source_text));
    }
    else
    {
        setRunningState(false);
    }
}

void MainWindow::onSourceEnded()
{
    setRunningState(false);
    status_label_->setText(tr("Source ended"));
    fps_label_->setText(tr("FPS: --"));
}

void MainWindow::onBrowse()
{
    QString const path{QFileDialog::getOpenFileName(
        this,
        tr("Open video file"),
        QString{},
        tr("Video files (*.mp4 *.avi *.mov *.mkv *.webm *.m4v);;All files (*)"))};
    if (!path.isEmpty())
    {
        source_edit_->setText(path);
    }
}

void MainWindow::onError(QString const& message)
{
    setRunningState(false);
    status_label_->setText(tr("Error: %1").arg(message));
}

void MainWindow::onFps(double fps)
{
    fps_label_->setText(tr("FPS: %1").arg(fps, 0, 'f', 1));
}

void MainWindow::onStats(std::int32_t numDetections, double inferenceMs)
{
    if (running_)
    {
        status_label_->setText(
            tr("Running - %1 person(s), %2 ms").arg(numDetections).arg(inferenceMs, 0, 'f', 1));
    }
}

void MainWindow::onAlert(QString const& zoneName, QString const& action, std::int32_t trackId)
{
    status_label_->setText(tr("ALERT: %1 (track %2) in %3").arg(action).arg(trackId).arg(zoneName));

    QString const timestamp{QDateTime::currentDateTime().toString("hh:mm:ss")};
    auto* item{new QListWidgetItem(
        tr("[%1] %2 (track %3) in %4").arg(timestamp).arg(action).arg(trackId).arg(zoneName))};
    item->setForeground(Qt::red);
    alerts_list_->insertItem(0, item);
    while (alerts_list_->count() > 200)
    {
        delete alerts_list_->takeItem(alerts_list_->count() - 1);
    }
}

void MainWindow::onDrawZone()
{
    setEditingZones(true);
    video_->setEditMode(true);
    status_label_->setText(tr("Click on the video to add zone points (>= 3), then Finish"));
}

void MainWindow::onFinishZone()
{
    QVector<QPointF> const points{video_->editPoints()};
    if (points.size() < 3)
    {
        status_label_->setText(tr("A zone needs at least 3 points"));
        return;
    }

    Core::Zone zone{};
    zone.name    = tr("Zone %1").arg(static_cast<int>(zones_.size()) + 1).toStdString();
    zone.enabled = true;
    zone.polygon.reserve(points.size());
    for (QPointF const& point : points)
    {
        zone.polygon.emplace_back(static_cast<float>(point.x()), static_cast<float>(point.y()));
    }
    zones_.push_back(std::move(zone));
    pipeline_->setZones(zones_);

    video_->setEditMode(false);
    setEditingZones(false);
    refreshZoneList();
    status_label_->setText(tr("Added zone (%1 total)").arg(static_cast<int>(zones_.size())));
}

void MainWindow::onCancelZone()
{
    video_->setEditMode(false);
    setEditingZones(false);
    status_label_->setText(tr("Zone drawing cancelled"));
}

void MainWindow::onDeleteZone()
{
    int const row{zone_list_->currentRow()};
    if (row < 0 || row >= static_cast<int>(zones_.size()))
    {
        return;
    }
    zones_.erase(zones_.begin() + row);
    pipeline_->setZones(zones_);
    refreshZoneList();
    status_label_->setText(tr("Deleted zone (%1 left)").arg(static_cast<int>(zones_.size())));
}

void MainWindow::onZoneItemChanged(QListWidgetItem* item)
{
    int const row{zone_list_->row(item)};
    if (row < 0 || row >= static_cast<int>(zones_.size()))
    {
        return;
    }
    bool const enabled{item->checkState() == Qt::Checked};
    if (zones_[static_cast<std::size_t>(row)].enabled != enabled)
    {
        zones_[static_cast<std::size_t>(row)].enabled = enabled;
        pipeline_->setZones(zones_);
    }
}

void MainWindow::onClearAlerts()
{
    alerts_list_->clear();
}

void MainWindow::onCooldownChanged(int seconds)
{
    pipeline_->setAlertCooldownMs(static_cast<std::int64_t>(seconds) * 1000);
}

void MainWindow::setRunningState(bool running)
{
    running_ = running;
    start_stop_button_->setText(running ? tr("Stop") : tr("Start"));
    source_edit_->setEnabled(!running);
    browse_button_->setEnabled(!running);
}

void MainWindow::setEditingZones(bool editing)
{
    editing_zones_ = editing;
    draw_zone_button_->setEnabled(!editing);
    finish_zone_button_->setEnabled(editing && video_->editPoints().size() >= 3);
    cancel_zone_button_->setEnabled(editing);
    delete_zone_button_->setEnabled(!editing);
    zone_list_->setEnabled(!editing);
}

void MainWindow::refreshZoneList()
{
    QSignalBlocker const blocker{zone_list_};
    zone_list_->clear();
    for (Core::Zone const& zone : zones_)
    {
        auto* item{new QListWidgetItem(QString::fromStdString(zone.name), zone_list_)};
        item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
        item->setCheckState(zone.enabled ? Qt::Checked : Qt::Unchecked);
    }
}

}  // namespace UI
}  // namespace ZoneGuardAI
