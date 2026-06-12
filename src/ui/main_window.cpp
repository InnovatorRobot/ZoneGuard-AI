#include "ui/main_window.h"

#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMetaObject>
#include <QProcessEnvironment>
#include <QPushButton>
#include <QStatusBar>
#include <QVBoxLayout>
#include <QWidget>
#include <opencv2/opencv.hpp>

#include "core/pipeline.h"
#include "ui/video_widget.h"
#include "vision/frame_source.h"

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
    start_stop_button_ = std::make_unique<QPushButton>(tr("Start"));

    controls_layout->addWidget(source_label);
    controls_layout->addWidget(source_edit_.get(), /*stretch=*/1);
    controls_layout->addWidget(start_stop_button_.get());

    // --- Video display ---
    video_ = std::make_unique<VideoWidget>();

    main_layout->addLayout(controls_layout);
    main_layout->addWidget(video_.get(), /*stretch=*/1);
    setCentralWidget(central_widget);

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
    connect(source_edit_.get(), &QLineEdit::returnPressed, this, &MainWindow::onStartStop);

    resize(900, 680);
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

void MainWindow::setRunningState(bool running)
{
    running_ = running;
    start_stop_button_->setText(running ? tr("Stop") : tr("Start"));
    source_edit_->setEnabled(!running);
}
