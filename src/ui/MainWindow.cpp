#include "ui/MainWindow.h"

#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QStatusBar>
#include <QVBoxLayout>
#include <QWidget>

#include "ui/VideoWidget.h"
#include "vision/FrameSource.h"

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent)
{
    setWindowTitle(tr("ZoneGuard-AI"));

    auto* central = new QWidget(this);
    auto* layout  = new QVBoxLayout(central);

    // --- Source control row ---
    auto* controls    = new QHBoxLayout();
    auto* sourceLabel = new QLabel(tr("Source:"), central);
    sourceEdit_       = new QLineEdit(central);
    sourceEdit_->setText(QStringLiteral("0"));
    sourceEdit_->setToolTip(tr("Camera index (e.g. 0), a video file path, or an RTSP/HTTP URL"));
    startStopButton_ = new QPushButton(tr("Start"), central);

    controls->addWidget(sourceLabel);
    controls->addWidget(sourceEdit_, /*stretch=*/1);
    controls->addWidget(startStopButton_);

    // --- Video display ---
    video_ = new VideoWidget(central);

    layout->addLayout(controls);
    layout->addWidget(video_, /*stretch=*/1);
    setCentralWidget(central);

    // --- Status bar ---
    statusLabel_ = new QLabel(tr("Idle"), this);
    fpsLabel_    = new QLabel(tr("FPS: --"), this);
    statusBar()->addWidget(statusLabel_, /*stretch=*/1);
    statusBar()->addPermanentWidget(fpsLabel_);

    // --- Capture source + wiring ---
    source_ = new FrameSource(this);
    connect(source_, &FrameSource::frameReady, video_, &VideoWidget::setFrame);
    connect(source_, &FrameSource::fpsUpdated, this, &MainWindow::onFps);
    connect(source_, &FrameSource::sourceEnded, this, &MainWindow::onSourceEnded);
    connect(source_, &FrameSource::errorOccurred, this, &MainWindow::onError);

    connect(startStopButton_, &QPushButton::clicked, this, &MainWindow::onStartStop);
    connect(sourceEdit_, &QLineEdit::returnPressed, this, &MainWindow::onStartStop);

    resize(900, 680);
}

MainWindow::~MainWindow()
{
    if (source_)
    {
        source_->stop();
    }
}

void MainWindow::onStartStop()
{
    if (running_)
    {
        source_->stop();
        setRunningState(false);
        statusLabel_->setText(tr("Stopped"));
        video_->clear();
        return;
    }

    QString const src = sourceEdit_->text().trimmed();
    if (src.isEmpty())
    {
        onError(tr("Please enter a source."));
        return;
    }

    statusLabel_->setText(tr("Opening %1...").arg(src));
    if (source_->start(src))
    {
        setRunningState(true);
        statusLabel_->setText(tr("Running: %1").arg(src));
    }
    else
    {
        setRunningState(false);
    }
}

void MainWindow::onSourceEnded()
{
    setRunningState(false);
    statusLabel_->setText(tr("Source ended"));
    fpsLabel_->setText(tr("FPS: --"));
}

void MainWindow::onError(QString const& message)
{
    setRunningState(false);
    statusLabel_->setText(tr("Error: %1").arg(message));
}

void MainWindow::onFps(double fps)
{
    fpsLabel_->setText(tr("FPS: %1").arg(fps, 0, 'f', 1));
}

void MainWindow::setRunningState(bool running)
{
    running_ = running;
    startStopButton_->setText(running ? tr("Stop") : tr("Start"));
    sourceEdit_->setEnabled(!running);
}
