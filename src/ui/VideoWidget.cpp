#include "ui/VideoWidget.h"

#include <QPainter>

VideoWidget::VideoWidget(QWidget* parent) : QWidget(parent)
{
    setAttribute(Qt::WA_OpaquePaintEvent);
    setMinimumSize(320, 240);
    QPalette pal = palette();
    pal.setColor(QPalette::Window, Qt::black);
    setAutoFillBackground(true);
    setPalette(pal);
}

void VideoWidget::setFrame(QImage const& frame)
{
    frame_ = frame;
    update();
}

void VideoWidget::clear()
{
    frame_ = QImage();
    update();
}

void VideoWidget::paintEvent(QPaintEvent* /*event*/)
{
    QPainter painter(this);
    painter.fillRect(rect(), Qt::black);

    if (frame_.isNull())
    {
        painter.setPen(Qt::gray);
        painter.drawText(rect(), Qt::AlignCenter, tr("No video"));
        return;
    }

    // Scale to fit while preserving aspect ratio (letterbox), centered.
    QSize const scaled = frame_.size().scaled(size(), Qt::KeepAspectRatio);
    QRect const target(QPoint((width() - scaled.width()) / 2, (height() - scaled.height()) / 2),
                       scaled);
    painter.setRenderHint(QPainter::SmoothPixmapTransform, true);
    painter.drawImage(target, frame_);
}
