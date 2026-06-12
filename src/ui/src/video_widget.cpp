#include "ui/video_widget.h"

#include <QMouseEvent>
#include <QPainter>
#include <QPolygon>

namespace ZoneGuardAI
{
namespace UI
{
VideoWidget::VideoWidget(QWidget* parent) : QWidget(parent)
{
    setAttribute(Qt::WA_OpaquePaintEvent);
    setMinimumSize(320, 240);
    QPalette pal{palette()};
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
    frame_ = QImage{};
    update();
}

void VideoWidget::setEditMode(bool enabled)
{
    edit_mode_ = enabled;
    edit_points_.clear();
    setCursor(enabled ? Qt::CrossCursor : Qt::ArrowCursor);
    emit editPointsChanged(edit_points_.size());
    update();
}

void VideoWidget::clearEditPoints()
{
    edit_points_.clear();
    emit editPointsChanged(edit_points_.size());
    update();
}

QRect VideoWidget::frameTargetRect() const
{
    if (frame_.isNull())
    {
        return {};
    }
    QSize const scaled{frame_.size().scaled(size(), Qt::KeepAspectRatio)};
    return QRect(QPoint((width() - scaled.width()) / 2, (height() - scaled.height()) / 2), scaled);
}

void VideoWidget::mousePressEvent(QMouseEvent* event)
{
    if (!edit_mode_ || frame_.isNull() || event->button() != Qt::LeftButton)
    {
        QWidget::mousePressEvent(event);
        return;
    }

    QRect const target{frameTargetRect()};
    if (target.width() <= 0 || target.height() <= 0 || !target.contains(event->pos()))
    {
        return;
    }

    float const normalizedX{static_cast<float>(event->pos().x() - target.left()) /
                            static_cast<float>(target.width())};
    float const normalizedY{static_cast<float>(event->pos().y() - target.top()) /
                            static_cast<float>(target.height())};
    edit_points_.append(QPointF(normalizedX, normalizedY));
    emit editPointsChanged(edit_points_.size());
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
    QRect const target{frameTargetRect()};
    painter.setRenderHint(QPainter::SmoothPixmapTransform, true);
    painter.drawImage(target, frame_);

    if (!edit_mode_ || edit_points_.isEmpty())
    {
        return;
    }

    // Draw the in-progress zone polygon over the frame.
    QVector<QPoint> pixels{};
    pixels.reserve(edit_points_.size());
    for (QPointF const& point : edit_points_)
    {
        pixels.append(QPoint(target.left() + static_cast<int>(point.x() * target.width()),
                             target.top() + static_cast<int>(point.y() * target.height())));
    }

    QPen pen{QColor(255, 215, 0)};
    pen.setWidth(2);
    painter.setPen(pen);
    if (pixels.size() >= 2)
    {
        painter.drawPolyline(QPolygon(pixels));
        if (pixels.size() >= 3)
        {
            painter.drawLine(pixels.back(), pixels.front());
        }
    }
    painter.setBrush(QColor(255, 215, 0));
    for (QPoint const& pixel : pixels)
    {
        painter.drawEllipse(pixel, 4, 4);
    }
}

}  // namespace UI
}  // namespace ZoneGuardAI
