#ifndef ZoneGuardAI_UI_VIDEO_WIDGET_H_
#define ZoneGuardAI_UI_VIDEO_WIDGET_H_

#include <QImage>
#include <QPointF>
#include <QRect>
#include <QVector>
#include <QWidget>

namespace ZoneGuardAI
{
namespace UI
{
/**
 * Displays video frames with aspect-correct letterboxing.
 *
 * The latest frame is stored and painted in `paintEvent` via QPainter. When in
 * zone-edit mode the widget collects clicked points (in normalized [0, 1]
 * image coordinates) and draws the in-progress polygon, so the user can sketch
 * a monitoring zone directly on the live video.
 */
class VideoWidget : public QWidget
{
    Q_OBJECT

 public:
    explicit VideoWidget(QWidget* parent = nullptr);

    /** Clicked vertices of the in-progress zone, in normalized [0, 1] coords. */
    QVector<QPointF> const& editPoints() const { return edit_points_; }

 public slots:
    /** Set the frame to display (triggers a repaint). */
    void setFrame(QImage const& frame);

    /** Clear the displayed frame. */
    void clear();

    /** Enable/disable zone drawing; clears any in-progress points. */
    void setEditMode(bool enabled);

    /** Discard the in-progress zone points. */
    void clearEditPoints();

 signals:
    /** Emitted whenever the in-progress zone points change. */
    void editPointsChanged(int count);

 protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    QSize sizeHint() const override { return {640, 480}; }

 private:
    /** Rectangle the (letterboxed) frame occupies inside the widget. */
    QRect frameTargetRect() const;

    QImage frame_;
    bool edit_mode_{false};
    QVector<QPointF> edit_points_{};
};

}  // namespace UI
}  // namespace ZoneGuardAI

#endif  // ZoneGuardAI_UI_VIDEO_WIDGET_H_
