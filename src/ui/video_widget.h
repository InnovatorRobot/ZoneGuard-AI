#pragma once

#include <QImage>
#include <QWidget>

/**
 * Displays video frames with aspect-correct letterboxing.
 *
 * The latest frame is stored and painted in `paintEvent` via QPainter. Later
 * milestones will draw overlays here (skeletons, boxes, action labels,
 * monitoring zones), so all rendering goes through this single widget.
 */
class VideoWidget : public QWidget
{
    Q_OBJECT

 public:
    explicit VideoWidget(QWidget* parent = nullptr);

 public slots:
    /** Set the frame to display (triggers a repaint). */
    void setFrame(QImage const& frame);

    /** Clear the displayed frame. */
    void clear();

 protected:
    void paintEvent(QPaintEvent* event) override;
    QSize sizeHint() const override { return {640, 480}; }

 private:
    QImage frame_;
};
