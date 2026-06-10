#include <cstdint>

#include <QApplication>
#include <QImage>
#include <QMetaType>
#include <opencv2/opencv.hpp>

#include "ui/main_window.h"

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

    // Types passed across thread boundaries via queued signal/slot connections.
    qRegisterMetaType<QImage>("QImage");
    qRegisterMetaType<cv::Mat>("cv::Mat");
    qRegisterMetaType<std::int32_t>("std::int32_t");

    QApplication::setApplicationName(QStringLiteral("ZoneGuard-AI"));
    QApplication::setApplicationVersion(QStringLiteral("0.1.0"));

    MainWindow window;
    window.show();

    return app.exec();
}
