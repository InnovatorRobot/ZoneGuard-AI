#include <QApplication>
#include <QImage>
#include <QMetaType>

#include "ui/MainWindow.h"

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

    // QImage is queued across the capture thread -> GUI thread connection.
    // It is registered by Qt by default, but declare it explicitly for clarity.
    qRegisterMetaType<QImage>("QImage");

    QApplication::setApplicationName(QStringLiteral("ZoneGuard-AI"));
    QApplication::setApplicationVersion(QStringLiteral("0.1.0"));

    MainWindow window;
    window.show();

    return app.exec();
}
