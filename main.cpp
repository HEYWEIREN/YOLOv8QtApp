#include "YOLOv8QtApp.h"
#include <QApplication>

int main(int argc, char* argv[]) {
    QApplication a(argc, argv);
    YOLOv8QtApp w;
    w.show();
    return a.exec();
}