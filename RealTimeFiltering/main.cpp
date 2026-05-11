#include <QApplication>
#include "RealTimeFiltering.h"

int main(int argc, char* argv[]) {
    QApplication a(argc, argv);
    RealTimeFiltering w;
    w.show();
    return a.exec();
}