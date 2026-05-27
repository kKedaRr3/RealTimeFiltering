#pragma once
#include <QMainWindow>
#include <QPushButton>
#include <QLabel>
#include <QButtonGroup>
#include <QTimer>
#include <opencv2/opencv.hpp>
#include <QHBoxLayout>

class RealTimeFiltering : public QMainWindow {
    Q_OBJECT

public:
    RealTimeFiltering(QWidget* parent = nullptr);
    ~RealTimeFiltering();

private slots:
    void updateFrame();        // Wywoływane co 30ms
    void filterChanged(int id); // Wywoływane po kliknięciu przycisku

private:
    void addButtons(QHBoxLayout* topBar);
    QLabel* videoLabel;        // Miejsce na wideo
    QButtonGroup* btnGroup;    // Grupa przycisków
    cv::VideoCapture cap;      // Obiekt OpenCV do wideo
    QTimer* timer;             // Zegar odświeżania
    int activeFilter = 0;      // ID wybranego filtra

    cv::TickMeter tm;
    int frameCounter = 0;
    QLabel* fpsLabel;
};

