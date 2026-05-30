#pragma once
#include <QMainWindow>
#include <QPushButton>
#include <QLabel>
#include <QButtonGroup>
#include <QTimer>
#include <QHBoxLayout>
#include <QThread>
#include <QMutex>
#include <opencv2/opencv.hpp>
#include <QSlider>
#include <QLabel>

class CameraWorker : public QObject {
    Q_OBJECT
public:
    CameraWorker(cv::VideoCapture* capture) : cap(capture), running(true) {}
    void stop() { running = false; }

public slots:
    void process(); 

signals:
    void frameReady(cv::Mat frame); 

private:
    cv::VideoCapture* cap;
    bool running;
};

class RealTimeFiltering : public QMainWindow {
    Q_OBJECT
public:
    RealTimeFiltering(QWidget* parent = nullptr);
    ~RealTimeFiltering();

private slots:
    void onFrameReceived(cv::Mat frame); 
    void filterChanged(int id);

private:
    void addButtons(QHBoxLayout* topBar);

    QLabel* videoLabel;
    QLabel* fpsLabel;
    QButtonGroup* btnGroup;
    cv::VideoCapture cap;

    QThread* workerThread;
    CameraWorker* worker;

    int activeFilter = 0;
    cv::TickMeter tm;
    int frameCounter = 0;

    QList<QPushButton*> filterButtons;
    QList<int> filterPipeline;

    void addSliders(QVBoxLayout* layout);
    QSlider* highPassSlider;
    QSlider* lowPassSlider;
    QSlider* thresholdSlider;
    QSlider* edgeSlider;
    QSlider* medianSlider;
    QSlider* posterizeSlider;
};