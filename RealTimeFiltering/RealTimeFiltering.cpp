#include "RealTimeFiltering.h"
#include <QVBoxLayout>
#include <QImage>
#include <QPixmap>
#include "CudaFilters.h" 
#include <QDebug>

void CameraWorker::process() {
    while (running) {
        cv::Mat frame;
        if (cap->isOpened()) {
            (*cap) >> frame; 
            if (!frame.empty()) {
                emit frameReady(frame);
            }
        }
        QThread::msleep(1);
    }
}

RealTimeFiltering::RealTimeFiltering(QWidget* parent) : QMainWindow(parent) {
    auto* central = new QWidget(this);
    auto* layout = new QVBoxLayout(central);
    auto* topBar = new QHBoxLayout();
    btnGroup = new QButtonGroup(this);
    btnGroup->setExclusive(true);

    addButtons(topBar);
    activeFilter = 0;

    videoLabel = new QLabel("Inicjalizacja...");
    videoLabel->setAlignment(Qt::AlignCenter);
    videoLabel->setMinimumSize(640, 480);
    videoLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    videoLabel->setStyleSheet("background-color: black;");

    fpsLabel = new QLabel("FPS: 0");
    fpsLabel->setStyleSheet("color: yellow; font-weight: bold; background-color: rgba(0,0,0,150); padding: 5px;");
    fpsLabel->setFixedHeight(30);

    layout->addLayout(topBar, 0);
    layout->addWidget(videoLabel, 1);

    QHBoxLayout* bottomLayout = new QHBoxLayout();
    bottomLayout->addStretch();
    bottomLayout->addWidget(fpsLabel);
    layout->addLayout(bottomLayout, 0);

    setCentralWidget(central);

    cap.open(0, cv::CAP_DSHOW);
    if (!isCudaAvailable()) qDebug() << "Brak CUDA";

    workerThread = new QThread();
    worker = new CameraWorker(&cap);
    worker->moveToThread(workerThread);

    connect(workerThread, &QThread::started, worker, &CameraWorker::process);
    connect(worker, &CameraWorker::frameReady, this, &RealTimeFiltering::onFrameReceived);
    connect(btnGroup, QOverload<int>::of(&QButtonGroup::idClicked), this, &RealTimeFiltering::filterChanged);

    workerThread->start(); 
}

RealTimeFiltering::~RealTimeFiltering() {
    worker->stop();
    workerThread->quit();
    workerThread->wait(); 
    cap.release();
    freeCudaBuffer();
}

void RealTimeFiltering::filterChanged(int id) {
    activeFilter = id;
}

void RealTimeFiltering::onFrameReceived(cv::Mat frame) {
    tm.start();

    if (activeFilter == 1) {
        initCudaBuffer(frame.cols, frame.rows, frame.channels());
        applyHighPassCuda(frame.data, frame.cols, frame.rows);
    }
    else if (activeFilter == 2) {
        initCudaBuffer(frame.cols, frame.rows, frame.channels());
        applyLowPassCuda(frame.data, frame.cols, frame.rows);
    }
    else if (activeFilter == 3) {
        initCudaBuffer(frame.cols, frame.rows, frame.channels());
        applyThresholdCuda(frame.data, frame.cols, frame.rows, frame.channels(), 150);
    }
    else if (activeFilter == 4) {
        initCudaBuffer(frame.cols, frame.rows, frame.channels());
        applyEdgeDetectionCuda(frame.data, frame.cols, frame.rows);
    }

    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    QImage qimg(frame.data, frame.cols, frame.rows, (int)frame.step, QImage::Format_RGB888);

    videoLabel->setPixmap(QPixmap::fromImage(qimg).scaled(
        videoLabel->size(),
        Qt::KeepAspectRatio,
        Qt::FastTransformation
    ));

    tm.stop();
    frameCounter++;
    if (frameCounter >= 15) {
        fpsLabel->setText(QString("FPS: %1").arg(QString::number(tm.getFPS(), 'f', 1)));
        tm.reset();
        frameCounter = 0;
    }
}

void RealTimeFiltering::addButtons(QHBoxLayout* topBar) {
    QStringList names = { "Oryginał", "Górnoprzepustowy", "Dolnoprzepustowy", "Binaryzacja" };
    for (int i = 0; i < names.size(); ++i) {
        auto* btn = new QPushButton(names[i]);
        btn->setCheckable(true);
        if (i == 0) btn->setChecked(true);
        btnGroup->addButton(btn, i);
        topBar->addWidget(btn);
    }
}