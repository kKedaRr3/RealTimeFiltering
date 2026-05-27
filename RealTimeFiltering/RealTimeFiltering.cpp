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
    btnGroup->setExclusive(false);

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
    if (id == 0) { 
        if (filterButtons[0]->isChecked()) {
            for (int i = 1; i < filterButtons.size(); ++i) {
                filterButtons[i]->setChecked(false);
            }
            filterPipeline.clear();
        }
    }
    else { 
        if (filterButtons[id]->isChecked()) {
            filterButtons[0]->setChecked(false);
            if (!filterPipeline.contains(id)) {
                filterPipeline.append(id);
            }
        }
        else {
            filterPipeline.removeAll(id);
        }
    }
}

void RealTimeFiltering::onFrameReceived(cv::Mat frame) {
    tm.start();

    if (frame.empty()) return;

    if (!filterButtons[0]->isChecked() && !filterPipeline.isEmpty()) {
        initCudaBuffer(frame.cols, frame.rows, frame.channels());

        for (int filterId : filterPipeline) {
            if (filterId == 1) {
                applyHighPassCuda(frame.data, frame.cols, frame.rows);
            }
            else if (filterId == 2) {
                applyLowPassCuda(frame.data, frame.cols, frame.rows);
            }
            else if (filterId == 3) {
                applyThresholdCuda(frame.data, frame.cols, frame.rows, frame.channels(), 150);
            }
            else if (filterId == 4) {
                applyEdgeDetectionCuda(frame.data, frame.cols, frame.rows);
            }
        }
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
    QStringList names = { "Oryginał", "Górnoprzepustowy", "Dolnoprzepustowy", "Binaryzacja", "Detekcja krawędzi" };
    for (int i = 0; i < names.size(); ++i) {
        auto* btn = new QPushButton(names[i]);
        btn->setCheckable(true);
        if (i == 0) btn->setChecked(true);
        btnGroup->addButton(btn, i);
        topBar->addWidget(btn);
        filterButtons.append(btn);
    }
}