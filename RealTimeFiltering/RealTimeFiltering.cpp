#include "RealTimeFiltering.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QImage>
#include <QPixmap>
#include "CudaFilters.h" 
#include <QDebug>



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

    QHBoxLayout* bottomLayout = new QHBoxLayout();
    fpsLabel = new QLabel("FPS: 0");
    // Stylizacja licznika: biały tekst na lekko przezroczystym tle
    fpsLabel->setStyleSheet("color: yellow; font-weight: bold; background-color: rgba(0,0,0,150); padding: 5px;");

    bottomLayout->addStretch(); // Popycha label do prawej strony
    bottomLayout->addWidget(fpsLabel);
    
    layout->addLayout(topBar, 0);      
    layout->addWidget(videoLabel, 1);  
    layout->addLayout(bottomLayout, 0);

    setCentralWidget(central);

    if (!isCudaAvailable()) {
        qDebug() << "!!! BŁĄD: Brak karty NVIDIA lub sterowników CUDA! !!!";
    }
    else {
        qDebug() << "CUDA jest gotowe do pracy.";
    }

    // --- OTWIERANIE KAMERY (Tryb DirectShow) ---
    cap.open(0, cv::CAP_DSHOW);

    if (!cap.isOpened()) {
        qDebug() << "BŁĄD: Nie można otworzyć kamery przez DirectShow!";
    }

    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &RealTimeFiltering::updateFrame);
    connect(btnGroup, QOverload<int>::of(&QButtonGroup::idClicked), this, &RealTimeFiltering::filterChanged);

    timer->start(0);
}

RealTimeFiltering::~RealTimeFiltering() {
    timer->stop();
    cap.release();
    freeCudaBuffer();
}

void RealTimeFiltering::filterChanged(int id) {
    activeFilter = id;
}

void RealTimeFiltering::addButtons(QHBoxLayout* topBar) {
    QStringList names = {
        "Oryginał",
        "Górnoprzepustowy",
        "Dolnoprzepustowy",
        "Binaryzacja",
        "Krawędzie"
    };

    for (int i = 0; i < names.size(); ++i) {
        auto* btn = new QPushButton(names[i]);
        btn->setCheckable(true);

        if (i == 0) btn->setChecked(true);

        btnGroup->addButton(btn, i);
        topBar->addWidget(btn);
    }
}

void RealTimeFiltering::updateFrame() {
    if (!cap.isOpened()) return;

    tm.start();

    cv::Mat frame;
    cap >> frame;

    if (frame.empty()) return;

    if (activeFilter == 0) {
        // Original video
    }
    else if (activeFilter == 1) {
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
        double currentFps = tm.getFPS();
        fpsLabel->setText(QString("FPS: %1").arg(QString::number(currentFps, 'f', 1)));

        tm.reset();    
        frameCounter = 0;
    }

}