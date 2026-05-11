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

    for (int i = 0; i < 4; ++i) {
        auto* btn = new QPushButton(QString("Filtr %1").arg(i + 1));
        btn->setCheckable(true);
        btnGroup->addButton(btn, i);
        topBar->addWidget(btn);
    }
    btnGroup->button(0)->setChecked(true);
    activeFilter = 0;

    videoLabel = new QLabel("Inicjalizacja...");
    videoLabel->setAlignment(Qt::AlignCenter);
    videoLabel->setMinimumSize(640, 480);

    layout->addLayout(topBar);
    layout->addWidget(videoLabel);
    setCentralWidget(central);

    if (!isCudaAvailable()) {
        qDebug() << "!!! BŁĄD: Brak karty NVIDIA lub sterowników CUDA! !!!";
    }
    else {
        qDebug() << "CUDA jest gotowe do pracy.";
    }

    // --- OTWIERANIE KAMERY (Tryb DirectShow) ---
    // cv::CAP_DSHOW omija błędy AMD/MediaFoundation widoczne w Twoich logach
    cap.open(0, cv::CAP_DSHOW);

    if (!cap.isOpened()) {
        qDebug() << "BŁĄD: Nie można otworzyć kamery przez DirectShow!";
    }

    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &RealTimeFiltering::updateFrame);
    connect(btnGroup, QOverload<int>::of(&QButtonGroup::idClicked), this, &RealTimeFiltering::filterChanged);

    timer->start(16);
}

RealTimeFiltering::~RealTimeFiltering() {
    timer->stop();
    cap.release();
    freeCudaBuffer();
}

void RealTimeFiltering::filterChanged(int id) {
    activeFilter = id;
}

void RealTimeFiltering::updateFrame() {
    if (!cap.isOpened()) return;

    cv::Mat frame;
    cap >> frame;

    if (frame.empty() || frame.cols <= 0 || frame.rows <= 0) return;

    // Wymuszamy 3 kanały BGR
    if (frame.channels() != 3) {
        cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
    }

    // --- FILTRY ---
    if (activeFilter == 3) { // Twoja CUDA
        initCudaBuffer(frame.cols, frame.rows, frame.channels());
        applyThresholdCuda(frame.data, frame.cols, frame.rows, frame.channels(), 150);
    }
    else if (activeFilter == 1) {
        cv::bitwise_not(frame, frame);
    }
    else if (activeFilter == 2) {
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
    }

    // --- BEZPIECZNE WYŚWIETLANIE ---
    // RGB conversion
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

    // Tworzymy obraz Qt
    QImage qimg(rgb.data, rgb.cols, rgb.rows, (int)rgb.step, QImage::Format_RGB888);

    // Robimy kopię, żeby Qt nie czytało z pamięci OpenCV po wyjściu z tej funkcji
    QPixmap pix = QPixmap::fromImage(qimg.copy());

    if (!pix.isNull()) {
        videoLabel->setPixmap(pix.scaled(videoLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
}