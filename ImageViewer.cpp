#include "ImageViewer.h"
#include <QImage>
#include <QPixmap>
#include <QKeyEvent>

ImageViewer::ImageViewer(QWidget* parent)
    : QWidget(parent), m_currentIndex(0)
{
    setWindowTitle("Image Viewer");
    resize(800, 600);

    m_labelImage = new QLabel(this);
    m_labelImage->setAlignment(Qt::AlignCenter);
    m_labelImage->setMinimumSize(400, 300);

    m_btnPrev = new QPushButton("Previous", this);
    m_btnNext = new QPushButton("Next", this);

    connect(m_btnPrev, &QPushButton::clicked, this, &ImageViewer::showPrevImage);
    connect(m_btnNext, &QPushButton::clicked, this, &ImageViewer::showNextImage);

    QHBoxLayout* btnLayout = new QHBoxLayout();
    btnLayout->addWidget(m_btnPrev);
    btnLayout->addWidget(m_btnNext);

    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    mainLayout->addWidget(m_labelImage, 1);
    mainLayout->addLayout(btnLayout);

    setLayout(mainLayout);
}

ImageViewer::~ImageViewer()
{
}

void ImageViewer::setImageBuffer(const std::vector<cv::Mat>& images)
{
    if (images.empty()) return;
    m_images = images;
    m_currentIndex = 0;
    updateImageDisplay();
}

void ImageViewer::updateImageDisplay()
{
    if (m_images.empty()) return;
    cv::Mat img = m_images[m_currentIndex];
    QImage qimg = cvMatToQImage(img);
    m_labelImage->setPixmap(QPixmap::fromImage(qimg).scaled(m_labelImage->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void ImageViewer::showPrevImage()
{
    if (m_images.empty()) return;
    m_currentIndex = (m_currentIndex - 1 + m_images.size()) % m_images.size();
    updateImageDisplay();
}

void ImageViewer::showNextImage()
{
    if (m_images.empty()) return;
    m_currentIndex = (m_currentIndex + 1) % m_images.size();
    updateImageDisplay();
}

QImage ImageViewer::cvMatToQImage(const cv::Mat& mat)
{
    if (mat.type() == CV_8UC3)
    {
        return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_BGR888).copy();
    }
    else if (mat.type() == CV_8UC1)
    {
        return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8).copy();
    }
    return QImage();
}

void ImageViewer::keyPressEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_Left)
    {
        showPrevImage();
    }
    else if (event->key() == Qt::Key_Right)
    {
        showNextImage();
    }
    else
    {
        QWidget::keyPressEvent(event);
    }
}
