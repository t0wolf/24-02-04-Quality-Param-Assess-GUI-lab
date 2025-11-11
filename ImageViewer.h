#pragma once
#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <opencv2/opencv.hpp>

class ImageViewer : public QWidget
{
    Q_OBJECT
public:
    explicit ImageViewer(QWidget* parent = nullptr);
    ~ImageViewer();

    // 设置图片数组（一次性全部更新）
    void setImageBuffer(const std::vector<cv::Mat>& images);

protected:
    void keyPressEvent(QKeyEvent* event) override; // 支持键盘左右切换

private slots:
    void showPrevImage();
    void showNextImage();

private:
    QLabel* m_labelImage;
    QPushButton* m_btnPrev;
    QPushButton* m_btnNext;
    std::vector<cv::Mat> m_images;
    int m_currentIndex;

    void updateImageDisplay();
    QImage cvMatToQImage(const cv::Mat& mat);
};
