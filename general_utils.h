#ifndef GENERAL_UTILS_H
#define GENERAL_UTILS_H

#include <opencv2/opencv.hpp>
#include <QImage>
#include <QPainter>
#include <QDebug>
#include <QDateTime>
#include <QFile>
#include <QDir>
#include <windows.h> // Windows API
#include <stdexcept> // 用于异常处理

namespace GeneralUtils
{
    cv::Mat qImage2cvMat(const QImage& inImage, bool inCloneImageData = true);

    QImage matToQImage(const cv::Mat &mat);

    void drawTextOnImage(QImage& image, const QString& text, const QPoint& position, const QFont& font = QFont(), const QColor& color = Qt::black);

    float calculateMean(QVector<float>& values);

    float calculateStandardDeviation(QVector<float>& values, float mean);

    bool isWithinRange(float currentValue, QVector<float>& historicalValues, float numStandardDeviations);

    QString generateDate();

    int saveInferenceImage(QString& saveRootPath, QString& viewName, QString& patientName, std::vector<cv::Mat>& img);

    std::string toLowerCase(std::string& strInput);

    std::string toUpperCase(std::string& strInput);

    bool fileExists(std::string& strPath);

    void concatVectors(std::vector<cv::Mat>& vecImages1, std::vector<cv::Mat>& vecImages2, std::vector<cv::Mat>& vecResult);

    QString formatVector(const QVector<int>& vector);
}

#endif // GENERAL_UTILS_H
