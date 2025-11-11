#include "general_utils.h"

cv::Mat GeneralUtils::qImage2cvMat(const QImage& inImage, bool inCloneImageData)
{
    switch ( inImage.format() )
    {
    // 8-bit, 4 channel
    case QImage::Format_ARGB32:
    case QImage::Format_ARGB32_Premultiplied:
    {
        cv::Mat mat(inImage.height(), inImage.width(),
                    CV_8UC4,
                    const_cast<uchar*>(inImage.bits()),
                    static_cast<size_t>(inImage.bytesPerLine())
                    );

        return (inCloneImageData ? mat.clone() : mat);
    }

        // 8-bit, 3 channel
    case QImage::Format_RGB32:
    {
        if (!inCloneImageData)
        {
            qWarning() << "Conversion requires cloning because we use a temporary QImage";
        }

        QImage swapped = inImage.rgbSwapped();

        return cv::Mat(swapped.height(), swapped.width(),
                       CV_8UC3,
                       const_cast<uchar*>(swapped.bits()),
                       static_cast<size_t>(swapped.bytesPerLine())
                       ).clone();
    }

        // 8-bit, 3 channel
    case QImage::Format_RGB888:
    {
        QImage swapped = inImage.rgbSwapped();

        return cv::Mat(swapped.height(), swapped.width(),
                       CV_8UC3,
                       const_cast<uchar*>(swapped.bits()),
                       static_cast<size_t>(swapped.bytesPerLine())
                       ).clone();
    }

        // 8-bit, 1 channel
    case QImage::Format_Indexed8:
    {
        cv::Mat mat(inImage.height(), inImage.width(),
                    CV_8UC1,
                    const_cast<uchar*>(inImage.bits()),
                    static_cast<size_t>(inImage.bytesPerLine())
                    );

        return (inCloneImageData ? mat.clone() : mat);
    }

    default:
        qWarning() << "QImage format not handled in switch:" << inImage.format();
        break;
    }

    return cv::Mat();
}

QImage GeneralUtils::matToQImage(const cv::Mat& mat)
{
    switch (mat.type())
    {
    case CV_8UC1:
    {
        QImage img(mat.data, mat.cols, mat.rows, mat.cols, QImage::Format_Grayscale8);
        return img;
    }
    case CV_8UC3:
    {
        QImage img(mat.data, mat.cols, mat.rows, mat.cols * 3, QImage::Format_RGB888);
        return img.rgbSwapped();
    }
    case CV_8UC4:
    {
        QImage img(mat.data, mat.cols, mat.rows, mat.cols * 4, QImage::Format_ARGB32);
        return img;
    }
    default:
    {
        return QImage();
    }
    }
}

void GeneralUtils::drawTextOnImage(QImage& image, const QString& text, const QPoint& position, const QFont& font, const QColor& color)
{
    QPainter painter(&image);
    painter.setFont(font);
    painter.setPen(color);
    painter.drawText(position, text);
    painter.end();
}

float GeneralUtils::calculateMean(QVector<float>& values)
{
    float sum = std::accumulate(values.begin(), values.end(), 0.0f);
    return sum / values.size();
}

float GeneralUtils::calculateStandardDeviation(QVector<float>& values, float mean)
{
    float sum = 0.0f;
    for (float value : values) {
        sum += std::pow(value - mean, 2);
    }
    return std::sqrt(sum / values.size());
}

bool GeneralUtils::isWithinRange(float currentValue, QVector<float>& historicalValues, float numStandardDeviations)
{
    float mean = calculateMean(historicalValues);
    float standardDeviation = calculateStandardDeviation(historicalValues, mean);

    // 当前值与平均值之间的差异在一定的标准差范围内
    bool isReasonable = (std::fabs(currentValue - mean) <= numStandardDeviations * standardDeviation);
    return isReasonable;
}

QString GeneralUtils::generateDate()
{
    return QDateTime::currentDateTime().toString("yyyyMMddHHmmss");
}

int GeneralUtils::saveInferenceImage(QString& saveRootPath, QString& viewName, QString& patientName, std::vector<cv::Mat>& img)
{
    QString subFolderName = generateDate();
    QString folderPath = saveRootPath + "/" + viewName;
    QDir().mkpath(folderPath);

    QString saveFolderPath = folderPath + "/" + subFolderName + "_cycle";
    QDir().mkpath(saveFolderPath);

    for (int i = 0; i < img.size(); ++i)
    {
        QString filePath = saveFolderPath + "/" + QString("image_%1.jpg").arg(i);
        cv::imwrite(filePath.toStdString(), img[i]);
    }

    return 1;
}

std::string GeneralUtils::toLowerCase(std::string& strInput)
{
    std::string result = strInput;
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
        return std::tolower(c);
        });
    return result;
}

std::string GeneralUtils::toUpperCase(std::string& strInput)
{
    std::string result = strInput;
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
        return std::toupper(c);
        });
    return result;
}

bool GeneralUtils::fileExists(std::string& strPath)
{
    DWORD fileAttr = GetFileAttributesA(strPath.c_str());
    return (fileAttr != INVALID_FILE_ATTRIBUTES && !(fileAttr & FILE_ATTRIBUTE_DIRECTORY));
}

void GeneralUtils::concatVectors(std::vector<cv::Mat>& vecImages1, std::vector<cv::Mat>& vecImages2, std::vector<cv::Mat>& vecResult)
{
    vecResult = vecImages1;  // 先复制第一个向量
    vecResult.insert(vecResult.end(), vecImages2.begin(), vecImages2.end()); // 拼接第二个向量
}

QString GeneralUtils::formatVector(const QVector<int>& vector)
{
    // 创建一个空的QString来存储结果
    QString result;
    if (vector.isEmpty())
    {
        result = QString("Empty Vector");
        return result;
    }

    // 遍历QVector并使用arg格式化字符串
    for (int i = 0; i < vector.size(); ++i) {
        // 将当前元素格式化并添加到结果字符串中
        result += QString::number(vector[i]); // 将整数转换为QString并添加
        if (i < vector.size() - 1) {
            result += ", "; // 如果不是最后一个元素，添加分隔符
        }
    }

    return result;
}
