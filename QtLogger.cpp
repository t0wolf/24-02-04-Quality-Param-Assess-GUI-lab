#include "QtLogger.h"


QtLogger& QtLogger::instance()
{
    static QtLogger instance;  // 单例模式
    return instance;
}

void QtLogger::logMessage(const QString& message)
{
    QMutexLocker locker(&logMutex);

    QString timeStamp = QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss");
    QString logEntry = QString("[%1] %2").arg(timeStamp, message);

    qDebug() << logEntry; // 输出到控制台

    if (logFile.isOpen()) {
        QTextStream out(&logFile);
        out << logEntry << "\n";
    }
}

bool QtLogger::startLoggingToFile(const QString& filePath)
{
    QMutexLocker locker(&logMutex);

    QFileInfo fileInfo(filePath);
    QDir dir = fileInfo.absoluteDir();

    // 如果目录不存在，则创建目录
    if (!dir.exists()) {
        if (!dir.mkpath(".")) {
            qDebug() << "Failed to create directory for log file:" << dir.absolutePath();
            return false;
        }
    }

    logFile.setFileName(filePath);

    if (!logFile.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text)) {
        qDebug() << "Failed to open log file:" << filePath;
        return false;
    }
    return true;
}

void QtLogger::stopLoggingToFile()
{
    QMutexLocker locker(&logMutex);
    if (logFile.isOpen()) {
        logFile.close();
    }
}
