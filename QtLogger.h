#pragma once

#include <QObject>
#include <QFile>
#include <QTextStream>
#include <QDateTime>
#include <QDebug>
#include <QMutex>
#include <QDir>


class QtLogger : public QObject 
{
    Q_OBJECT

public:
    static QtLogger& instance();

    void logMessage(const QString& message);

    bool startLoggingToFile(const QString& filePath);

    void stopLoggingToFile();

private:
    QtLogger(QObject* parent = nullptr) : QObject(parent) {}
    ~QtLogger() {
        stopLoggingToFile();
    }

    Q_DISABLE_COPY(QtLogger) // 禁用拷贝构造和赋值

    QMutex logMutex;
    QFile logFile;
};