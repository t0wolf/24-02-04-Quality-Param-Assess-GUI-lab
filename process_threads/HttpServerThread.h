#pragma once

#include <QThread>
#include <QTcpServer>
#include <QTcpSocket>
#include <QThreadPool>
#include <QUrlQuery>
#include <QJsonDocument>
#include <QJsonObject>
#include <QDebug>
#include <QDir>
#include <QString>
#include <QFile>
#include <QMimeDatabase>
#include "config_parse.h"


class HttpServer : public QTcpServer
{
    Q_OBJECT

public:
    HttpServer(QObject* parent = nullptr, ConfigParse* config = nullptr);

signals:
    void sigInstIDAvailable(const QString& instID);

protected:
    void incomingConnection(qintptr socketDescriptor) override;

    QString extractStinstid(const QString& request);

    QString parseParameters(const QString& request, const QString queryParamName);

    void handleStinstid(QTcpSocket* socket, const QString& request);

    QString handleGetAIMeterage(QTcpSocket* socket, const QString& request);

private slots:
    void readData();

    void handleRequest(QTcpSocket* socket);

    void sendErrorResponse(QTcpSocket* socket, int statusCode, const QString& statusText) {
        QByteArray response;
        response.append("HTTP/1.1 " + QByteArray::number(statusCode) + " " + statusText.toUtf8() + "\r\n");
        response.append("Content-Type: text/plain\r\n");
        response.append("Connection: close\r\n");
        response.append("\r\n");
        response.append(statusText.toUtf8());

        socket->write(response);
        socket->disconnectFromHost();
    }

    //void readData(QTcpServer* socket);

private:
    QString m_ipAddress;
    int m_port;

    //QString m_directory = "D:/Data/Param-Assess-Data/html";
    QString m_directory = QDir::currentPath();
};

