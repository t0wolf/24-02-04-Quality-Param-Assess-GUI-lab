#include "HttpJSONReceiver.h"


HttpJSONReceiver::~HttpJSONReceiver()
{
    this->requestInterruption();
    this->quit();
    this->wait();
}

void HttpJSONReceiver::run()
{
    QNetworkAccessManager manager;
    while (!isInterruptionRequested()) 
    {
        QNetworkRequest request(m_url);
        QNetworkReply* reply = manager.get(request);

        // 连接信号，等待请求完成
        connect(reply, &QNetworkReply::finished, this, &HttpJSONReceiver::handleReplyFinished);

        // 进入事件循环，等待请求完成
        exec();
    }
}

void HttpJSONReceiver::handleReplyFinished() 
{
    QNetworkReply* reply = qobject_cast<QNetworkReply*>(sender());

    if (reply && reply->error() == QNetworkReply::NoError) 
    {
        QByteArray data = reply->readAll();
        QJsonParseError parseError;
        QJsonDocument jsonDoc = QJsonDocument::fromJson(data, &parseError);

        if (parseError.error == QJsonParseError::NoError && jsonDoc.isObject()) 
        {
            QJsonObject jsonObject = jsonDoc.object();
            if (jsonObject.contains(m_checkIDKey) && jsonObject[m_checkIDKey].isString())
            {
                QString checkId = jsonObject[m_checkIDKey].toString();
                emit sigCheckIDReceived(checkId);
            }
            else 
            {
                qDebug() << "Key 'check_id' not found or not a string.";
            }
        }

        else 
        {
            qDebug() << "JSON Parse Error:" << parseError.errorString();
        }
    }

    else
    {
        qDebug() << "Error:" << reply->errorString();
    }

    reply->deleteLater();
}