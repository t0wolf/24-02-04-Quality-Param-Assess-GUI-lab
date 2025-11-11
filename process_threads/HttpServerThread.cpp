#include "HttpServerThread.h"


HttpServer::HttpServer(QObject* parent, ConfigParse* config)
    : QTcpServer(parent)
    , m_ipAddress(QString("127.0.0.1"))
    , m_port(1080)
{
    std::string ipAddress = "";

    if (config != nullptr)
    {
        bool ret = config->getSpecifiedNode(std::string("IP_ADDRESS"), ipAddress);
        if (ret)
            m_ipAddress = QString::fromStdString(ipAddress);

        ret = config->getSpecifiedNode(std::string("PORT"), ipAddress);
        if (ret)
            m_port = std::atoi(ipAddress.c_str());
    }

    if (!listen(QHostAddress(m_ipAddress), m_port)) {
        qDebug() << "Failed to start server:" << errorString();
        return;
    }
    qDebug() << "[I] Server started on " << m_ipAddress << ":" << m_port;
}

void HttpServer::incomingConnection(qintptr socketDescriptor)
{
    QTcpSocket* socket = new QTcpSocket(this);
    //socket->setSocketDescriptor(socketDescriptor);

    //connect(socket, &QTcpSocket::readyRead, this, &HttpServer::readData);
    //connect(socket, &QTcpSocket::disconnected, socket, &QTcpSocket::deleteLater);
    if (socket->setSocketDescriptor(socketDescriptor)) {
        connect(socket, &QTcpSocket::readyRead, this, [this, socket]() {
            handleRequest(socket);
            });
        connect(socket, &QTcpSocket::disconnected, socket, &QTcpSocket::deleteLater);
    }
    else {
        delete socket;
    }
}

QString HttpServer::extractStinstid(const QString& request)
{
    QStringList lines = request.split('\n');
    QStringList parts = lines.first().split(' ');
    QString url = parts[1]; // 假设第一行是 GET /getStudyInstID?stinstid=... HTTP/1.1
    QUrl parsedUrl(url);
    QString query = parsedUrl.query();
    QStringList params = query.split('&');
    foreach(const QString & param, lines) {
        QStringList keyValue = param.split('=');
        if (keyValue.first() == "stinstid") {
            return keyValue.last();
        }
    }
    return QString();
}

QString HttpServer::parseParameters(const QString& request, const QString queryParamName)
{
    QStringList lines = request.split('\n');
    for (auto& line : lines)
    {
        if (line.contains(queryParamName) && line.contains("="))
        {
            if (line.contains(" "))  // 假设是这种标准的格式: GET /getstudyInstID?stinstid=1.2.840.10008.192.15.3233 HTTP/1.1
            {
                if (line.size() < 1)
                    continue;
                QString url = line.split(" ")[1];
                QUrl parsedUrl(url);
                line = parsedUrl.query();
            }

            QStringList keyValue = line.split("=");
            if (keyValue.first() == queryParamName)
                return keyValue.last();
        }
    }

    return QString();
}

void HttpServer::handleStinstid(QTcpSocket* socket, const QString& request)
{
    if (!request.isEmpty())
    {
        //QStringList lines = request.split('\n');
        //QString query = lines[8];
        //QStringList keyValue = query.split('=');
        //QString stinstid = keyValue.last();
        ////QString stinstid = extractStinstid(request);

        QString stinstid = parseParameters(request, "stinstid");

        if (stinstid.isEmpty())
        {
            QByteArray response = "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/plain\r\n"
                "Connection: close\r\n\r\n"
                "Study Instance ID parsing failed.\n";
            return;
        }

        emit sigInstIDAvailable(stinstid);

        // 构造响应
        QByteArray response = "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/plain\r\n"
            "Content-Length: " + QString::number(stinstid.length()).toUtf8() + "\r\n"
            "Connection: close\r\n\r\n"
            + stinstid.toUtf8();

        socket->write(response);
        socket->disconnectFromHost();
        socket->waitForDisconnected();
    }
}

QString HttpServer::handleGetAIMeterage(QTcpSocket* socket, const QString& request)
{
    if (!request.isEmpty())
    {
        QString stinstid = parseParameters(request, "stinstid");

        return stinstid;
    }

    return QString();
}

void HttpServer::handleRequest(QTcpSocket* socket)
{
    if (!socket->canReadLine()) return;

    QByteArray requestLine = socket->readAll();
    QStringList tokens = QString(requestLine).split(' ');
    //if (tokens.size() < 2 || tokens[0] != "GET") {
    //    sendErrorResponse(socket, 400, "Bad Request");
    //    return;
    //}

    QString requestPath = tokens[1];
    QString filePath;

    if (requestPath.contains("/getstudyInstID"))
    {
        handleStinstid(socket, requestLine);
        return;
    }

    else if (requestPath.contains("/getAIMeterage"))
    {
        QString strInstID = handleGetAIMeterage(socket, requestLine);
        if (strInstID.isEmpty())
        {
            sendErrorResponse(socket, 404, "Instance ID Not Found");
            return;
        }
        filePath = m_directory + "/html/" + strInstID + ".html";  // Change to your specific HTML file
    }

    else
    {
        filePath = m_directory + requestPath;
    }

    QFile file(filePath);
    if (!file.exists() || !file.open(QIODevice::ReadOnly)) {
        sendErrorResponse(socket, 404, "HTML or Image Not Found");
        return;
    }

    QByteArray content = file.readAll();
    file.close();

    QMimeDatabase mimeDb;
    QMimeType mimeType = mimeDb.mimeTypeForFile(filePath);
    QByteArray mime = mimeType.name().toUtf8();

    QByteArray response;
    response.append("HTTP/1.1 200 OK\r\n");
    response.append("Content-Type: " + mime + "\r\n");
    response.append("Content-Length: " + QByteArray::number(content.size()) + "\r\n");
    response.append("Connection: close\r\n");
    response.append("\r\n");
    response.append(content);

    socket->write(response);
    socket->disconnectFromHost();
}

void HttpServer::readData()
{
    QTcpSocket* socket = qobject_cast<QTcpSocket*>(sender());
    if (!socket) return;

    if (socket->canReadLine()) {
        QByteArray data = socket->readAll();
        QString request(data);
        QString stinstid = extractStinstid(request);

        emit sigInstIDAvailable(stinstid);

        // 构造响应
        QByteArray response = "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/plain\r\n"
            "Content-Length: " + QString::number(stinstid.length()).toUtf8() + "\r\n"
            "Connection: close\r\n\r\n"
            + stinstid.toUtf8();

        socket->write(response);
        socket->disconnectFromHost();
        socket->waitForDisconnected();
    }
}