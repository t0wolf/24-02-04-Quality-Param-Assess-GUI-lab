#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QApplication>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , m_currPatientName(QString(""))
    , m_currPatientID(QString(""))
    , m_currCheckID(QString(""))
{
    m_configFile = new ConfigParse(std::string("D:/Resources/20240221/configs.yaml"));
    QtLogger::instance().logMessage("[I] Configuration file loaded.");

    m_settingsDialog = new SettingsDialog(this, m_configFile);
    m_progressSuperThread = new ProgressSuperThread(this);
    m_paramBrowserWidget = new ParamBrowserWidget(this);
    m_paramBrowserWidget->hide();

    m_patientManagerWindow = new PatientManagerWindow(this);

    m_httpJSONReceiveThread = new HttpServer(this, m_configFile);

    m_qualityDisplayWidget = new QualityDisplayWidget(this, m_progressSuperThread, m_configFile);
    QtLogger::instance().logMessage("[I] Quality Control model loaded.");

    //setupPatientManager();
    m_paramTransmitter = new ParamTransmitter(m_configFile, m_patientManagerWindow, this);
    m_paramDisplayWidget = new ParamDisplayWidget(this, m_progressSuperThread, m_paramBrowserWidget, m_paramTransmitter);
    QtLogger::instance().logMessage("[I] Parameter Assessing model loaded.");
    m_displayStreamWidget = new DisplayStreamWidget(m_qualityDisplayWidget, m_paramDisplayWidget, m_progressSuperThread, m_configFile, this);
    QtLogger::instance().logMessage("[I] DisplayStreamWidget initialized.");

    ui->setupUi(this);
    ui->horizontalLayout_3->layout()->addWidget(m_displayStreamWidget);
    ui->rightSideWidget->layout()->addWidget(m_qualityDisplayWidget);
    ui->rightSideWidget->layout()->addWidget(m_paramDisplayWidget);
    ui->rightSideWidget->layout()->setSpacing(0);
    ui->rightSideWidget->layout()->setContentsMargins(1, 1, 1, 1);
    setLayout(ui->rightSideWidget->layout());

    connect(this->m_displayStreamWidget->m_infoExtractThread, &InfoExtractThread::sigPatientInfo, this, &MainWindow::setPatientInfo);
    connect(this->m_displayStreamWidget->m_modelsInferThread, &ModelsInferenceThread::sigReinitailizeLabel, this->m_qualityDisplayWidget, &QualityDisplayWidget::setLabelInitialize);
    //connect(this->m_httpJSONReceiveThread, &HttpServer::sigInstIDAvailable, this, &MainWindow::setParamSaverInitByCheckID);
    connect(this->m_httpJSONReceiveThread, &HttpServer::sigInstIDAvailable, this, &MainWindow::setParamTransmitterInitByCheckID);
    connect(this->m_httpJSONReceiveThread, &HttpServer::sigInstIDAvailable, m_displayStreamWidget->m_paramAssessHtmlShower, &ParamAssessHTMLShower::slotReceiveStudyInstID);
    connect(this, &MainWindow::sigPatientName, m_displayStreamWidget->m_paramAssessHtmlShower, &ParamAssessHTMLShower::slotReceivePatientName);
    connect(this, &MainWindow::sigPatientName, this->m_displayStreamWidget->m_modelsInferThread->m_paramAssessThread, &ParamAssessThread::setPatientName);
    connect(this->m_displayStreamWidget->m_modelsInferThread, &ModelsInferenceThread::sigViewNameQualityScoreToTransmitter, this->m_paramTransmitter, &ParamTransmitter::slotHandleQualityScore);

    if (!loadCredentials("user_config.txt")) {
        QMessageBox::warning(this, QStringLiteral("警告"), QStringLiteral("无法读取配置文件，修改密码功能可能异常"));
        m_password.clear();
    }

    setupMenu();
    //m_httpJSONReceiveThread->start();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::setupMenu()
{
    QMenu* configMenu = nullptr;
    foreach(QAction * act, menuBar()->actions()) {
        if (act->text() == QStringLiteral("配置")) {
            configMenu = act->menu();
            break;
        }
    }

    if (configMenu) {
        QAction* changePwdAction = new QAction(QStringLiteral("修改密码"), this);
        configMenu->addAction(changePwdAction);
        connect(changePwdAction, &QAction::triggered, this, &MainWindow::onChangePassword);
    }
}

bool MainWindow::loadCredentials(const QString& filePath)
{
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return false;

    QStringList lines;
    QTextStream in(&file);
    while (!in.atEnd())
        lines.append(in.readLine());
    file.close();

    int credIndex = -1;
    for (int i = 0; i < lines.size(); ++i) {
        if (lines[i].trimmed() == "credentials:") {
            credIndex = i;
            break;
        }
    }
    if (credIndex == -1)
        return false;

    QString username, password;
    for (int i = credIndex + 1; i < lines.size(); ++i) {
        QString line = lines[i].trimmed();
        if (line.startsWith("username:")) {
            int first = line.indexOf("\"");
            int last = line.lastIndexOf("\"");
            if (first != -1 && last != -1 && last > first)
                username = line.mid(first + 1, last - first - 1);
        }
        if (line.startsWith("password:")) {
            int first = line.indexOf("\"");
            int last = line.lastIndexOf("\"");
            if (first != -1 && last != -1 && last > first)
                password = line.mid(first + 1, last - first - 1);
        }
    }
    if (username.isEmpty() || password.isEmpty())
        return false;

    m_username = username;
    m_password = password;
    return true;
}

bool MainWindow::saveCredentials(const QString& filePath, const QString& username, const QString& password)
{
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return false;

    QStringList lines;
    QTextStream in(&file);
    while (!in.atEnd())
        lines.append(in.readLine());
    file.close();

    int credIndex = -1;
    for (int i = 0; i < lines.size(); ++i) {
        if (lines[i].trimmed() == "credentials:") {
            credIndex = i;
            break;
        }
    }

    QString usernameLine = QString("  username: \"%1\"").arg(username);
    QString passwordLine = QString("  password: \"%1\"").arg(md5Hash(password)); // 加密后写入

    if (credIndex == -1) {
        lines.append("");
        lines.append("credentials:");
        lines.append(usernameLine);
        lines.append(passwordLine);
    }
    else {
        int uIndex = -1, pIndex = -1;
        for (int i = credIndex + 1; i < lines.size(); ++i) {
            QString trimmed = lines[i].trimmed();
            if (trimmed.startsWith("username:"))
                uIndex = i;
            else if (trimmed.startsWith("password:"))
                pIndex = i;
            if (uIndex != -1 && pIndex != -1)
                break;
        }
        if (uIndex != -1)
            lines[uIndex] = usernameLine;
        else
            lines.insert(credIndex + 1, usernameLine);
        if (pIndex != -1)
            lines[pIndex] = passwordLine;
        else {
            if (uIndex != -1)
                lines.insert(uIndex + 1, passwordLine);
            else
                lines.insert(credIndex + 2, passwordLine);
        }
    }

    if (!file.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Truncate))
        return false;

    QTextStream out(&file);
    for (const QString& line : lines)
        out << line << "\n";
    file.close();
    return true;
}

void MainWindow::onChangePassword()
{
    bool ok;
    QString oldPwd = QInputDialog::getText(this, QStringLiteral("验证旧密码"), QStringLiteral("请输入旧密码:"), QLineEdit::Password, "", &ok);
    if (!ok) return;

    if (md5Hash(oldPwd) != m_password) {
        QMessageBox::warning(this, QStringLiteral("错误"), QStringLiteral("旧密码错误！"));
        return;
    }

    QString newPwd = QInputDialog::getText(this, QStringLiteral("修改密码"), QStringLiteral("请输入新密码:"), QLineEdit::Password, "", &ok);
    if (!ok) return;

    if (newPwd.isEmpty()) {
        QMessageBox::warning(this, QStringLiteral("错误"), QStringLiteral("新密码不能为空！"));
        return;
    }

    if (saveCredentials("user_config.txt", m_username, newPwd)) {
        m_password = md5Hash(newPwd);
        QMessageBox::information(this, QStringLiteral("成功"), QStringLiteral("密码修改成功！"));
    }
    else {
        QMessageBox::critical(this, QStringLiteral("失败"), QStringLiteral("保存密码失败！"));
    }
}

void MainWindow::setupPatientManager() {
    // 创建 PatientManagerWindow 实例
    m_patientManagerWindow = new PatientManagerWindow(this);

    // 将 PatientManagerWindow 添加到 horizontalLayout_3 中
    //ui->horizontalLayout_3->addWidget(m_patientManagerWindow);

    // 设置宽度（可根据需求调整）
    //m_patientManagerWindow->setFixedWidth(250); // 设置固定宽度，防止挤压右侧 UI 
}

void MainWindow::setParamSaverInitByCheckID(const QString& checkID)
{
    if (checkID != m_currCheckID)
    {
        m_currCheckID = checkID;
        m_paramDisplayWidget->initParamSaver(checkID);
    }
}

void MainWindow::setParamTransmitterInitByCheckID(const QString& checkID)
{
    if (checkID != m_currCheckID)
    {
        m_currCheckID = checkID;
        m_paramTransmitter->initParamTransmitter(checkID);
    }
}

void MainWindow::on_clearAllButton_clicked()
{
    setQualityParamWidgetClear();
}

void MainWindow::on_browserParamButton_clicked()
{
    m_paramBrowserWidget->show();
}

void MainWindow::onActionSettingsTriggered()
{
    //m_settingsDialog->setAttribute(Qt::WA_DeleteOnClose);
    m_settingsDialog->show();
    m_settingsDialog->raise();
    m_settingsDialog->activateWindow();
}

void MainWindow::onAboutSettingsTriggered()
{
    m_aboutDialog = new AboutDialog(this);
    m_aboutDialog->setAttribute(Qt::WA_DeleteOnClose);
    m_aboutDialog->show();
    m_aboutDialog->raise();
    m_aboutDialog->activateWindow();
}

void MainWindow::onPatientListButtonClicked()
{
    m_patientManagerWindow->show();
    m_patientManagerWindow->setWindowTitle("病人列表");
    m_patientManagerWindow->exec(); // 以模式对话框的形式显示
}

void MainWindow::setQualityParamWidgetClear()
{
    m_displayStreamWidget->m_modelsInferThread->setModelInferThreadClear();
    m_qualityDisplayWidget->labelReInit();

    std::string rootPath = "";
    m_configFile->getSpecifiedNode("SYSTEM_PRED_CSV_SAVE_PATH", rootPath);
    if (rootPath.empty())
        rootPath = "D:/Data/";
    
    m_paramDisplayWidget->saveParamToCsv(QString::fromStdString(rootPath), m_currPatientName);
    m_paramDisplayWidget->clearParamValues();
    m_paramDisplayWidget->paramLabelsReInit();
    m_paramBrowserWidget->clearWidget();
}

void MainWindow::setPatientInfo(QString patientName, QString patientID)
{
    if (!patientName.isEmpty())
    {
        QString modifiedPatientName = removePunctuationAndSpaces(patientName);
        ui->patientNameLabel->setText(modifiedPatientName);
        if (removePunctuationAndSpaces(m_currPatientName) != modifiedPatientName)
        {
            // update current patient's name
            setQualityParamWidgetClear();
            m_patientManagerWindow->addPatient(patientName, patientID);
            m_patientManagerWindow->switchPatient(patientID);
            m_currPatientName = patientName;
            m_currPatientID = patientID;
            emit sigPatientName(modifiedPatientName);
        }
    }

    if (!patientID.isEmpty())
        ui->patientIDLabel->setText(patientID);
}

void logException(const QString& message)
{
    qCritical() << "发生异常：" << message;
    // 将日志信息写入到文件
    QFile file("./error.log");
    if (file.open(QIODevice::WriteOnly | QIODevice::Append)) {
        QTextStream stream(&file);
        stream << QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss") << " - " << message << '\n';
        file.close();
    }
}

void customMessageHandler(QtMsgType type, const QMessageLogContext& context, const QString& msg)
{
    QString logMessage;
    switch (type) {
    case QtDebugMsg:
        logMessage = QString("Debug: %1").arg(msg);
        break;
    case QtInfoMsg:
        logMessage = QString("Info: %1").arg(msg);
        break;
    case QtWarningMsg:
        logMessage = QString("Warning: %1").arg(msg);
        break;
    case QtCriticalMsg:
        logMessage = QString("Critical: %1").arg(msg);
        break;
    case QtFatalMsg:
        logMessage = QString("Fatal: %1").arg(msg);
        break;
    }

    QFile file("log.txt");
    if (file.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text)) {
        QTextStream out(&file);
        out << QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss") << " " << logMessage << "\n";
        file.close();
    }
}