#include "logindialog.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFile>
#include <QTextStream>
#include <QMessageBox>

LoginDialog::LoginDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(QStringLiteral("登录"));

    // 创建控件
    QLabel *userLabel = new QLabel(QStringLiteral("用户名:"), this);
    usernameEdit = new QLineEdit(this);

    QLabel *pwdLabel = new QLabel(QStringLiteral("密码:"), this);
    passwordEdit = new QLineEdit(this);
    passwordEdit->setEchoMode(QLineEdit::Password);

    loginButton = new QPushButton(QStringLiteral("登录"), this);
    statusLabel = new QLabel(this);
    statusLabel->setStyleSheet("color: red");

    // 布局
    QVBoxLayout *mainLayout = new QVBoxLayout(this);

    QHBoxLayout *userLayout = new QHBoxLayout();
    userLayout->addWidget(userLabel);
    userLayout->addWidget(usernameEdit);

    QHBoxLayout *pwdLayout = new QHBoxLayout();
    pwdLayout->addWidget(pwdLabel);
    pwdLayout->addWidget(passwordEdit);

    mainLayout->addLayout(userLayout);
    mainLayout->addLayout(pwdLayout);
    mainLayout->addWidget(loginButton);
    mainLayout->addWidget(statusLabel);

    setLayout(mainLayout);
    resize(300, 150);

    // 连接信号
    connect(loginButton, &QPushButton::clicked, this, &LoginDialog::onLoginClicked);

    if (!loadCredentials("user_config.txt")) {
        // 创建默认配置文件
        if (createDefaultConfigFile("user_config.txt")) {
            m_storedUsername = "admin";
            m_storedPassword = "Admin@123";
            QMessageBox::information(this, QStringLiteral("提示"), QStringLiteral("配置文件不存在，已创建默认配置文件，默认用户名密码为 admin / Admin@123"));
        } else {
            QMessageBox::warning(this, QStringLiteral("警告"), QStringLiteral("无法读取配置文件，且创建默认配置文件失败，将使用默认用户名密码"));
            m_storedUsername = "admin";
            m_storedPassword = "Admin@123";
        }
    }
}

bool LoginDialog::createDefaultConfigFile(const QString &filePath)
{
    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
        return false;

    QTextStream out(&file);

    const QString defaultUser = "admin";
    const QString defaultPwd = "Admin@123";
    QString md5Pwd = md5Hash(defaultPwd);

    out << "credentials:\n";
    out << "  username: \"" << defaultUser << "\"\n";
    out << "  password: \"" << md5Pwd << "\"\n";

    file.close();
    return true;
}

bool LoginDialog::loadCredentials(const QString &filePath)
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

    m_storedUsername = username;
    m_storedPassword = password;
    return true;
}

void LoginDialog::onLoginClicked()
{
    m_username = usernameEdit->text();
    m_password = passwordEdit->text();


    if (m_username.isEmpty() || m_password.isEmpty()) {
        statusLabel->setText(QStringLiteral("用户名和密码不能为空"));
        return;
    }

    QString inputPwdMd5 = md5Hash(m_password);

    if (m_username == m_storedUsername && inputPwdMd5 == m_storedPassword) {
        accept();
    } else {
        statusLabel->setText(QStringLiteral("用户名或密码错误！"));
    }
}
