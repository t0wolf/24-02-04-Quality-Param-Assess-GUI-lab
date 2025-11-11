#ifndef LOGIN_DIALOG_H
#define LOGIN_DIALOG_H

#include <QDialog>
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>
#include <QString>
#include <QCryptographicHash>

class LoginDialog : public QDialog
{
    Q_OBJECT
public:
    explicit LoginDialog(QWidget *parent = nullptr);

    QString getUsername() const { return m_username; }
    QString getPassword() const { return m_password; }

    QString md5Hash(const QString &input)
    {
        return QString(QCryptographicHash::hash(input.toUtf8(), QCryptographicHash::Md5).toHex());
    }

private slots:
    void onLoginClicked();

private:
    QLineEdit *usernameEdit;
    QLineEdit *passwordEdit;
    QPushButton *loginButton;
    QLabel *statusLabel;

    QString m_storedUsername;
    QString m_storedPassword;

    QString m_username;
    QString m_password;

    bool loadCredentials(const QString &filePath);

    bool createDefaultConfigFile(const QString &filePath);

};

#endif // LOGIN_DIALOG_H
