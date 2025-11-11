#include "mainwindow.h"
#include "logindialog.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    //qInstallMessageHandler(customMessageHandler);
    //QApplication a(argc, argv);
    QtLogger::instance().startLoggingToFile("echo_ai_system.log");
    SafeApplication a(argc, argv);
    //设置暗色系皮肤
    QApplication::setStyle(new DarkStyle);
    // QApplication::setPalette(QApplication::style()->standardPalette());

    //设置中文字体
    // a.setFont(QFont("Microsoft Yahei", 9));

    //设置中文编码
#if (QT_VERSION <= QT_VERSION_CHECK(5,0,0))
#if _MSC_VER
    QTextCodec *codec = QTextCodec::codecForName("GBK");
#else
    QTextCodec *codec = QTextCodec::codecForName("UTF-8");
#endif
    QTextCodec::setCodecForLocale(codec);
    QTextCodec::setCodecForCStrings(codec);
    QTextCodec::setCodecForTr(codec);
#else
    QTextCodec *codec = QTextCodec::codecForName("UTF-8");
    QTextCodec::setCodecForLocale(codec);
#endif
    FramelessWindow framelessWindow;
    framelessWindow.setWindowTitle("心脏超声影像处理软件");
    framelessWindow.setWindowState(Qt::WindowMaximized);

    MainWindow w;
    framelessWindow.setContent(&w);
    framelessWindow.show();
    //framelessWindow.showFullScreen();
    //HttpServer server;
    // w.show();
    return a.exec();
    //LoginDialog loginDlg;

    //if (loginDlg.exec() == QDialog::Accepted)
    //{
    //    FramelessWindow framelessWindow;
    //    framelessWindow.setWindowTitle("心脏超声影像处理软件");
    //    framelessWindow.setWindowState(Qt::WindowMaximized);

    //    MainWindow w;
    //    framelessWindow.setContent(&w);
    //    framelessWindow.show();
    //    //framelessWindow.showFullScreen();
    //    //HttpServer server;
    //    // w.show();
    //    return a.exec();
    //}
    //return 0;
}
