/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.12.11
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionSettings;
    QAction *actionAbout;
    QWidget *centralwidget;
    QGridLayout *gridLayout;
    QVBoxLayout *verticalLayout;
    QFrame *frame;
    QGridLayout *gridLayout_2;
    QPushButton *browserParamButton;
    QSpacerItem *horizontalSpacer_4;
    QLabel *label_2;
    QLabel *patientNameLabel;
    QSpacerItem *horizontalSpacer;
    QSpacerItem *horizontalSpacer_3;
    QLabel *patientIDLabel;
    QPushButton *patientListButton;
    QLabel *label;
    QSpacerItem *horizontalSpacer_2;
    QPushButton *clearAllButton;
    QSpacerItem *horizontalSpacer_5;
    QWidget *widget;
    QHBoxLayout *horizontalLayout_4;
    QWidget *displayWidget;
    QVBoxLayout *verticalLayout_2;
    QHBoxLayout *horizontalLayout_3;
    QWidget *rightSideWidget;
    QVBoxLayout *verticalLayout_4;
    QVBoxLayout *verticalLayout_3;
    QMenuBar *menubar;
    QMenu *actionMenu;
    QMenu *aboutMenu;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(1920, 1080);
        actionSettings = new QAction(MainWindow);
        actionSettings->setObjectName(QString::fromUtf8("actionSettings"));
        actionSettings->setCheckable(false);
        actionAbout = new QAction(MainWindow);
        actionAbout->setObjectName(QString::fromUtf8("actionAbout"));
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        gridLayout = new QGridLayout(centralwidget);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        frame = new QFrame(centralwidget);
        frame->setObjectName(QString::fromUtf8("frame"));
        frame->setMaximumSize(QSize(16777215, 75));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        gridLayout_2 = new QGridLayout(frame);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        browserParamButton = new QPushButton(frame);
        browserParamButton->setObjectName(QString::fromUtf8("browserParamButton"));
        browserParamButton->setMinimumSize(QSize(200, 50));
        QFont font;
        font.setPointSize(17);
        font.setBold(true);
        font.setWeight(75);
        browserParamButton->setFont(font);
        browserParamButton->setStyleSheet(QString::fromUtf8("QPushButton {\n"
"    /* \350\256\276\347\275\256\346\214\211\351\222\256\347\232\204\351\273\230\350\256\244\347\212\266\346\200\201\346\240\267\345\274\217 */\n"
"    background-color: #505050; /* \346\232\227\347\201\260\350\211\262\350\203\214\346\231\257 */\n"
"    color: #E0E0E0; /* \346\265\205\347\201\260\350\211\262\346\226\207\345\255\227 */\n"
"    border: 1px solid #707070; /* \350\276\271\346\241\206\351\242\234\350\211\262\347\250\215\344\272\256\344\270\200\344\272\233\347\232\204\347\201\260\350\211\262 */\n"
"    border-radius: 4px; /* \350\276\271\346\241\206\345\234\206\350\247\222 */\n"
"    padding: 5px; /* \345\206\205\351\203\250\345\241\253\345\205\205 */\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    /* \350\256\276\347\275\256\351\274\240\346\240\207\346\202\254\345\201\234\345\234\250\346\214\211\351\222\256\344\270\212\346\227\266\347\232\204\346\240\267\345\274\217 */\n"
"    background-color: #626262; /* \351\274\240\346\240\207\346\202\254\345\201\234\346\227\266\350\203\214\346\231\257"
                        "\351\242\234\350\211\262\345\217\230\346\267\261 */\n"
"    border: 1px solid #909090; /* \351\274\240\346\240\207\346\202\254\345\201\234\346\227\266\350\276\271\346\241\206\351\242\234\350\211\262\345\217\230\344\272\256 */\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    /* \350\256\276\347\275\256\346\214\211\351\222\256\350\242\253\346\214\211\344\270\213\346\227\266\347\232\204\346\240\267\345\274\217 */\n"
"    background-color: #787878; /* \346\214\211\351\222\256\346\214\211\344\270\213\346\227\266\347\232\204\350\203\214\346\231\257\351\242\234\350\211\262\345\217\230\345\276\227\346\233\264\346\267\261 */\n"
"    border: 1px solid #A0A0A0; /* \346\214\211\351\222\256\346\214\211\344\270\213\346\227\266\350\276\271\346\241\206\351\242\234\350\211\262\345\217\230\344\272\256 */\n"
"}\n"
"\n"
"QPushButton:disabled {\n"
"    /* \350\256\276\347\275\256\346\214\211\351\222\256\344\270\215\345\217\257\347\224\250\357\274\210\347\246\201\347\224\250\357\274\211\347\212\266\346\200\201\347\232\204\346\240\267\345"
                        "\274\217 */\n"
"    background-color: #303030; /* \347\246\201\347\224\250\347\212\266\346\200\201\347\232\204\346\232\227\350\211\262\350\203\214\346\231\257 */\n"
"    color: #606060; /* \347\246\201\347\224\250\347\212\266\346\200\201\347\232\204\346\226\207\345\255\227\351\242\234\350\211\262\350\276\203\346\232\227 */\n"
"    border: 1px solid #404040; /* \347\246\201\347\224\250\347\212\266\346\200\201\347\232\204\350\276\271\346\241\206\351\242\234\350\211\262 */\n"
"}"));

        gridLayout_2->addWidget(browserParamButton, 0, 8, 1, 1);

        horizontalSpacer_4 = new QSpacerItem(40, 20, QSizePolicy::Preferred, QSizePolicy::Minimum);

        gridLayout_2->addItem(horizontalSpacer_4, 0, 7, 1, 1);

        label_2 = new QLabel(frame);
        label_2->setObjectName(QString::fromUtf8("label_2"));
        QFont font1;
        font1.setFamily(QString::fromUtf8("\345\276\256\350\275\257\351\233\205\351\273\221"));
        font1.setBold(true);
        font1.setWeight(75);
        label_2->setFont(font1);
        label_2->setStyleSheet(QString::fromUtf8("QLabel {\n"
"    /* border: 3px solid #3399ff; */ /* \350\276\271\346\241\206\350\256\276\344\270\272\344\272\256\350\223\235\350\211\262\357\274\214\345\256\275\345\272\246\344\270\2723px */\n"
"    border-radius: 10px; /* \350\276\271\346\241\206\345\234\206\350\247\222\344\270\27210px */\n"
"    background-color: #2d2d30; /* \350\203\214\346\231\257\351\242\234\350\211\262\347\250\215\344\272\256\344\272\216\347\252\227\344\275\223\350\203\214\346\231\257 */\n"
"    color: #ffffff; /* \346\226\207\345\255\227\351\242\234\350\211\262\344\270\272\347\231\275\350\211\262 */\n"
"    padding: 8px; /* \350\256\276\347\275\256\345\206\205\350\276\271\350\267\235 */\n"
"    font-family: '\345\276\256\350\275\257\351\233\205\351\273\221', Arial, sans-serif; /* \351\246\226\351\200\211\345\276\256\350\275\257\351\233\205\351\273\221\357\274\214\345\246\202\346\236\234\347\263\273\347\273\237\344\270\215\346\224\257\346\214\201\345\210\231\345\233\236\351\200\200\350\207\263Arial */\n"
"    font-size: 14px; /* \345\255"
                        "\227\344\275\223\345\244\247\345\260\217\350\256\276\347\275\256\344\270\27214px */\n"
"    font-weight: bold; /* \345\255\227\344\275\223\345\212\240\347\262\227 */\n"
"}"));

        gridLayout_2->addWidget(label_2, 0, 3, 1, 1);

        patientNameLabel = new QLabel(frame);
        patientNameLabel->setObjectName(QString::fromUtf8("patientNameLabel"));
        patientNameLabel->setMinimumSize(QSize(150, 0));
        patientNameLabel->setFont(font1);
        patientNameLabel->setStyleSheet(QString::fromUtf8("QLabel {\n"
"    border: 3px solid #3399ff; /* \350\276\271\346\241\206\350\256\276\344\270\272\344\272\256\350\223\235\350\211\262\357\274\214\345\256\275\345\272\246\344\270\2723px */\n"
"    border-radius: 10px; /* \350\276\271\346\241\206\345\234\206\350\247\222\344\270\27210px */\n"
"    background-color: #2d2d30; /* \350\203\214\346\231\257\351\242\234\350\211\262\347\250\215\344\272\256\344\272\216\347\252\227\344\275\223\350\203\214\346\231\257 */\n"
"    color: #ffffff; /* \346\226\207\345\255\227\351\242\234\350\211\262\344\270\272\347\231\275\350\211\262 */\n"
"    padding: 8px; /* \350\256\276\347\275\256\345\206\205\350\276\271\350\267\235 */\n"
"    font-family: '\345\276\256\350\275\257\351\233\205\351\273\221', Arial, sans-serif; /* \351\246\226\351\200\211\345\276\256\350\275\257\351\233\205\351\273\221\357\274\214\345\246\202\346\236\234\347\263\273\347\273\237\344\270\215\346\224\257\346\214\201\345\210\231\345\233\236\351\200\200\350\207\263Arial */\n"
"    font-size: 14px; /* \345\255\227\344"
                        "\275\223\345\244\247\345\260\217\350\256\276\347\275\256\344\270\27214px */\n"
"    font-weight: bold; /* \345\255\227\344\275\223\345\212\240\347\262\227 */\n"
"}"));

        gridLayout_2->addWidget(patientNameLabel, 0, 1, 1, 1);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout_2->addItem(horizontalSpacer, 0, 11, 1, 1);

        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Preferred, QSizePolicy::Minimum);

        gridLayout_2->addItem(horizontalSpacer_3, 0, 5, 1, 1);

        patientIDLabel = new QLabel(frame);
        patientIDLabel->setObjectName(QString::fromUtf8("patientIDLabel"));
        patientIDLabel->setMinimumSize(QSize(150, 0));
        patientIDLabel->setFont(font1);
        patientIDLabel->setStyleSheet(QString::fromUtf8("QLabel {\n"
"    border: 3px solid #3399ff; /* \350\276\271\346\241\206\350\256\276\344\270\272\344\272\256\350\223\235\350\211\262\357\274\214\345\256\275\345\272\246\344\270\2723px */\n"
"    border-radius: 10px; /* \350\276\271\346\241\206\345\234\206\350\247\222\344\270\27210px */\n"
"    background-color: #2d2d30; /* \350\203\214\346\231\257\351\242\234\350\211\262\347\250\215\344\272\256\344\272\216\347\252\227\344\275\223\350\203\214\346\231\257 */\n"
"    color: #ffffff; /* \346\226\207\345\255\227\351\242\234\350\211\262\344\270\272\347\231\275\350\211\262 */\n"
"    padding: 8px; /* \350\256\276\347\275\256\345\206\205\350\276\271\350\267\235 */\n"
"    font-family: '\345\276\256\350\275\257\351\233\205\351\273\221', Arial, sans-serif; /* \351\246\226\351\200\211\345\276\256\350\275\257\351\233\205\351\273\221\357\274\214\345\246\202\346\236\234\347\263\273\347\273\237\344\270\215\346\224\257\346\214\201\345\210\231\345\233\236\351\200\200\350\207\263Arial */\n"
"    font-size: 14px; /* \345\255\227\344"
                        "\275\223\345\244\247\345\260\217\350\256\276\347\275\256\344\270\27214px */\n"
"    font-weight: bold; /* \345\255\227\344\275\223\345\212\240\347\262\227 */\n"
"}"));

        gridLayout_2->addWidget(patientIDLabel, 0, 4, 1, 1);

        patientListButton = new QPushButton(frame);
        patientListButton->setObjectName(QString::fromUtf8("patientListButton"));
        patientListButton->setMinimumSize(QSize(200, 50));
        patientListButton->setFont(font);
        patientListButton->setStyleSheet(QString::fromUtf8("QPushButton {\n"
"    /* \350\256\276\347\275\256\346\214\211\351\222\256\347\232\204\351\273\230\350\256\244\347\212\266\346\200\201\346\240\267\345\274\217 */\n"
"    background-color: #505050; /* \346\232\227\347\201\260\350\211\262\350\203\214\346\231\257 */\n"
"    color: #E0E0E0; /* \346\265\205\347\201\260\350\211\262\346\226\207\345\255\227 */\n"
"    border: 1px solid #707070; /* \350\276\271\346\241\206\351\242\234\350\211\262\347\250\215\344\272\256\344\270\200\344\272\233\347\232\204\347\201\260\350\211\262 */\n"
"    border-radius: 4px; /* \350\276\271\346\241\206\345\234\206\350\247\222 */\n"
"    padding: 5px; /* \345\206\205\351\203\250\345\241\253\345\205\205 */\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    /* \350\256\276\347\275\256\351\274\240\346\240\207\346\202\254\345\201\234\345\234\250\346\214\211\351\222\256\344\270\212\346\227\266\347\232\204\346\240\267\345\274\217 */\n"
"    background-color: #626262; /* \351\274\240\346\240\207\346\202\254\345\201\234\346\227\266\350\203\214\346\231\257"
                        "\351\242\234\350\211\262\345\217\230\346\267\261 */\n"
"    border: 1px solid #909090; /* \351\274\240\346\240\207\346\202\254\345\201\234\346\227\266\350\276\271\346\241\206\351\242\234\350\211\262\345\217\230\344\272\256 */\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    /* \350\256\276\347\275\256\346\214\211\351\222\256\350\242\253\346\214\211\344\270\213\346\227\266\347\232\204\346\240\267\345\274\217 */\n"
"    background-color: #787878; /* \346\214\211\351\222\256\346\214\211\344\270\213\346\227\266\347\232\204\350\203\214\346\231\257\351\242\234\350\211\262\345\217\230\345\276\227\346\233\264\346\267\261 */\n"
"    border: 1px solid #A0A0A0; /* \346\214\211\351\222\256\346\214\211\344\270\213\346\227\266\350\276\271\346\241\206\351\242\234\350\211\262\345\217\230\344\272\256 */\n"
"}\n"
"\n"
"QPushButton:disabled {\n"
"    /* \350\256\276\347\275\256\346\214\211\351\222\256\344\270\215\345\217\257\347\224\250\357\274\210\347\246\201\347\224\250\357\274\211\347\212\266\346\200\201\347\232\204\346\240\267\345"
                        "\274\217 */\n"
"    background-color: #303030; /* \347\246\201\347\224\250\347\212\266\346\200\201\347\232\204\346\232\227\350\211\262\350\203\214\346\231\257 */\n"
"    color: #606060; /* \347\246\201\347\224\250\347\212\266\346\200\201\347\232\204\346\226\207\345\255\227\351\242\234\350\211\262\350\276\203\346\232\227 */\n"
"    border: 1px solid #404040; /* \347\246\201\347\224\250\347\212\266\346\200\201\347\232\204\350\276\271\346\241\206\351\242\234\350\211\262 */\n"
"}"));

        gridLayout_2->addWidget(patientListButton, 0, 10, 1, 1);

        label = new QLabel(frame);
        label->setObjectName(QString::fromUtf8("label"));
        label->setFont(font1);
        label->setStyleSheet(QString::fromUtf8("QLabel {\n"
"    /* border: 3px solid #3399ff; */ /* \350\276\271\346\241\206\350\256\276\344\270\272\344\272\256\350\223\235\350\211\262\357\274\214\345\256\275\345\272\246\344\270\2723px */\n"
"    border-radius: 10px; /* \350\276\271\346\241\206\345\234\206\350\247\222\344\270\27210px */\n"
"    background-color: #2d2d30; /* \350\203\214\346\231\257\351\242\234\350\211\262\347\250\215\344\272\256\344\272\216\347\252\227\344\275\223\350\203\214\346\231\257 */\n"
"    color: #ffffff; /* \346\226\207\345\255\227\351\242\234\350\211\262\344\270\272\347\231\275\350\211\262 */\n"
"    padding: 8px; /* \350\256\276\347\275\256\345\206\205\350\276\271\350\267\235 */\n"
"    font-family: '\345\276\256\350\275\257\351\233\205\351\273\221', Arial, sans-serif; /* \351\246\226\351\200\211\345\276\256\350\275\257\351\233\205\351\273\221\357\274\214\345\246\202\346\236\234\347\263\273\347\273\237\344\270\215\346\224\257\346\214\201\345\210\231\345\233\236\351\200\200\350\207\263Arial */\n"
"    font-size: 14px; /* \345\255"
                        "\227\344\275\223\345\244\247\345\260\217\350\256\276\347\275\256\344\270\27214px */\n"
"    font-weight: bold; /* \345\255\227\344\275\223\345\212\240\347\262\227 */\n"
"}"));

        gridLayout_2->addWidget(label, 0, 0, 1, 1);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Preferred, QSizePolicy::Minimum);

        gridLayout_2->addItem(horizontalSpacer_2, 0, 2, 1, 1);

        clearAllButton = new QPushButton(frame);
        clearAllButton->setObjectName(QString::fromUtf8("clearAllButton"));
        clearAllButton->setMinimumSize(QSize(200, 50));
        clearAllButton->setFont(font);
        clearAllButton->setStyleSheet(QString::fromUtf8("QPushButton {\n"
"    /* \350\256\276\347\275\256\346\214\211\351\222\256\347\232\204\351\273\230\350\256\244\347\212\266\346\200\201\346\240\267\345\274\217 */\n"
"    background-color: #505050; /* \346\232\227\347\201\260\350\211\262\350\203\214\346\231\257 */\n"
"    color: #E0E0E0; /* \346\265\205\347\201\260\350\211\262\346\226\207\345\255\227 */\n"
"    border: 1px solid #707070; /* \350\276\271\346\241\206\351\242\234\350\211\262\347\250\215\344\272\256\344\270\200\344\272\233\347\232\204\347\201\260\350\211\262 */\n"
"    border-radius: 4px; /* \350\276\271\346\241\206\345\234\206\350\247\222 */\n"
"    padding: 5px; /* \345\206\205\351\203\250\345\241\253\345\205\205 */\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    /* \350\256\276\347\275\256\351\274\240\346\240\207\346\202\254\345\201\234\345\234\250\346\214\211\351\222\256\344\270\212\346\227\266\347\232\204\346\240\267\345\274\217 */\n"
"    background-color: #626262; /* \351\274\240\346\240\207\346\202\254\345\201\234\346\227\266\350\203\214\346\231\257"
                        "\351\242\234\350\211\262\345\217\230\346\267\261 */\n"
"    border: 1px solid #909090; /* \351\274\240\346\240\207\346\202\254\345\201\234\346\227\266\350\276\271\346\241\206\351\242\234\350\211\262\345\217\230\344\272\256 */\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    /* \350\256\276\347\275\256\346\214\211\351\222\256\350\242\253\346\214\211\344\270\213\346\227\266\347\232\204\346\240\267\345\274\217 */\n"
"    background-color: #787878; /* \346\214\211\351\222\256\346\214\211\344\270\213\346\227\266\347\232\204\350\203\214\346\231\257\351\242\234\350\211\262\345\217\230\345\276\227\346\233\264\346\267\261 */\n"
"    border: 1px solid #A0A0A0; /* \346\214\211\351\222\256\346\214\211\344\270\213\346\227\266\350\276\271\346\241\206\351\242\234\350\211\262\345\217\230\344\272\256 */\n"
"}\n"
"\n"
"QPushButton:disabled {\n"
"    /* \350\256\276\347\275\256\346\214\211\351\222\256\344\270\215\345\217\257\347\224\250\357\274\210\347\246\201\347\224\250\357\274\211\347\212\266\346\200\201\347\232\204\346\240\267\345"
                        "\274\217 */\n"
"    background-color: #303030; /* \347\246\201\347\224\250\347\212\266\346\200\201\347\232\204\346\232\227\350\211\262\350\203\214\346\231\257 */\n"
"    color: #606060; /* \347\246\201\347\224\250\347\212\266\346\200\201\347\232\204\346\226\207\345\255\227\351\242\234\350\211\262\350\276\203\346\232\227 */\n"
"    border: 1px solid #404040; /* \347\246\201\347\224\250\347\212\266\346\200\201\347\232\204\350\276\271\346\241\206\351\242\234\350\211\262 */\n"
"}"));

        gridLayout_2->addWidget(clearAllButton, 0, 6, 1, 1);

        horizontalSpacer_5 = new QSpacerItem(40, 20, QSizePolicy::Preferred, QSizePolicy::Minimum);

        gridLayout_2->addItem(horizontalSpacer_5, 0, 9, 1, 1);


        verticalLayout->addWidget(frame);

        widget = new QWidget(centralwidget);
        widget->setObjectName(QString::fromUtf8("widget"));
        horizontalLayout_4 = new QHBoxLayout(widget);
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        displayWidget = new QWidget(widget);
        displayWidget->setObjectName(QString::fromUtf8("displayWidget"));
        displayWidget->setMinimumSize(QSize(1000, 600));
        verticalLayout_2 = new QVBoxLayout(displayWidget);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setSpacing(7);
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));

        verticalLayout_2->addLayout(horizontalLayout_3);


        horizontalLayout_4->addWidget(displayWidget);

        rightSideWidget = new QWidget(widget);
        rightSideWidget->setObjectName(QString::fromUtf8("rightSideWidget"));
        rightSideWidget->setMinimumSize(QSize(80, 0));
        rightSideWidget->setMaximumSize(QSize(500, 16777215));
        verticalLayout_4 = new QVBoxLayout(rightSideWidget);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        verticalLayout_3 = new QVBoxLayout();
        verticalLayout_3->setSpacing(1);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));

        verticalLayout_4->addLayout(verticalLayout_3);


        horizontalLayout_4->addWidget(rightSideWidget);


        verticalLayout->addWidget(widget);


        gridLayout->addLayout(verticalLayout, 0, 0, 1, 1);

        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName(QString::fromUtf8("menubar"));
        menubar->setGeometry(QRect(0, 0, 1920, 23));
        actionMenu = new QMenu(menubar);
        actionMenu->setObjectName(QString::fromUtf8("actionMenu"));
        aboutMenu = new QMenu(menubar);
        aboutMenu->setObjectName(QString::fromUtf8("aboutMenu"));
        MainWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        MainWindow->setStatusBar(statusbar);

        menubar->addAction(actionMenu->menuAction());
        menubar->addAction(aboutMenu->menuAction());
        actionMenu->addAction(actionSettings);
        aboutMenu->addAction(actionAbout);

        retranslateUi(MainWindow);
        QObject::connect(actionSettings, SIGNAL(triggered()), MainWindow, SLOT(onActionSettingsTriggered()));
        QObject::connect(actionAbout, SIGNAL(triggered()), MainWindow, SLOT(onAboutSettingsTriggered()));
        QObject::connect(patientListButton, SIGNAL(clicked()), MainWindow, SLOT(onPatientListButtonClicked()));

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", nullptr));
        actionSettings->setText(QApplication::translate("MainWindow", "\347\263\273\347\273\237\351\205\215\347\275\256", nullptr));
        actionAbout->setText(QApplication::translate("MainWindow", "\345\205\263\344\272\216", nullptr));
        browserParamButton->setText(QApplication::translate("MainWindow", "\346\265\217\350\247\210\345\217\202\346\225\260", nullptr));
        label_2->setText(QApplication::translate("MainWindow", "\347\227\205\344\272\272ID\357\274\232", nullptr));
        patientNameLabel->setText(QString());
        patientIDLabel->setText(QString());
        patientListButton->setText(QApplication::translate("MainWindow", "\347\227\205\344\272\272\345\210\227\350\241\250", nullptr));
        label->setText(QApplication::translate("MainWindow", "\347\227\205\344\272\272\345\247\223\345\220\215\357\274\232", nullptr));
        clearAllButton->setText(QApplication::translate("MainWindow", "\346\270\205\347\251\272", nullptr));
        actionMenu->setTitle(QApplication::translate("MainWindow", "\351\205\215\347\275\256", nullptr));
        aboutMenu->setTitle(QApplication::translate("MainWindow", "\345\270\256\345\212\251", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
