/********************************************************************************
** Form generated from reading UI file 'quality_detail_widget.ui'
**
** Created by: Qt User Interface Compiler version 5.12.11
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_QUALITY_DETAIL_WIDGET_H
#define UI_QUALITY_DETAIL_WIDGET_H

#include <QtCore/QVariant>
#include <QtGui/QIcon>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_QualityDetailWidget
{
public:
    QGridLayout *gridLayout;
    QFrame *frame;
    QVBoxLayout *verticalLayout;
    QFrame *frame_2;
    QHBoxLayout *horizontalLayout;
    QPushButton *saveButton;
    QPushButton *pushButton;
    QSpacerItem *horizontalSpacer;
    QGraphicsView *videoPlayLabel;
    QTableWidget *qualityScoresTable;

    void setupUi(QWidget *QualityDetailWidget)
    {
        if (QualityDetailWidget->objectName().isEmpty())
            QualityDetailWidget->setObjectName(QString::fromUtf8("QualityDetailWidget"));
        QualityDetailWidget->resize(645, 687);
        gridLayout = new QGridLayout(QualityDetailWidget);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        frame = new QFrame(QualityDetailWidget);
        frame->setObjectName(QString::fromUtf8("frame"));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        verticalLayout = new QVBoxLayout(frame);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        frame_2 = new QFrame(frame);
        frame_2->setObjectName(QString::fromUtf8("frame_2"));
        frame_2->setFrameShape(QFrame::StyledPanel);
        frame_2->setFrameShadow(QFrame::Raised);
        horizontalLayout = new QHBoxLayout(frame_2);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        saveButton = new QPushButton(frame_2);
        saveButton->setObjectName(QString::fromUtf8("saveButton"));
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/resources/icon_save_bright.png"), QSize(), QIcon::Normal, QIcon::Off);
        saveButton->setIcon(icon);

        horizontalLayout->addWidget(saveButton);

        pushButton = new QPushButton(frame_2);
        pushButton->setObjectName(QString::fromUtf8("pushButton"));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/resources/icon_delete_bright.png"), QSize(), QIcon::Normal, QIcon::Off);
        pushButton->setIcon(icon1);

        horizontalLayout->addWidget(pushButton);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);


        verticalLayout->addWidget(frame_2);

        videoPlayLabel = new QGraphicsView(frame);
        videoPlayLabel->setObjectName(QString::fromUtf8("videoPlayLabel"));
        videoPlayLabel->setMinimumSize(QSize(300, 400));

        verticalLayout->addWidget(videoPlayLabel);

        qualityScoresTable = new QTableWidget(frame);
        if (qualityScoresTable->columnCount() < 2)
            qualityScoresTable->setColumnCount(2);
        if (qualityScoresTable->rowCount() < 5)
            qualityScoresTable->setRowCount(5);
        qualityScoresTable->setObjectName(QString::fromUtf8("qualityScoresTable"));
        qualityScoresTable->setRowCount(5);
        qualityScoresTable->setColumnCount(2);
        qualityScoresTable->horizontalHeader()->setVisible(true);
        qualityScoresTable->horizontalHeader()->setCascadingSectionResizes(false);
        qualityScoresTable->horizontalHeader()->setDefaultSectionSize(300);
        qualityScoresTable->horizontalHeader()->setProperty("showSortIndicator", QVariant(false));

        verticalLayout->addWidget(qualityScoresTable);


        gridLayout->addWidget(frame, 1, 0, 1, 1);


        retranslateUi(QualityDetailWidget);

        QMetaObject::connectSlotsByName(QualityDetailWidget);
    } // setupUi

    void retranslateUi(QWidget *QualityDetailWidget)
    {
        QualityDetailWidget->setWindowTitle(QApplication::translate("QualityDetailWidget", "Form", nullptr));
        saveButton->setText(QApplication::translate("QualityDetailWidget", "\344\277\235\345\255\230", nullptr));
        pushButton->setText(QApplication::translate("QualityDetailWidget", "\345\210\240\351\231\244", nullptr));
    } // retranslateUi

};

namespace Ui {
    class QualityDetailWidget: public Ui_QualityDetailWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_QUALITY_DETAIL_WIDGET_H
