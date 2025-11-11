/********************************************************************************
** Form generated from reading UI file 'param_premium_widget.ui'
**
** Created by: Qt User Interface Compiler version 5.15.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_PARAM_PREMIUM_WIDGET_H
#define UI_PARAM_PREMIUM_WIDGET_H

#include <QtCore/QVariant>
#include <QtGui/QIcon>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ParamPremiumWidget
{
public:
    QVBoxLayout *verticalLayout_2;
    QFrame *frame;
    QVBoxLayout *verticalLayout;
    QFrame *frame_2;
    QHBoxLayout *horizontalLayout;
    QPushButton *saveButton;
    QPushButton *deleteButton;
    QSpacerItem *horizontalSpacer;
    QGraphicsView *premiumLabel;

    void setupUi(QWidget *ParamPremiumWidget)
    {
        if (ParamPremiumWidget->objectName().isEmpty())
            ParamPremiumWidget->setObjectName(QString::fromUtf8("ParamPremiumWidget"));
        ParamPremiumWidget->resize(594, 528);
        verticalLayout_2 = new QVBoxLayout(ParamPremiumWidget);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        frame = new QFrame(ParamPremiumWidget);
        frame->setObjectName(QString::fromUtf8("frame"));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        verticalLayout = new QVBoxLayout(frame);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        frame_2 = new QFrame(frame);
        frame_2->setObjectName(QString::fromUtf8("frame_2"));
        frame_2->setMaximumSize(QSize(16777215, 60));
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

        deleteButton = new QPushButton(frame_2);
        deleteButton->setObjectName(QString::fromUtf8("deleteButton"));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/resources/icon_delete_bright.png"), QSize(), QIcon::Normal, QIcon::Off);
        deleteButton->setIcon(icon1);

        horizontalLayout->addWidget(deleteButton);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);


        verticalLayout->addWidget(frame_2);

        premiumLabel = new QGraphicsView(frame);
        premiumLabel->setObjectName(QString::fromUtf8("premiumLabel"));

        verticalLayout->addWidget(premiumLabel);


        verticalLayout_2->addWidget(frame);


        retranslateUi(ParamPremiumWidget);

        QMetaObject::connectSlotsByName(ParamPremiumWidget);
    } // setupUi

    void retranslateUi(QWidget *ParamPremiumWidget)
    {
        ParamPremiumWidget->setWindowTitle(QCoreApplication::translate("ParamPremiumWidget", "Form", nullptr));
        saveButton->setText(QCoreApplication::translate("ParamPremiumWidget", "\344\277\235\345\255\230", nullptr));
        deleteButton->setText(QCoreApplication::translate("ParamPremiumWidget", "\345\210\240\351\231\244", nullptr));
    } // retranslateUi

};

namespace Ui {
    class ParamPremiumWidget: public Ui_ParamPremiumWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_PARAM_PREMIUM_WIDGET_H
