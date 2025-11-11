/********************************************************************************
** Form generated from reading UI file 'ParamBrowserWidget.ui'
**
** Created by: Qt User Interface Compiler version 5.15.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_PARAMBROWSERWIDGET_H
#define UI_PARAMBROWSERWIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ParamBrowserWidgetClass
{
public:
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout;
    QPushButton *saveAllButton;
    QPushButton *deleteAllButton;
    QSpacerItem *horizontalSpacer;
    QListWidget *imageList;

    void setupUi(QWidget *ParamBrowserWidgetClass)
    {
        if (ParamBrowserWidgetClass->objectName().isEmpty())
            ParamBrowserWidgetClass->setObjectName(QString::fromUtf8("ParamBrowserWidgetClass"));
        ParamBrowserWidgetClass->resize(900, 400);
        verticalLayout = new QVBoxLayout(ParamBrowserWidgetClass);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        saveAllButton = new QPushButton(ParamBrowserWidgetClass);
        saveAllButton->setObjectName(QString::fromUtf8("saveAllButton"));
        saveAllButton->setMinimumSize(QSize(150, 0));
        saveAllButton->setMaximumSize(QSize(150, 16777215));

        horizontalLayout->addWidget(saveAllButton);

        deleteAllButton = new QPushButton(ParamBrowserWidgetClass);
        deleteAllButton->setObjectName(QString::fromUtf8("deleteAllButton"));
        deleteAllButton->setMinimumSize(QSize(150, 0));
        deleteAllButton->setMaximumSize(QSize(150, 16777215));

        horizontalLayout->addWidget(deleteAllButton);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);


        verticalLayout->addLayout(horizontalLayout);

        imageList = new QListWidget(ParamBrowserWidgetClass);
        imageList->setObjectName(QString::fromUtf8("imageList"));
        imageList->setFlow(QListView::LeftToRight);

        verticalLayout->addWidget(imageList);


        retranslateUi(ParamBrowserWidgetClass);

        QMetaObject::connectSlotsByName(ParamBrowserWidgetClass);
    } // setupUi

    void retranslateUi(QWidget *ParamBrowserWidgetClass)
    {
        ParamBrowserWidgetClass->setWindowTitle(QCoreApplication::translate("ParamBrowserWidgetClass", "ParamBrowserWidget", nullptr));
        saveAllButton->setText(QCoreApplication::translate("ParamBrowserWidgetClass", "\345\205\250\351\203\250\344\277\235\345\255\230", nullptr));
        deleteAllButton->setText(QCoreApplication::translate("ParamBrowserWidgetClass", "\345\205\250\351\203\250\345\210\240\351\231\244", nullptr));
    } // retranslateUi

};

namespace Ui {
    class ParamBrowserWidgetClass: public Ui_ParamBrowserWidgetClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_PARAMBROWSERWIDGET_H
