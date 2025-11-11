/********************************************************************************
** Form generated from reading UI file 'display_stream_widget.ui'
**
** Created by: Qt User Interface Compiler version 5.15.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DISPLAY_STREAM_WIDGET_H
#define UI_DISPLAY_STREAM_WIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_DisplayStreamWidget
{
public:
    QVBoxLayout *verticalLayout_2;
    QFrame *frame_2;
    QHBoxLayout *horizontalLayout;
    QPushButton *beginButton;
    QPushButton *pauseButton;
    QFrame *line;
    QPushButton *freezeButton;
    QPushButton *hideButton;
    QSpacerItem *horizontalSpacer;
    QFrame *frame;
    QVBoxLayout *verticalLayout;
    QGraphicsView *displayView;
    QLabel *displayLabel;

    void setupUi(QWidget *DisplayStreamWidget)
    {
        if (DisplayStreamWidget->objectName().isEmpty())
            DisplayStreamWidget->setObjectName(QString::fromUtf8("DisplayStreamWidget"));
        DisplayStreamWidget->resize(1064, 697);
        DisplayStreamWidget->setMaximumSize(QSize(16777215, 16777215));
        verticalLayout_2 = new QVBoxLayout(DisplayStreamWidget);
        verticalLayout_2->setSpacing(2);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        frame_2 = new QFrame(DisplayStreamWidget);
        frame_2->setObjectName(QString::fromUtf8("frame_2"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(frame_2->sizePolicy().hasHeightForWidth());
        frame_2->setSizePolicy(sizePolicy);
        frame_2->setMinimumSize(QSize(0, 65));
        frame_2->setMaximumSize(QSize(16777215, 16777215));
        frame_2->setStyleSheet(QString::fromUtf8("QFrame{\n"
"background-color: palette(mid);\n"
"padding: 1px\n"
"}"));
        frame_2->setFrameShape(QFrame::StyledPanel);
        frame_2->setFrameShadow(QFrame::Raised);
        horizontalLayout = new QHBoxLayout(frame_2);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setContentsMargins(-1, 9, -1, -1);
        beginButton = new QPushButton(frame_2);
        beginButton->setObjectName(QString::fromUtf8("beginButton"));
        beginButton->setMaximumSize(QSize(16777215, 70));
        QFont font;
        font.setFamily(QString::fromUtf8("Agency FB"));
        font.setPointSize(12);
        font.setBold(true);
        font.setWeight(75);
        beginButton->setFont(font);

        horizontalLayout->addWidget(beginButton);

        pauseButton = new QPushButton(frame_2);
        pauseButton->setObjectName(QString::fromUtf8("pauseButton"));
        pauseButton->setMaximumSize(QSize(16777215, 70));
        pauseButton->setFont(font);

        horizontalLayout->addWidget(pauseButton);

        line = new QFrame(frame_2);
        line->setObjectName(QString::fromUtf8("line"));
        line->setFrameShape(QFrame::VLine);
        line->setFrameShadow(QFrame::Sunken);

        horizontalLayout->addWidget(line);

        freezeButton = new QPushButton(frame_2);
        freezeButton->setObjectName(QString::fromUtf8("freezeButton"));
        freezeButton->setMaximumSize(QSize(16777215, 70));
        freezeButton->setFont(font);

        horizontalLayout->addWidget(freezeButton);

        hideButton = new QPushButton(frame_2);
        hideButton->setObjectName(QString::fromUtf8("hideButton"));
        hideButton->setMaximumSize(QSize(16777215, 70));
        hideButton->setFont(font);

        horizontalLayout->addWidget(hideButton);

        horizontalSpacer = new QSpacerItem(60, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);


        verticalLayout_2->addWidget(frame_2);

        frame = new QFrame(DisplayStreamWidget);
        frame->setObjectName(QString::fromUtf8("frame"));
        QSizePolicy sizePolicy1(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(frame->sizePolicy().hasHeightForWidth());
        frame->setSizePolicy(sizePolicy1);
        frame->setMinimumSize(QSize(0, 0));
        frame->setMaximumSize(QSize(16777215, 16777215));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        verticalLayout = new QVBoxLayout(frame);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        displayView = new QGraphicsView(frame);
        displayView->setObjectName(QString::fromUtf8("displayView"));

        verticalLayout->addWidget(displayView);

        displayLabel = new QLabel(frame);
        displayLabel->setObjectName(QString::fromUtf8("displayLabel"));

        verticalLayout->addWidget(displayLabel);


        verticalLayout_2->addWidget(frame);


        retranslateUi(DisplayStreamWidget);

        QMetaObject::connectSlotsByName(DisplayStreamWidget);
    } // setupUi

    void retranslateUi(QWidget *DisplayStreamWidget)
    {
        DisplayStreamWidget->setWindowTitle(QCoreApplication::translate("DisplayStreamWidget", "Form", nullptr));
        beginButton->setText(QCoreApplication::translate("DisplayStreamWidget", "\345\274\200\345\247\213\346\243\200\346\237\245", nullptr));
        pauseButton->setText(QCoreApplication::translate("DisplayStreamWidget", "\346\232\202\345\201\234\346\243\200\346\237\245", nullptr));
        freezeButton->setText(QCoreApplication::translate("DisplayStreamWidget", "\345\206\273\347\273\223", nullptr));
        hideButton->setText(QCoreApplication::translate("DisplayStreamWidget", "\351\232\220\350\227\217", nullptr));
        displayLabel->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class DisplayStreamWidget: public Ui_DisplayStreamWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DISPLAY_STREAM_WIDGET_H
