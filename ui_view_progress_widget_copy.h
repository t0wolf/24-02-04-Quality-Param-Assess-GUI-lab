/********************************************************************************
** Form generated from reading UI file 'view_progress_widget_copy.ui'
**
** Created by: Qt User Interface Compiler version 5.12.11
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_VIEW_PROGRESS_WIDGET_COPY_H
#define UI_VIEW_PROGRESS_WIDGET_COPY_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ViewProgressWidget
{
public:
    QVBoxLayout *verticalLayout_2;
    QFrame *frame;
    QVBoxLayout *verticalLayout_3;
    QWidget *widget;
    QGridLayout *gridLayout;
    QListWidget *apicalProgressList;
    QListWidget *paramProgressList;
    QListWidget *shortProgressList;
    QListWidget *plaxParamProgressList;
    QFrame *frame_2;
    QGridLayout *gridLayout_2;
    QLabel *paramDemoLabel;

    void setupUi(QWidget *ViewProgressWidget)
    {
        if (ViewProgressWidget->objectName().isEmpty())
            ViewProgressWidget->setObjectName(QString::fromUtf8("ViewProgressWidget"));
        ViewProgressWidget->resize(470, 579);
        ViewProgressWidget->setMinimumSize(QSize(400, 0));
        verticalLayout_2 = new QVBoxLayout(ViewProgressWidget);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        frame = new QFrame(ViewProgressWidget);
        frame->setObjectName(QString::fromUtf8("frame"));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        verticalLayout_3 = new QVBoxLayout(frame);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        widget = new QWidget(frame);
        widget->setObjectName(QString::fromUtf8("widget"));
        gridLayout = new QGridLayout(widget);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        apicalProgressList = new QListWidget(widget);
        new QListWidgetItem(apicalProgressList);
        new QListWidgetItem(apicalProgressList);
        new QListWidgetItem(apicalProgressList);
        new QListWidgetItem(apicalProgressList);
        apicalProgressList->setObjectName(QString::fromUtf8("apicalProgressList"));
        apicalProgressList->setMaximumSize(QSize(16777215, 150));
        QFont font;
        font.setPointSize(14);
        apicalProgressList->setFont(font);
        apicalProgressList->setStyleSheet(QString::fromUtf8("\n"
"QListWidget {\n"
"    background-color: #2b2b2b;\n"
"    border: 2px solid #3c3f41;\n"
"}\n"
"\n"
"QListWidget::item {\n"
"    padding: 5px;\n"
"    color: #a9b7c6;\n"
"    border-bottom: 1px solid #3c3f41;\n"
"}\n"
"\n"
"QListWidget::item:last-child {\n"
"    border-bottom: none;\n"
"}\n"
"\n"
"QListWidget::item:selected {\n"
"    background-color: #214283;\n"
"    color: #a9b7c6;\n"
"}\n"
"\n"
"QListWidget::item:hover {\n"
"    background-color: #3c3f41;\n"
"}\n"
"\n"
"QListWidget::item:disabled {\n"
"    color: #808080;\n"
"}\n"
"\n"
"QListWidget::item:checked {\n"
"    background-color: #214283;\n"
"}"));

        gridLayout->addWidget(apicalProgressList, 1, 1, 1, 1);

        paramProgressList = new QListWidget(widget);
        new QListWidgetItem(paramProgressList);
        new QListWidgetItem(paramProgressList);
        new QListWidgetItem(paramProgressList);
        new QListWidgetItem(paramProgressList);
        new QListWidgetItem(paramProgressList);
        new QListWidgetItem(paramProgressList);
        paramProgressList->setObjectName(QString::fromUtf8("paramProgressList"));
        QFont font1;
        font1.setPointSize(12);
        paramProgressList->setFont(font1);
        paramProgressList->setStyleSheet(QString::fromUtf8("\n"
"QListWidget {\n"
"    background-color: #2b2b2b;\n"
"    border: 2px solid #3c3f41;\n"
"}\n"
"\n"
"QListWidget::item {\n"
"    padding: 5px;\n"
"    color: #a9b7c6;\n"
"    border-bottom: 1px solid #3c3f41;\n"
"}\n"
"\n"
"QListWidget::item:last-child {\n"
"    border-bottom: none;\n"
"}\n"
"\n"
"QListWidget::item:selected {\n"
"    background-color: #214283;\n"
"    color: #a9b7c6;\n"
"}\n"
"\n"
"QListWidget::item:hover {\n"
"    background-color: #3c3f41;\n"
"}\n"
"\n"
"QListWidget::item:disabled {\n"
"    color: #808080;\n"
"}\n"
"\n"
"QListWidget::item:checked {\n"
"    background-color: #214283;\n"
"}"));

        gridLayout->addWidget(paramProgressList, 2, 2, 1, 1);

        shortProgressList = new QListWidget(widget);
        new QListWidgetItem(shortProgressList);
        new QListWidgetItem(shortProgressList);
        new QListWidgetItem(shortProgressList);
        new QListWidgetItem(shortProgressList);
        new QListWidgetItem(shortProgressList);
        shortProgressList->setObjectName(QString::fromUtf8("shortProgressList"));
        shortProgressList->setMaximumSize(QSize(16777215, 150));
        shortProgressList->setFont(font1);
        shortProgressList->setStyleSheet(QString::fromUtf8("\n"
"QListWidget {\n"
"    background-color: #2b2b2b;\n"
"    border: 2px solid #3c3f41;\n"
"}\n"
"\n"
"QListWidget::item {\n"
"    padding: 5px;\n"
"    color: #a9b7c6;\n"
"    border-bottom: 1px solid #3c3f41;\n"
"}\n"
"\n"
"QListWidget::item:last-child {\n"
"    border-bottom: none;\n"
"}\n"
"\n"
"QListWidget::item:selected {\n"
"    background-color: #214283;\n"
"    color: #a9b7c6;\n"
"}\n"
"\n"
"QListWidget::item:hover {\n"
"    background-color: #3c3f41;\n"
"}\n"
"\n"
"QListWidget::item:disabled {\n"
"    color: #808080;\n"
"}\n"
"\n"
"QListWidget::item:checked {\n"
"    background-color: #214283;\n"
"}"));

        gridLayout->addWidget(shortProgressList, 1, 2, 1, 1);

        plaxParamProgressList = new QListWidget(widget);
        new QListWidgetItem(plaxParamProgressList);
        new QListWidgetItem(plaxParamProgressList);
        new QListWidgetItem(plaxParamProgressList);
        new QListWidgetItem(plaxParamProgressList);
        new QListWidgetItem(plaxParamProgressList);
        new QListWidgetItem(plaxParamProgressList);
        new QListWidgetItem(plaxParamProgressList);
        new QListWidgetItem(plaxParamProgressList);
        new QListWidgetItem(plaxParamProgressList);
        new QListWidgetItem(plaxParamProgressList);
        plaxParamProgressList->setObjectName(QString::fromUtf8("plaxParamProgressList"));
        plaxParamProgressList->setFont(font1);
        plaxParamProgressList->setStyleSheet(QString::fromUtf8("\n"
"QListWidget {\n"
"    background-color: #2b2b2b;\n"
"    border: 2px solid #3c3f41;\n"
"}\n"
"\n"
"QListWidget::item {\n"
"    padding: 5px;\n"
"    color: #a9b7c6;\n"
"    border-bottom: 1px solid #3c3f41;\n"
"}\n"
"\n"
"QListWidget::item:last-child {\n"
"    border-bottom: none;\n"
"}\n"
"\n"
"QListWidget::item:selected {\n"
"    background-color: #214283;\n"
"    color: #a9b7c6;\n"
"}\n"
"\n"
"QListWidget::item:hover {\n"
"    background-color: #3c3f41;\n"
"}\n"
"\n"
"QListWidget::item:disabled {\n"
"    color: #808080;\n"
"}\n"
"\n"
"QListWidget::item:checked {\n"
"    background-color: #214283;\n"
"}"));

        gridLayout->addWidget(plaxParamProgressList, 2, 1, 1, 1);


        verticalLayout_3->addWidget(widget);

        frame_2 = new QFrame(frame);
        frame_2->setObjectName(QString::fromUtf8("frame_2"));
        frame_2->setMinimumSize(QSize(100, 400));
        frame_2->setFrameShape(QFrame::StyledPanel);
        frame_2->setFrameShadow(QFrame::Raised);
        gridLayout_2 = new QGridLayout(frame_2);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        paramDemoLabel = new QLabel(frame_2);
        paramDemoLabel->setObjectName(QString::fromUtf8("paramDemoLabel"));

        gridLayout_2->addWidget(paramDemoLabel, 0, 0, 1, 1);


        verticalLayout_3->addWidget(frame_2);


        verticalLayout_2->addWidget(frame);


        retranslateUi(ViewProgressWidget);

        QMetaObject::connectSlotsByName(ViewProgressWidget);
    } // setupUi

    void retranslateUi(QWidget *ViewProgressWidget)
    {
        ViewProgressWidget->setWindowTitle(QApplication::translate("ViewProgressWidget", "Form", nullptr));

        const bool __sortingEnabled = apicalProgressList->isSortingEnabled();
        apicalProgressList->setSortingEnabled(false);
        QListWidgetItem *___qlistwidgetitem = apicalProgressList->item(0);
        ___qlistwidgetitem->setText(QApplication::translate("ViewProgressWidget", "A2C", nullptr));
        QListWidgetItem *___qlistwidgetitem1 = apicalProgressList->item(1);
        ___qlistwidgetitem1->setText(QApplication::translate("ViewProgressWidget", "A3C", nullptr));
        QListWidgetItem *___qlistwidgetitem2 = apicalProgressList->item(2);
        ___qlistwidgetitem2->setText(QApplication::translate("ViewProgressWidget", "A4C", nullptr));
        QListWidgetItem *___qlistwidgetitem3 = apicalProgressList->item(3);
        ___qlistwidgetitem3->setText(QApplication::translate("ViewProgressWidget", "A5C", nullptr));
        apicalProgressList->setSortingEnabled(__sortingEnabled);


        const bool __sortingEnabled1 = paramProgressList->isSortingEnabled();
        paramProgressList->setSortingEnabled(false);
        QListWidgetItem *___qlistwidgetitem4 = paramProgressList->item(0);
        ___qlistwidgetitem4->setText(QApplication::translate("ViewProgressWidget", "\344\272\214\345\260\226\347\223\243\350\241\200\346\265\201\346\265\201\351\200\237", nullptr));
        QListWidgetItem *___qlistwidgetitem5 = paramProgressList->item(1);
        ___qlistwidgetitem5->setText(QApplication::translate("ViewProgressWidget", "\350\202\272\345\212\250\350\204\211\347\223\243\346\265\201\351\200\237", nullptr));
        QListWidgetItem *___qlistwidgetitem6 = paramProgressList->item(2);
        ___qlistwidgetitem6->setText(QApplication::translate("ViewProgressWidget", "\344\270\211\345\260\226\347\223\243\345\217\215\346\265\201\346\265\201\351\200\237", nullptr));
        QListWidgetItem *___qlistwidgetitem7 = paramProgressList->item(3);
        ___qlistwidgetitem7->setText(QApplication::translate("ViewProgressWidget", "\344\270\273\345\212\250\350\204\211\347\223\243\346\265\201\351\200\237\345\217\212\351\200\237\345\272\246\346\227\266\351\227\264\347\247\257\345\210\206", nullptr));
        QListWidgetItem *___qlistwidgetitem8 = paramProgressList->item(4);
        ___qlistwidgetitem8->setText(QApplication::translate("ViewProgressWidget", "\344\272\214\345\260\226\347\223\243\345\256\244\351\227\264\351\232\224\347\223\243\347\216\257\351\200\237\345\272\246", nullptr));
        QListWidgetItem *___qlistwidgetitem9 = paramProgressList->item(5);
        ___qlistwidgetitem9->setText(QApplication::translate("ViewProgressWidget", "\344\272\214\345\260\226\347\223\243\344\276\247\345\243\201\347\223\243\347\216\257\351\200\237\345\272\246", nullptr));
        paramProgressList->setSortingEnabled(__sortingEnabled1);


        const bool __sortingEnabled2 = shortProgressList->isSortingEnabled();
        shortProgressList->setSortingEnabled(false);
        QListWidgetItem *___qlistwidgetitem10 = shortProgressList->item(0);
        ___qlistwidgetitem10->setText(QApplication::translate("ViewProgressWidget", "PLAX", nullptr));
        QListWidgetItem *___qlistwidgetitem11 = shortProgressList->item(1);
        ___qlistwidgetitem11->setText(QApplication::translate("ViewProgressWidget", "PSAXA", nullptr));
        QListWidgetItem *___qlistwidgetitem12 = shortProgressList->item(2);
        ___qlistwidgetitem12->setText(QApplication::translate("ViewProgressWidget", "PSAXMV", nullptr));
        QListWidgetItem *___qlistwidgetitem13 = shortProgressList->item(3);
        ___qlistwidgetitem13->setText(QApplication::translate("ViewProgressWidget", "PSAXGV", nullptr));
        QListWidgetItem *___qlistwidgetitem14 = shortProgressList->item(4);
        ___qlistwidgetitem14->setText(QApplication::translate("ViewProgressWidget", "PSAXPM", nullptr));
        shortProgressList->setSortingEnabled(__sortingEnabled2);


        const bool __sortingEnabled3 = plaxParamProgressList->isSortingEnabled();
        plaxParamProgressList->setSortingEnabled(false);
        QListWidgetItem *___qlistwidgetitem15 = plaxParamProgressList->item(0);
        ___qlistwidgetitem15->setText(QApplication::translate("ViewProgressWidget", "\345\256\244\351\227\264\351\232\224\345\216\232\345\272\246", nullptr));
        QListWidgetItem *___qlistwidgetitem16 = plaxParamProgressList->item(1);
        ___qlistwidgetitem16->setText(QApplication::translate("ViewProgressWidget", "\345\267\246\345\256\244\345\206\205\345\276\204", nullptr));
        QListWidgetItem *___qlistwidgetitem17 = plaxParamProgressList->item(2);
        ___qlistwidgetitem17->setText(QApplication::translate("ViewProgressWidget", "\345\267\246\345\256\244\345\220\216\345\243\201\345\216\232\345\272\246", nullptr));
        QListWidgetItem *___qlistwidgetitem18 = plaxParamProgressList->item(3);
        ___qlistwidgetitem18->setText(QApplication::translate("ViewProgressWidget", "\344\270\273\345\212\250\350\204\211\347\223\243\347\216\257\347\233\264\345\276\204", nullptr));
        QListWidgetItem *___qlistwidgetitem19 = plaxParamProgressList->item(4);
        ___qlistwidgetitem19->setText(QApplication::translate("ViewProgressWidget", "\347\252\246\351\203\250\347\233\264\345\276\204", nullptr));
        QListWidgetItem *___qlistwidgetitem20 = plaxParamProgressList->item(5);
        ___qlistwidgetitem20->setText(QApplication::translate("ViewProgressWidget", "\347\252\246\347\256\241\344\272\244\347\225\214\347\233\264\345\276\204", nullptr));
        QListWidgetItem *___qlistwidgetitem21 = plaxParamProgressList->item(6);
        ___qlistwidgetitem21->setText(QApplication::translate("ViewProgressWidget", "\345\215\207\344\270\273\345\212\250\350\204\211\347\233\264\345\276\204", nullptr));
        QListWidgetItem *___qlistwidgetitem22 = plaxParamProgressList->item(7);
        ___qlistwidgetitem22->setText(QApplication::translate("ViewProgressWidget", "\345\267\246\346\210\277\345\211\215\345\220\216\345\276\204", nullptr));
        QListWidgetItem *___qlistwidgetitem23 = plaxParamProgressList->item(8);
        ___qlistwidgetitem23->setText(QApplication::translate("ViewProgressWidget", "\345\267\246\345\256\244\345\256\271\347\247\257", nullptr));
        QListWidgetItem *___qlistwidgetitem24 = plaxParamProgressList->item(9);
        ___qlistwidgetitem24->setText(QApplication::translate("ViewProgressWidget", "\345\260\204\350\241\200\345\210\206\346\225\260", nullptr));
        plaxParamProgressList->setSortingEnabled(__sortingEnabled3);

        paramDemoLabel->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class ViewProgressWidget: public Ui_ViewProgressWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_VIEW_PROGRESS_WIDGET_COPY_H
