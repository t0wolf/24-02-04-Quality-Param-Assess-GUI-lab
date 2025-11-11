#include "PatientManagerWindow.h"

PatientManagerWindow::PatientManagerWindow(QWidget* parent)
    : QDialog(parent), m_patientListWidget(new QListWidget), infoLabel(new QLabel), m_bIsPatientSelected(false) {

    // 设置布局
    //layout = new QVBoxLayout;
    //layout->addWidget(m_patientListWidget);
    //layout->addWidget(infoLabel);
    //setLayout(layout);

    setupUI();
    applyDarkStyle();
    // 连接信号和槽
    connect(m_patientListWidget, &QListWidget::itemClicked, this, &PatientManagerWindow::on_patientList_itemClicked);

    // 创建病人历史数据目录
    m_patientHistDir = QDir(QDir::homePath() + "/patient_hist");
    if (!m_patientHistDir.exists()) {
        m_patientHistDir.mkpath("."); // 创建 patient_hist 目录
    }
}

void PatientManagerWindow::applyDarkStyle() {
    // 设置整体背景颜色
    QPalette palette = this->palette();
    palette.setColor(QPalette::Window, QColor(45, 45, 45));
    palette.setColor(QPalette::WindowText, Qt::white);
    palette.setColor(QPalette::Base, QColor(60, 60, 60));
    palette.setColor(QPalette::Text, Qt::white);
    setPalette(palette);

    // 设置按钮样式
    QString buttonStyle = "QPushButton {"
        "background-color: #555;"
        "color: white;"
        "border: 2px solid #888;"
        "border-radius: 5px;"
        "padding: 5px;"
        "}"
        "QPushButton:hover {"
        "background-color: #777;"
        "}"
        "QPushButton:pressed {"
        "background-color: #999;"
        "}";

    foreach(QPushButton * button, findChildren<QPushButton*>()) {
        button->setStyleSheet(buttonStyle);
    }

    // 设置列表样式
    m_patientListWidget->setStyleSheet("QListWidget {"
        "background-color: #333;"
        "color: white;"
        "border: 1px solid #444;"
        "border-radius: 5px;"
        "padding: 5px;"
        "}"
        "QListWidget::item {"
        "padding: 10px;"
        "}"
        "QListWidget::item:selected {"
        "background-color: #555;"
        "}");

    // 设置输入框样式
    searchInput->setStyleSheet("QLineEdit {"
        "background-color: #444;"
        "color: white;"
        "border: 1px solid #666;"
        "border-radius: 5px;"
        "padding: 5px;"
        "}");
}

void PatientManagerWindow::addPatient(const QString& name, const QString& id) {
    // 检查是否已经存在该病人
    for (int i = 0; i < m_patientListWidget->count(); ++i) {
        if (m_patientListWidget->item(i)->text().contains(id)) {
            return; // 如果病人已存在，则不做任何操作
        }
    }

    // 添加新病人到列表
    m_patientListWidget->addItem(name + " (ID: " + id + ")");
    m_currentPatient.name = name;
    m_currentPatient.id = id;
    m_currentPatient.measurements.clear(); // 清空当前病人的数据
    m_bIsPatientSelected = true; // 标记病人已被选择

    // 显示病人信息
    //displayPatientInfo();
}

void PatientManagerWindow::updatePatientMeasurement(const QString& parameterName, float value) {
    if (!m_bIsPatientSelected) return; // 如果没有选择病人，则不执行更新

    // 增量更新病人数据
    m_currentPatient.measurements[parameterName].append(value);

    saveCurrentPatientData();

    // 将更新的信息显示在 infoLabel 中
    //displayPatientInfo();
}

void PatientManagerWindow::switchPatient(const QString& patientId) {
    // 保存当前病人的数据
    saveCurrentPatientData();

    // 查找并切换到新病人
    for (int i = 0; i < m_patientListWidget->count(); ++i) {
        if (m_patientListWidget->item(i)->text().contains(patientId)) {
            QString selectedPatientText = m_patientListWidget->item(i)->text();
            QStringList patientInfo = selectedPatientText.split(" (ID: ");
            QString name = patientInfo[0];
            QString id = patientInfo[1].chopped(1); // 去掉末尾的 ')'

            // 切换到新病人
            m_currentPatient.name = name;
            m_currentPatient.id = id;
            m_currentPatient.measurements.clear(); // 清空当前病人的数据
            m_bIsPatientSelected = true; // 标记病人已被选择

            //displayPatientInfo(); // 显示新病人的信息
            return;
        }
    }
}

void PatientManagerWindow::addNewPatient()
{
    bool ok;
    QString name = QInputDialog::getText(this, "添加病人", "输入病人名称:", QLineEdit::Normal, "", &ok);
    if (ok && !name.isEmpty()) {
        QString patientId = QString::number(qrand() % 100000); // 模拟生成一个随机ID
        QString patientInfo = name + " (ID: " + patientId + ")";
        m_patientListWidget->addItem(patientInfo);

        // 保存到文件
        QFile file(patientId + ".txt"); // 可以根据需要调整文件保存路径
        if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QTextStream out(&file);
            out << "Measurements:\n"; // 这里可以根据实际情况保存更多信息
            file.close();
        }
    }
}

void PatientManagerWindow::deletePatient()
{
    QListWidgetItem* selectedItem = m_patientListWidget->currentItem();
    if (selectedItem) {
        QString patientId = selectedItem->text().section(' ', 3, 3).chopped(1).remove(')'); // 提取 ID
        QMessageBox::StandardButton reply;
        reply = QMessageBox::question(this, "删除病人", "是否确认删除该病人?",
            QMessageBox::Yes | QMessageBox::No);
        if (reply == QMessageBox::Yes) {
            // 删除文件
            QFile file(patientId + ".txt"); // 可以根据需要调整文件路径
            if (file.exists()) {
                file.remove();
            }
            delete selectedItem; // 从列表中删除
        }
    }
    else {
        QMessageBox::warning(this, "删除病人", "请选择需要删除的病人.");
    }
}

void PatientManagerWindow::searchPatient()
{
    QString searchTerm = searchInput->text();
    if (searchTerm.isEmpty()) {
        QMessageBox::warning(this, "Search Patient", "Please enter a name or ID to search.");
        return;
    }

    bool found = false;
    for (int i = 0; i < m_patientListWidget->count(); ++i) {
        QListWidgetItem* item = m_patientListWidget->item(i);
        if (item->text().contains(searchTerm, Qt::CaseInsensitive)) {
            m_patientListWidget->setCurrentItem(item); // 高亮显示找到的项
            found = true;
            break;
        }
    }

    if (!found) {
        QMessageBox::information(this, "Search Patient", "No matching patient found.");
    }
}

void PatientManagerWindow::setupUI()
{
    QVBoxLayout* layout = new QVBoxLayout(this);

    m_patientListWidget = new QListWidget(this);
    infoLabel = new QLabel(this);
    layout->addWidget(m_patientListWidget);
    layout->addWidget(infoLabel);

    // 增加病人按钮
    //QPushButton* addButton = new QPushButton("添加病人", this);
    //layout->addWidget(addButton);
    //connect(addButton, &QPushButton::clicked, this, &PatientManagerWindow::addNewPatient);

    // 删除病人按钮
    QPushButton* deleteButton = new QPushButton("删除病人", this);
    layout->addWidget(deleteButton);
    connect(deleteButton, &QPushButton::clicked, this, &PatientManagerWindow::deletePatient);

    // 查找病人按钮
    QPushButton* searchButton = new QPushButton("搜索病人", this);
    layout->addWidget(searchButton);
    connect(searchButton, &QPushButton::clicked, this, &PatientManagerWindow::searchPatient);

    // 输入框用于查找病人
    searchInput = new QLineEdit(this);
    searchInput->setPlaceholderText("请在此输入要搜索的病人ID");
    layout->addWidget(searchInput);
}

void PatientManagerWindow::saveCurrentPatientData() {
    if (!m_bIsPatientSelected) return; // 如果没有选择病人，则不执行保存

    QFile file(m_patientHistDir.filePath(m_currentPatient.id + ".txt"));
    if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream out(&file);
        out << m_currentPatient.name << "\n"; // 写入病人姓名
        for (auto it = m_currentPatient.measurements.constBegin(); it != m_currentPatient.measurements.constEnd(); ++it) {
            out << it.key() << ": "; // 写入参数名称
            for (double value : it.value()) {
                out << value << " "; // 写入每个测量参数的值
            }
            out << "\n"; // 换行
        }
        file.close();
    }
}

void PatientManagerWindow::displayPatientInfo() {
    QString info = "Name: " + m_currentPatient.name + "\nID: " + m_currentPatient.id + "\nMeasurements:\n";
    for (auto it = m_currentPatient.measurements.constBegin(); it != m_currentPatient.measurements.constEnd(); ++it) {
        info += it.key() + ": "; // 参数名称
        for (double value : it.value()) {
            info += QString::number(value) + " "; // 追加每个测量参数的值
        }
        info += "\n"; // 换行
    }
    infoLabel->setText(info); // 显示病人信息
}

void PatientManagerWindow::on_patientList_itemClicked(QListWidgetItem* item) 
{
    QString patientName = item->text().section(' ', 0, 1);
    QString patientId = item->text().section(' ', 3, 3).chopped(1).remove(')'); // 获取病人 ID

    // 加载病人的测量信息，显示在 infoLabel 中
    QFile file(m_patientHistDir.filePath(patientId + ".txt"));
    if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QString info = "历史测值: " + patientName + ":\n";
        QTextStream in(&file);
        QString line;
        while (in.readLineInto(&line)) {
            info += line + "\n"; // 逐行读取并拼接
        }
        infoLabel->setText(info); // 更新 infoLabel 显示内容
        file.close();
    }
    else {
        infoLabel->setText("无法读取该患者历史测值数据."); // 显示错误信息
    }
}