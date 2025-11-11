/****************************************************************************
** Meta object code from reading C++ file 'param_display_widget.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../../../param_display_widget.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'param_display_widget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_ParamDisplayWidget_t {
    QByteArrayData data[14];
    char stringdata0[168];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_ParamDisplayWidget_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_ParamDisplayWidget_t qt_meta_stringdata_ParamDisplayWidget = {
    {
QT_MOC_LITERAL(0, 0, 18), // "ParamDisplayWidget"
QT_MOC_LITERAL(1, 19, 14), // "setParamValues"
QT_MOC_LITERAL(2, 34, 0), // ""
QT_MOC_LITERAL(3, 35, 8), // "viewName"
QT_MOC_LITERAL(4, 44, 4), // "qVar"
QT_MOC_LITERAL(5, 49, 18), // "setParamValuesPics"
QT_MOC_LITERAL(6, 68, 7), // "qValues"
QT_MOC_LITERAL(7, 76, 9), // "qPremiums"
QT_MOC_LITERAL(8, 86, 20), // "setParamValueDeleted"
QT_MOC_LITERAL(9, 107, 9), // "paramName"
QT_MOC_LITERAL(10, 117, 13), // "onItemPressed"
QT_MOC_LITERAL(11, 131, 16), // "QListWidgetItem*"
QT_MOC_LITERAL(12, 148, 4), // "item"
QT_MOC_LITERAL(13, 153, 14) // "onItemReleased"

    },
    "ParamDisplayWidget\0setParamValues\0\0"
    "viewName\0qVar\0setParamValuesPics\0"
    "qValues\0qPremiums\0setParamValueDeleted\0"
    "paramName\0onItemPressed\0QListWidgetItem*\0"
    "item\0onItemReleased"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_ParamDisplayWidget[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    2,   39,    2, 0x0a /* Public */,
       5,    3,   44,    2, 0x0a /* Public */,
       8,    1,   51,    2, 0x0a /* Public */,
      10,    1,   54,    2, 0x0a /* Public */,
      13,    1,   57,    2, 0x0a /* Public */,

 // slots: parameters
    QMetaType::Void, QMetaType::QString, QMetaType::QVariant,    3,    4,
    QMetaType::Void, QMetaType::QString, QMetaType::QVariant, QMetaType::QVariant,    3,    6,    7,
    QMetaType::Void, QMetaType::QString,    9,
    QMetaType::Void, 0x80000000 | 11,   12,
    QMetaType::Void, 0x80000000 | 11,   12,

       0        // eod
};

void ParamDisplayWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<ParamDisplayWidget *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->setParamValues((*reinterpret_cast< const QString(*)>(_a[1])),(*reinterpret_cast< QVariant(*)>(_a[2]))); break;
        case 1: _t->setParamValuesPics((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< QVariant(*)>(_a[2])),(*reinterpret_cast< QVariant(*)>(_a[3]))); break;
        case 2: _t->setParamValueDeleted((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 3: _t->onItemPressed((*reinterpret_cast< QListWidgetItem*(*)>(_a[1]))); break;
        case 4: _t->onItemReleased((*reinterpret_cast< QListWidgetItem*(*)>(_a[1]))); break;
        default: ;
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject ParamDisplayWidget::staticMetaObject = { {
    QMetaObject::SuperData::link<QWidget::staticMetaObject>(),
    qt_meta_stringdata_ParamDisplayWidget.data,
    qt_meta_data_ParamDisplayWidget,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *ParamDisplayWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *ParamDisplayWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_ParamDisplayWidget.stringdata0))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int ParamDisplayWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 5)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 5;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 5)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 5;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
