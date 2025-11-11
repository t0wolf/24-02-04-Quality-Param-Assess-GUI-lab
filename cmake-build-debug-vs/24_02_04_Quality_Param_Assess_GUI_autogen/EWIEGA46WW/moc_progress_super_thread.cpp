/****************************************************************************
** Meta object code from reading C++ file 'progress_super_thread.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../../../progress_super_thread.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'progress_super_thread.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_ProgressSuperThread_t {
    QByteArrayData data[28];
    char stringdata0[468];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_ProgressSuperThread_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_ProgressSuperThread_t qt_meta_stringdata_ProgressSuperThread = {
    {
QT_MOC_LITERAL(0, 0, 19), // "ProgressSuperThread"
QT_MOC_LITERAL(1, 20, 25), // "uiProgressUpdateAvailable"
QT_MOC_LITERAL(2, 46, 0), // ""
QT_MOC_LITERAL(3, 47, 4), // "name"
QT_MOC_LITERAL(4, 52, 22), // "viewNameImageAvailable"
QT_MOC_LITERAL(5, 75, 8), // "viewName"
QT_MOC_LITERAL(6, 84, 6), // "qImage"
QT_MOC_LITERAL(7, 91, 20), // "paramValuesAvailable"
QT_MOC_LITERAL(8, 112, 4), // "qVar"
QT_MOC_LITERAL(9, 117, 30), // "sigParamValuePremiumsAvailable"
QT_MOC_LITERAL(10, 148, 7), // "qValues"
QT_MOC_LITERAL(11, 156, 8), // "qPremium"
QT_MOC_LITERAL(12, 165, 20), // "setProgressMapUpdate"
QT_MOC_LITERAL(13, 186, 18), // "setCurrentViewName"
QT_MOC_LITERAL(14, 205, 23), // "setCurrentViewNameImage"
QT_MOC_LITERAL(15, 229, 23), // "setCurrentViewNameVideo"
QT_MOC_LITERAL(16, 253, 15), // "setCurrentParam"
QT_MOC_LITERAL(17, 269, 24), // "setCurrentParamValuePics"
QT_MOC_LITERAL(18, 294, 16), // "isKeyframeEnable"
QT_MOC_LITERAL(19, 311, 11), // "std::string"
QT_MOC_LITERAL(20, 323, 19), // "isQualityControlled"
QT_MOC_LITERAL(21, 343, 13), // "singleResult&"
QT_MOC_LITERAL(22, 357, 11), // "videoResult"
QT_MOC_LITERAL(23, 369, 16), // "setParamComplete"
QT_MOC_LITERAL(24, 386, 14), // "paramEventName"
QT_MOC_LITERAL(25, 401, 18), // "setParamUncomplete"
QT_MOC_LITERAL(26, 420, 22), // "setViewQualityComplete"
QT_MOC_LITERAL(27, 443, 24) // "setViewQualityUncomplete"

    },
    "ProgressSuperThread\0uiProgressUpdateAvailable\0"
    "\0name\0viewNameImageAvailable\0viewName\0"
    "qImage\0paramValuesAvailable\0qVar\0"
    "sigParamValuePremiumsAvailable\0qValues\0"
    "qPremium\0setProgressMapUpdate\0"
    "setCurrentViewName\0setCurrentViewNameImage\0"
    "setCurrentViewNameVideo\0setCurrentParam\0"
    "setCurrentParamValuePics\0isKeyframeEnable\0"
    "std::string\0isQualityControlled\0"
    "singleResult&\0videoResult\0setParamComplete\0"
    "paramEventName\0setParamUncomplete\0"
    "setViewQualityComplete\0setViewQualityUncomplete"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_ProgressSuperThread[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
      16,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       4,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   94,    2, 0x06 /* Public */,
       4,    2,   97,    2, 0x06 /* Public */,
       7,    2,  102,    2, 0x06 /* Public */,
       9,    3,  107,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
      12,    1,  114,    2, 0x0a /* Public */,
      13,    1,  117,    2, 0x0a /* Public */,
      14,    2,  120,    2, 0x0a /* Public */,
      15,    2,  125,    2, 0x0a /* Public */,
      16,    2,  130,    2, 0x0a /* Public */,
      17,    3,  135,    2, 0x0a /* Public */,
      18,    1,  142,    2, 0x0a /* Public */,
      20,    1,  145,    2, 0x0a /* Public */,
      23,    1,  148,    2, 0x0a /* Public */,
      25,    1,  151,    2, 0x0a /* Public */,
      26,    1,  154,    2, 0x0a /* Public */,
      27,    1,  157,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::QString,    3,
    QMetaType::Void, QMetaType::QString, QMetaType::QImage,    5,    6,
    QMetaType::Void, QMetaType::QString, QMetaType::QVariant,    5,    8,
    QMetaType::Void, QMetaType::QString, QMetaType::QVariant, QMetaType::QVariant,    5,   10,   11,

 // slots: parameters
    QMetaType::Void, QMetaType::QString,    3,
    QMetaType::Void, QMetaType::QString,    5,
    QMetaType::Void, QMetaType::QString, QMetaType::QImage,    5,    6,
    QMetaType::Void, QMetaType::QString, QMetaType::QVariant,    5,    8,
    QMetaType::Void, QMetaType::QString, QMetaType::QVariant,    5,    8,
    QMetaType::Void, QMetaType::QString, QMetaType::QVariant, QMetaType::QVariant,    5,   10,   11,
    QMetaType::Bool, 0x80000000 | 19,    5,
    QMetaType::Bool, 0x80000000 | 21,   22,
    QMetaType::Void, QMetaType::QString,   24,
    QMetaType::Void, QMetaType::QString,   24,
    QMetaType::Void, QMetaType::QString,    5,
    QMetaType::Void, QMetaType::QString,    5,

       0        // eod
};

void ProgressSuperThread::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<ProgressSuperThread *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->uiProgressUpdateAvailable((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 1: _t->viewNameImageAvailable((*reinterpret_cast< const QString(*)>(_a[1])),(*reinterpret_cast< QImage(*)>(_a[2]))); break;
        case 2: _t->paramValuesAvailable((*reinterpret_cast< const QString(*)>(_a[1])),(*reinterpret_cast< QVariant(*)>(_a[2]))); break;
        case 3: _t->sigParamValuePremiumsAvailable((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< QVariant(*)>(_a[2])),(*reinterpret_cast< QVariant(*)>(_a[3]))); break;
        case 4: _t->setProgressMapUpdate((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 5: _t->setCurrentViewName((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 6: _t->setCurrentViewNameImage((*reinterpret_cast< const QString(*)>(_a[1])),(*reinterpret_cast< QImage(*)>(_a[2]))); break;
        case 7: _t->setCurrentViewNameVideo((*reinterpret_cast< const QString(*)>(_a[1])),(*reinterpret_cast< QVariant(*)>(_a[2]))); break;
        case 8: _t->setCurrentParam((*reinterpret_cast< const QString(*)>(_a[1])),(*reinterpret_cast< QVariant(*)>(_a[2]))); break;
        case 9: _t->setCurrentParamValuePics((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< QVariant(*)>(_a[2])),(*reinterpret_cast< QVariant(*)>(_a[3]))); break;
        case 10: { bool _r = _t->isKeyframeEnable((*reinterpret_cast< std::string(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = std::move(_r); }  break;
        case 11: { bool _r = _t->isQualityControlled((*reinterpret_cast< singleResult(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = std::move(_r); }  break;
        case 12: _t->setParamComplete((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 13: _t->setParamUncomplete((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 14: _t->setViewQualityComplete((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 15: _t->setViewQualityUncomplete((*reinterpret_cast< QString(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (ProgressSuperThread::*)(const QString );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ProgressSuperThread::uiProgressUpdateAvailable)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (ProgressSuperThread::*)(const QString , QImage );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ProgressSuperThread::viewNameImageAvailable)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (ProgressSuperThread::*)(const QString , QVariant );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ProgressSuperThread::paramValuesAvailable)) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (ProgressSuperThread::*)(QString , QVariant , QVariant );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ProgressSuperThread::sigParamValuePremiumsAvailable)) {
                *result = 3;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject ProgressSuperThread::staticMetaObject = { {
    QMetaObject::SuperData::link<QThread::staticMetaObject>(),
    qt_meta_stringdata_ProgressSuperThread.data,
    qt_meta_data_ProgressSuperThread,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *ProgressSuperThread::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *ProgressSuperThread::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_ProgressSuperThread.stringdata0))
        return static_cast<void*>(this);
    return QThread::qt_metacast(_clname);
}

int ProgressSuperThread::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QThread::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 16)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 16;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 16)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 16;
    }
    return _id;
}

// SIGNAL 0
void ProgressSuperThread::uiProgressUpdateAvailable(const QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void ProgressSuperThread::viewNameImageAvailable(const QString _t1, QImage _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void ProgressSuperThread::paramValuesAvailable(const QString _t1, QVariant _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void ProgressSuperThread::sigParamValuePremiumsAvailable(QString _t1, QVariant _t2, QVariant _t3)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t3))) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
