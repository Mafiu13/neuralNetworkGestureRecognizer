TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    neuralnetwork.cpp \
    mainview.cpp \
    nearualnetworkmanager.cpp \
    mainmenutext.cpp

HEADERS += \
    neuralnetwork.h \
    mainview.h \
    nearualnetworkmanager.h \
    mainmenutext.h \
    neuralnetworkconst.h

