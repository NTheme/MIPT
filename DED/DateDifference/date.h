/*

Copyright (c) NThemeDEV 2022

Program Task:
    Date Detetminer

Module Task:
    date.h

Created / By:
    22-08-05 / NThemeDEV

Update History / By / { Modifications }:

*/

#pragma once

#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>

//
// Defines
//
#define MAXSTRING 10000
#define SWAPDATES(a, b)   \
    do {                  \
        Date tmp = *a;    \
        *a = *b;          \
        *b = tmp;         \
    } while (0)

//
// Objects
//
typedef enum {
    VMDEFAULT,
    VMDASH = 0,
    VMSLASH,
    VMSPACE,
    VMSTICK
} OUTPUT;

typedef struct {
    int day, month;
    long long year;
} Date;

//
// Definitions
//
int leapyear(const long long);
int printdate(Date, int);
int readdate(const char *, Date *);
long long differencedatesdays(Date, Date);
Date differencedatesdate(Date, Date);
