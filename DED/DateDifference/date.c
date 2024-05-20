/*

Copyright (c) NThemeDEV 2022

Program Task:
    Date Detetminer

Module Task:
    date.c

Created / By:
    22-08-05 / NThemeDEV

Update History / By / { Modifications }:

*/

#include "date.h"

//
// Constants
//
const int MONTHS[] = { 31, 28, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
const char *DATETYPES[] = { "%d-%d-%Ld", "%d/%d/%Ld", "%d %d %Ld", "%d|%d|%Ld" };

//
// Functions
//
int leapyear(const long long year) {
    return ((year % 4 == 0 && year % 100 != 0) || year % 400 == 0);
}

int readdate(const char *string, Date *result) {
    int day = 0, month = 0;
    long long year = 0;

    for (int i = 0; sscanf(string, DATETYPES[i], &day, &month, &year) != 3; i++);

    if (month > 12 || month < 1)
        return -1;
    if (day < 0 || day > MONTHS[month - 1 + (month > 2 || (month == 2 && leapyear(year)))])
        return -1;

    result->day = day;
    result->month = month;
    result->year = year;

    return 0;
}

int printdate(Date date, int type) {
    if (type < VMDASH && type > VMSTICK) {
        printf("Unknown output type!\n");
        return 1;
    }

    printf(DATETYPES[type], date.day, date.month, date.year);
    return 0;
}

int comparedates(const Date *date1, const Date *date2) {
    if (date1->year > date2->year)
        return 1;
    if (date1->year < date2->year)
        return -1;
    if (date1->month > date2->month)
        return 1;
    if (date1->month < date2->month)
        return -1;
    if (date1->day > date2->day)
        return 1;
    if (date1->day < date2->day)
        return -1;
    return 0;
}

int differenceinsideyear(Date date1, Date date2) {
    int difference = 0;
    if (date2.month == date1.month) {
        difference = date2.day - date1.day;
    } else {
        for (int current = date1.month - 1; current < date2.month - 1; current++) {
            difference += MONTHS[current + (current > 1 || (current == 1 && leapyear(date1.year)))];
        }

        difference += date2.day - date1.day;
    }

    return difference;
}

long long differencedatesdays(Date date1, Date date2) {
    int ind = 1;
    if (comparedates(&date1, &date2) == 1) {
        SWAPDATES(&date1, &date2);
        ind = -1;
    }

    long long difference = 0;
    if (date2.year == date1.year) {
        difference = differenceinsideyear(date1, date2);
    } else {
        Date firstyear = { .day = 31, .month = 12, .year = date1.year },
            secondyear = { .day = 1, .month = 1, .year = date2.year };

        difference = ((date2.year - 1) - date1.year) * 365 + ((date2.year - 1) / 4 - date1.year / 4) -
            ((date2.year - 1) / 100 - date1.year / 100) + ((date2.year - 1) / 400 - date1.year / 400);

        difference += differenceinsideyear(date1, firstyear) + differenceinsideyear(secondyear, date2) + 1;
    }

    return difference * ind;
}

Date differencedatesdate(Date date1, Date date2) {
    Date difference = { .day = 0, .month = 0, .year = 0 };

    if (comparedates(&date1, &date2) == 1) {
        SWAPDATES(&date1, &date2);
    }

    difference.year = date2.year - date1.year;
    date1.year = date2.year;

    if (comparedates(&date1, &date2) == 1) {
        --difference.year;
        --date1.year;
    }

    difference.month = date2.month - date1.month;
    date1.month = date2.month;

    if (date1.day > date2.day) {
        --difference.month;
        if (--date1.month == 0) {
            date1.month = 12;
        }
    }

    if (difference.month < 0)
        difference.month += 12;

    difference.day = date2.day - date1.day;
    if (difference.day < 0) {
        difference.day += MONTHS[date1.month - 1 + (date1.month > 2 || (date1.month == 2 && leapyear(date1.year)))];
    }

    return difference;
}
