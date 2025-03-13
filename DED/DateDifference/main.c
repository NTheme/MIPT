/*

Copyright (c) NThemeDEV 2022

Program Task:
	Date Detetminer

Module Task:
	main

Created / By:
	22-08-03 / NThemeDEV

Update History / By / { Modifications }:

*/

#include "date.h"

int main(int argc, char *argv[]) {
	Date date1 = { .day = 0, .month = 0, .year = 0 },
		date2 = { .day = 0, .month = 0, .year = 0 };

	printf("---This program calculates difference between 2 dates---\n\n");

	OUTPUT separator = VMDEFAULT;
	for (int findtype = 1; findtype < argc; findtype++) {
		if (argv[findtype][0] == '-') {
			switch (argv[findtype][1]) {
			case 'a':
				separator = VMDASH;
				printf("Date separator received!\n");
				break;

			case 'l':
				separator = VMSLASH;
				printf("Date separator received!\n");
				break;

			case 'p':
				separator = VMSPACE;
				printf("Date separator received!\n");
				break;

			case 't':
				separator = VMSTICK;
				printf("Date separator received!\n");
				break;

			default:
				printf("Unknown separator type!\n");
				break;
			}
		} else if (date1.day == 0) {
			printf("Trying to receive first date from program arguments...");

			if (!readdate(argv[findtype], &date1)) {
				printf("OK!\n");
			} else {
				printf("FAILED!\n");
			}
		} else if (date2.day == 0) {
			printf("Trying to receive second date from program arguments...");

			if (!readdate(argv[findtype], &date2)) {
				printf("OK!\n");
			} else {
				printf("FAILED!\n");
			}
		}
	}

	if (separator == VMDEFAULT) {
		printf("Date separator set to default (DASH)!\n");
	}

	char datestring[MAXSTRING];
	if (date1.day == 0) {
		printf("Type your first date and press ENTER...");

		inputdate1:
		if (fgets(datestring, MAXSTRING, stdin) == NULL) {
			printf("FAILED!\nUnable to read date! Exiting...\n\n");
			return -1;
		}

		if (!readdate(datestring, &date1)) {
			printf("...OK!\n");
		} else {
			printf("...FAILED!\nTry again with typing first date...");
			goto inputdate1;
		}
	}

	if (date2.day == 0) {
		printf("Type your second date and press ENTER...");

		inputdate2:
		if (fgets(datestring, MAXSTRING, stdin) == NULL) {
			printf("FAILED!\nUnable to read date! Exiting...\n\n");
			return -1;
		}

		if (!readdate(datestring, &date2)) {
			printf("...OK!\n");
		} else {
			printf("...FAILED!\nTry again with typing second date...");
			goto inputdate2;
		}
	}

	printf("\nCounting data difference...");
	printdate(differencedatesdate(date1, date2), separator);

	printf("\nCounting difference in days...%Ld\n\nExiting...\n\n", differencedatesdays(date1, date2));

	return 0;
}
