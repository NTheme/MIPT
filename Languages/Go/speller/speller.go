//go:build !solution

package speller

import (
	"strings"
)

var units = []string{
	"zero", "one", "two", "three", "four",
	"five", "six", "seven", "eight", "nine",
	"ten", "eleven", "twelve", "thirteen", "fourteen",
	"fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
}

var tens = []string{
	"", "", "twenty", "thirty", "forty",
	"fifty", "sixty", "seventy", "eighty", "ninety",
}

var thousands = []string{
	"", "thousand", "million", "billion",
}

func Spell(n int64) string {
	if n == 0 {
		return units[0]
	}
	if n < 0 {
		return "minus " + Spell(-n)
	}

	var parts []string
	for i := 0; n > 0; i++ {
		if n%1000 != 0 {
			if len(thousands[i]) > 0 {
				parts = append([]string{thousands[i]}, parts...)
			}
			parts = append([]string{SpellThousands(n % 1000)}, parts...)
		}
		n /= 1000
	}
	return strings.Join(parts, " ")
}

func SpellThousands(n int64) string {
	var parts []string
	if n >= 100 {
		parts = append(parts, units[n/100], "hundred")
		n %= 100
	}
	if n >= 20 {
		var add []string
		add = append(add, tens[n/10])
		if n%10 > 0 {
			add = append(add, units[n%10])
		}
		parts = append(parts, strings.Join(add, "-"))
	} else if n > 0 {
		parts = append(parts, units[n])
	}
	return strings.Join(parts, " ")
}
