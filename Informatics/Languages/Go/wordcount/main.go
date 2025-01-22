//go:build !solution

package main

import (
	"bufio"
	"fmt"
	"os"
)

func assert(err error) {
	if err != nil {
		panic(err)
	}
}

func main() {
	count := make(map[string]int)

	for i := range len(os.Args[1:]) {
		file, err := os.Open(os.Args[i+1])
		assert(err)
		Scanner := bufio.NewScanner(file)

		for Scanner.Scan() {
			count[Scanner.Text()] += 1
		}
	}

	for key, val := range count {
		if val >= 2 {
			fmt.Printf("%d\t%s\n", val, key)
		}
	}
}
