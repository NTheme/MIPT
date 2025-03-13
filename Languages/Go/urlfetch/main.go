package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
)

func assert(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "fetch: %v\n", err)
		os.Exit(1)
	}
}

func main() {
	for index := range len(os.Args[1:]) {
		resp, err := http.Get(os.Args[index+1])
		assert(err)

		body, err := io.ReadAll(resp.Body)
		assert(err)

		fmt.Printf("%s", body)
		resp.Body.Close()
	}
}
