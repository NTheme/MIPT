package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

func assert(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "fetch: %v\n", err)
		os.Exit(1)
	}
}

func main() {
	timeStart := time.Now()

	channels := make(chan string)
	for index := range len(os.Args[1:]) {
		go fetch(os.Args[index+1], channels)
	}

	for range len(os.Args[1:]) {
		fmt.Println(<-channels)
	}

	fmt.Printf("%.2fs elapsed\n", time.Since(timeStart).Seconds())
}

func fetch(url string, channels chan<- string) {
	timeStart := time.Now()

	resp, err := http.Get(url)
	if err != nil {
		channels <- fmt.Sprint(err)
		return
	}
	defer resp.Body.Close()

	nbytes, err := io.Copy(io.Discard, resp.Body)
	assert(err)

	secs := time.Since(timeStart).Seconds()
	channels <- fmt.Sprintf("%.2fs %7d %s", secs, nbytes, url)
}
