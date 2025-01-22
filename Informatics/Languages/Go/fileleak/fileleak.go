//go:build !solution

package fileleak

import (
	"os"
	"path/filepath"
)

type testingT interface {
	Errorf(msg string, args ...interface{})
	Cleanup(func())
}

var path = "/proc/self/fd/"

func readDir() map[string]string {
	links, _ := os.ReadDir(path)
	opened := make(map[string]string)

	for index := range links {
		newLink, _ := os.Readlink(path + links[index].Name())
		newLink, _ = filepath.Abs(newLink)

		opened[links[index].Name()] = newLink
	}

	return opened
}

func VerifyNone(t testingT) {
	before := readDir()

	cleanClosure := func() {
		after := readDir()

		for link := range after {
			if before[link] == "" || before[link] != after[link] {
				t.Errorf("PANIC!")
			}
		}
	}

	t.Cleanup(cleanClosure)
}
