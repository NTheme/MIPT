package procfiles

import (
	"gitlab.com/slon/shad-go/gitfame/cmd/common"
	"path/filepath"
	"strings"
)

func ExtensionFilter(str string, extensions []string) bool {
	if extensions == nil {
		return true
	}
	for i := range extensions {
		if strings.HasSuffix(str, extensions[i]) {
			return true
		}
	}
	return false
}

func ExcludeFilter(str string, exclude []string) bool {
	if exclude == nil {
		return true
	}
	for i := range exclude {
		match, err := filepath.Match(exclude[i], str)
		common.Exit(err)
		if match {
			return false
		}
	}
	return true
}

func RestrictToFilter(str string, restrict []string) bool {
	if restrict == nil {
		return true
	}
	for i := range restrict {
		match, err := filepath.Match(restrict[i], str)
		common.Exit(err)
		if match {
			return true
		}
	}
	return false
}

func FilesFilter(files *[]string, extensions []string, compare func(string, []string) bool) {
	newFiles := make([]string, 0, len(*files))
	for i := range *files {
		match := compare((*files)[i], extensions)

		if match {
			newFiles = append(newFiles, (*files)[i])
		}
	}
	*files = newFiles
}
