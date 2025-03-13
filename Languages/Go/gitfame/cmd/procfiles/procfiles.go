package procfiles

import (
	"encoding/json"
	"gitlab.com/slon/shad-go/gitfame/cmd/common"
	"gitlab.com/slon/shad-go/gitfame/configs"
	"os"
	"os/exec"
	"strings"
)

type Language struct {
	Name       string   `json:"name"`
	Type       string   `json:"type"`
	Extensions []string `json:"extensions"`
}

var Languages []Language

func GetExtensions() (extensions []string) {
	for i := range *common.FlagLanguages {
		for j := range Languages {
			if strings.EqualFold(Languages[j].Name, (*common.FlagLanguages)[i]) {
				if extensions == nil {
					extensions = make([]string, 0, 5)
				}
				extensions = append(extensions, Languages[j].Extensions...)
			}
		}
	}
	return extensions
}

func Process() (files []string) {
	common.Exit(json.Unmarshal(configs.LanguageMappingFile, &Languages))

	common.Exit(os.Chdir(*common.FlagRepository))
	output, err := exec.Command("git", "ls-tree", "-r", "--name-only", *common.FlagRevision, *common.FlagRepository).Output()
	common.Exit(err)

	files = strings.Split(string(output), "\n")
	files = files[:len(files)-1]

	FilesFilter(&files, *common.FlagExtensions, ExtensionFilter)
	extensions := GetExtensions()

	FilesFilter(&files, extensions, ExtensionFilter)
	FilesFilter(&files, *common.FlagExclude, ExcludeFilter)
	FilesFilter(&files, *common.FlagRestrictTo, RestrictToFilter)
	return
}
