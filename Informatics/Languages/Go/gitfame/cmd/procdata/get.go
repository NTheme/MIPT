package procdata

import (
	"gitlab.com/slon/shad-go/gitfame/cmd/common"
	"os/exec"
	"strings"
)

func AppendMap(object map[string]map[string]struct{}, key, value string) {
	if object[key] == nil {
		object[key] = make(map[string]struct{})
	}
	object[key][value] = struct{}{}
}

func MakeCommitters(commits map[string]map[string]struct{}, touches map[string]map[string]struct{}, lines map[string]int) []Committer {
	result := make([]Committer, 0, len(commits))
	for authName := range commits {
		result = append(result, Committer{authName, lines[authName], len(commits[authName]), len(touches[authName])})
	}
	return result
}

func BlameSingle(file string, commits map[string]map[string]struct{}, touches map[string]map[string]struct{}) {
	logOutput, err := exec.Command("git", "log", "--format=%H%n%an%n%cn", *common.FlagRevision, "--", file).Output()
	common.Exit(err)

	logLines := strings.Split(string(logOutput), "\n")
	commit := logLines[0]
	authName := logLines[1]
	if *common.FlagUseCommitter {
		authName = logLines[2]
	}
	AppendMap(commits, authName, commit)
	AppendMap(touches, authName, file)
}

func BlameLots(file string, i int, blameLines []string, commits map[string]map[string]struct{}, touches map[string]map[string]struct{}, lines map[string]int) {
	commit := blameLines[i][:40]
	authName := blameLines[i+1][7:]
	if *common.FlagUseCommitter {
		authName = blameLines[i+5][10:]
	}
	lines[authName]++
	AppendMap(commits, authName, commit)
	AppendMap(touches, authName, file)
}

func Get(files []string) []Committer {
	commits := make(map[string]map[string]struct{})
	touches := make(map[string]map[string]struct{})
	lines := make(map[string]int)

	for i := range files {
		blameOutput, err := exec.Command("git", "blame", "--line-porcelain", *common.FlagRevision, files[i]).Output()
		common.Exit(err)

		blameLines := strings.Split(string(blameOutput), "\n")
		if len(blameLines) <= 1 {
			BlameSingle(files[i], commits, touches)
		}

		for j := 0; j+1 < len(blameLines); j += 12 {
			BlameLots(files[i], j, blameLines, commits, touches, lines)
			if strings.HasPrefix(blameLines[j+10], "previous") || strings.HasPrefix(blameLines[j+10], "boundary") {
				j++
			}
		}
	}

	return MakeCommitters(commits, touches, lines)
}
