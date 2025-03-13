package procdata

import (
	"gitlab.com/slon/shad-go/gitfame/cmd/common"
	"sort"
)

type Committer struct {
	Name    string `json:"name"`
	Lines   int    `json:"lines"`
	Commits int    `json:"commits"`
	Touches int    `json:"files"`
}

func Sort(data []Committer) {
	switch *common.FlagOrderBy {
	case "lines":
		sort.Slice(data, func(i, j int) bool {
			return LinesComparator(data[i], data[j])
		})
	case "commits":
		sort.Slice(data, func(i, j int) bool {
			return CommitsComparator(data[i], data[j])
		})
	case "files":
		sort.Slice(data, func(i, j int) bool {
			return FilesComparator(data[i], data[j])
		})
	}
}

func Process(files []string) {
	data := Get(files)
	Sort(data)
	Format(data)
}
