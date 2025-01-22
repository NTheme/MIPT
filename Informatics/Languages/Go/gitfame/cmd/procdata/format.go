package procdata

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"gitlab.com/slon/shad-go/gitfame/cmd/common"
	"os"
	"strconv"
	"text/tabwriter"
)

func TabularFormat(data []Committer) {
	writer := tabwriter.NewWriter(os.Stdout, 0, 1, 1, ' ', 0)
	_, err := fmt.Fprintln(writer, "Name\tLines\tCommits\tFiles")
	common.Exit(err)

	for i := range data {
		_, err := fmt.Fprintf(writer, "%s\t%d\t%d\t%d\n", data[i].Name, data[i].Lines, data[i].Commits, data[i].Touches)
		common.Exit(err)
	}
	common.Exit(writer.Flush())
}

func CSVFormat(data []Committer) {
	writer := csv.NewWriter(os.Stdout)
	common.Exit(writer.Write([]string{"Name", "Lines", "Commits", "Files"}))

	for i := range data {
		common.Exit(writer.Write([]string{data[i].Name, strconv.Itoa(data[i].Lines), strconv.Itoa(data[i].Commits), strconv.Itoa(data[i].Touches)}))
	}
	writer.Flush()
}

func JSONFormat(data []Committer) {
	output, err := json.Marshal(data)
	common.Exit(err)
	fmt.Println(string(output))
}

func JSONLinesFormat(data []Committer) {
	for i := range data {
		output, err := json.Marshal(data[i])
		common.Exit(err)
		fmt.Println(string(output))
	}
}

func Format(data []Committer) {
	switch *common.FlagFormat {
	case "tabular":
		TabularFormat(data)
	case "csv":
		CSVFormat(data)
	case "json":
		JSONFormat(data)
	case "json-lines":
		JSONLinesFormat(data)
	}
}
