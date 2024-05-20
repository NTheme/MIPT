package common

import (
	"fmt"
	flag "github.com/spf13/pflag"
	"os"
)

func Exit(err error) {
	if err != nil {
		_, _ = fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

var FlagRepository = flag.String("repository", ".", "Git repository directory")
var FlagRevision = flag.String("revision", "HEAD", "Pointer to last commit")
var FlagOrderBy = flag.String("order-by", "lines", "order values")
var FlagUseCommitter = flag.Bool("use-committer", false, "Use committer or Committer")
var FlagFormat = flag.String("format", "tabular", "Output format")
var FlagExtensions = flag.StringSlice("extensions", nil, "Extensions")
var FlagLanguages = flag.StringSlice("languages", nil, "Languages")
var FlagExclude = flag.StringSlice("exclude", nil, "GExcluded patterns")
var FlagRestrictTo = flag.StringSlice("restrict-to", nil, "Restricted files")

func InitFlags() {
	flag.Parse()

	switch *FlagOrderBy {
	case "lines", "commits", "files":
	default:
		Exit(fmt.Errorf("flag error"))
	}
	switch *FlagFormat {
	case "tabular", "csv", "json", "json-lines":
	default:
		Exit(fmt.Errorf("flag error"))
	}
}
