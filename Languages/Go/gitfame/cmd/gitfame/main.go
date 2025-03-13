//go:build !solution

package main

import (
	"gitlab.com/slon/shad-go/gitfame/cmd/common"
	"gitlab.com/slon/shad-go/gitfame/cmd/procdata"
	"gitlab.com/slon/shad-go/gitfame/cmd/procfiles"
)

func main() {
	common.InitFlags()
	procdata.Process(procfiles.Process())
}
