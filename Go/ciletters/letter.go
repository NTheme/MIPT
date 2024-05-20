//go:build !solution

package ciletters

import (
	"bytes"
	_ "embed"
	_ "strings"
	"text/template"
)

//go:embed ntemp.tmpl
var notificationTemplate string

func pHash(hash string) string {
	return hash[:8]
}

func pLog(log string) []string {
	cnt := 0
	last := len(log)
	var strs []string

	for i := len(log) - 1; i >= 0; i-- {
		if log[i] == '\n' || i == 0 {
			if i == 0 {
				i--
			}
			cnt++
			strs = append([]string{log[i+1 : last]}, strs...)
			last = i
			print(len(strs))
		}
		if cnt == 10 {
			break
		}
	}

	return strs
}

// MakeLetter generates text representation of notification using a template
func MakeLetter(notification *Notification) (string, error) {
	funcMap := template.FuncMap{
		"commit": pHash,
		"log":    pLog,
	}

	tmpl, err := template.New("notification").Funcs(funcMap).Parse(notificationTemplate)
	if err != nil {
		return "", err
	}

	var tpl bytes.Buffer
	if err := tmpl.Execute(&tpl, notification); err != nil {
		return "", err
	}

	return tpl.String(), nil
}
