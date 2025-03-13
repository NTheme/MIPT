//go:build !solution

package varfmt

import (
	"fmt"
	"strconv"
	"strings"
)

func Sprintf(format string, args ...interface{}) string {
	var result []string

	for i, pos := 0, 0; i < len(format); i++ {
		println(format[i:])
		if format[i] == '{' {
			last := strings.IndexByte(format[i:], '}')
			if last == -1 {
				result = append(result, format[i:])
				break
			}
			if last == 1 {
				result = append(result, fmt.Sprintf("%v", args[pos]))
			} else {
				num, _ := strconv.Atoi(format[i+1 : i+last])
				result = append(result, fmt.Sprintf("%v", args[num]))
			}
			pos++
			i += last
		} else {
			result = append(result, string(format[i]))
		}
	}
	return strings.Join(result, "")
}
