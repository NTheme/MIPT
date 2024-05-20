//go:build !solution

package spacecollapse

func CollapseSpaces(input string) string {
	runes := []rune(input)

	for i := 0; i < len(runes); i += 1 {
		if runes[i] == rune('\t') || runes[i] == rune('\n') || runes[i] == rune('\r') {
			runes[i] = rune(' ')
		}
	}
	j := 0
	for i := 0; i < len(runes); i, j = i+1, j+1 {
		for ; i+1 < len(runes) && runes[i] == 32 && runes[i+1] == 32; i += 1 {
		}
		runes[j] = runes[i]
	}
	return string(runes[:j])
}
