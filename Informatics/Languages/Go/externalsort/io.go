//go:build !change

package externalsort

import (
	"bytes"
	"fmt"
	"io"
)

type LineReader interface {
	ReadLine() (string, error)
}

type LineWriter interface {
	Write(l string) error
}

type NewLineReader struct {
	reader io.Reader
}

type NewLineWriter struct {
	writer io.Writer
}

func (reader *NewLineReader) ReadLine() (string, error) {
	buf := bytes.NewBufferString("")
	sym := make([]byte, 1)
	for {
		cnt, err := reader.reader.Read(sym)
		if err == io.EOF && cnt > 0 {
			buf.WriteByte(sym[0])
		}
		if err == io.EOF || err != nil || sym[0] == '\n' {
			return buf.String(), err
		}
		buf.WriteByte(sym[0])
	}
}

func (writer *NewLineWriter) Write(line string) error {
	buf := []byte(fmt.Sprintf("%s\n", line))
	for w := 0; w < len(buf); {
		cnt, err := writer.writer.Write(buf[w:])
		if err != nil {
			return err
		}
		w += cnt
	}
	return nil
}
