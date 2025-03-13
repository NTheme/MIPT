//go:build !solution

package otp

import (
	"io"
)

type cipherReader struct {
	r    io.Reader
	prng io.Reader
}

type cipherWriter struct {
	w    io.Writer
	prng io.Reader
}

func (c *cipherReader) Read(p []byte) (n int, err error) {
	n, err = c.r.Read(p)
	buf := make([]byte, n)
	c.prng.Read(buf)

	for i := 0; i < n; i++ {
		p[i] ^= buf[i]
	}
	return
}

func (c *cipherWriter) Write(p []byte) (n int, err error) {
	for i := 0; i < len(p); {
		buf := make([]byte, len(p))

		read := copy(buf, p[i:])
		buf2 := make([]byte, read)
		c.prng.Read(buf2)
		for j := 0; j < read; j++ {
			buf2[j] ^= buf[j]
		}
		shift, e := c.w.Write(buf2)
		n += shift
		if e != nil {
			err = e
			return
		}
		i += read
	}
	return
}

func NewReader(r io.Reader, prng io.Reader) io.Reader {
	return &cipherReader{r, prng}
}

func NewWriter(w io.Writer, prng io.Reader) io.Writer {
	return &cipherWriter{w, prng}
}
