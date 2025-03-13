//go:build !solution

package externalsort

import (
	"container/heap"
	"io"
	"os"
)

func NewReader(r io.Reader) LineReader {
	return &NewLineReader{r}
}

func NewWriter(w io.Writer) LineWriter {
	return &NewLineWriter{w}
}

type StringExtended struct {
	str    string
	reader LineReader
}

type Heap []StringExtended

func (h *Heap) Len() int { return len(*h) }

func (h *Heap) Less(i, j int) bool { return (*h)[i].str < (*h)[j].str }

func (h *Heap) Swap(i, j int) { (*h)[i], (*h)[j] = (*h)[j], (*h)[i] }

func (h *Heap) Push(x interface{}) { *h = append(*h, x.(StringExtended)) }

func (h *Heap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

func Merge(w LineWriter, readers ...LineReader) error {
	heapInstance := &Heap{}
	heap.Init(heapInstance)

	for i := range readers {
		str, err := readers[i].ReadLine()
		if len(str) == 0 {
			if err == io.EOF {
				continue
			}
			if err != nil {
				return err
			}
		}
		heap.Push(heapInstance, StringExtended{str, readers[i]})
	}

	var str string
	for len(*heapInstance) > 0 {
		stringExtended := heap.Pop(heapInstance).(StringExtended)
		err := w.Write(stringExtended.str)
		if err != nil {
			return err
		}

		str, err = stringExtended.reader.ReadLine()
		if len(str) == 0 {
			if err == io.EOF {
				continue
			}
			if err != nil {
				return err
			}
		}
		heap.Push(heapInstance, StringExtended{str, stringExtended.reader})
	}
	return nil
}

func Read(file *os.File, heapInstance *Heap) error {
	reader := NewReader(file)

	for {
		str, err := reader.ReadLine()
		if len(str) == 0 {
			if err == io.EOF {
				break
			}
			if err != nil {
				return err
			}
		}
		heap.Push(heapInstance, StringExtended{str, nil})
	}
	return nil
}

func Write(file *os.File, heapInstance *Heap) error {
	writer := NewWriter(file)

	for len(*heapInstance) > 0 {
		err := writer.Write(heap.Pop(heapInstance).(StringExtended).str)
		if err != nil {
			return err
		}
	}
	return nil
}

func ReadWrite(filename string, heapInstance *Heap, read bool) error {
	var file *os.File
	var err error

	if read {
		file, _ = os.Open(filename)
		err = Read(file, heapInstance)
	} else {
		file, _ = os.Create(filename)
		err = Write(file, heapInstance)
	}

	if err != nil {
		return err
	}

	err = file.Close()
	if err != nil {
		return err
	}
	return nil
}

func SortFile(filename string) error {
	heapInstance := &Heap{}
	heap.Init(heapInstance)

	err := ReadWrite(filename, heapInstance, true)
	if err != nil {
		return err
	}
	err = ReadWrite(filename, heapInstance, false)
	if err != nil {
		return err
	}

	return nil
}

func Sort(w io.Writer, in ...string) error {
	for index := range in {
		err := SortFile(in[index])
		if err != nil {
			return err
		}
	}

	readers := make([]LineReader, len(in))
	for index := range in {
		file, _ := os.Open(in[index])
		readers[index] = &NewLineReader{file}
	}
	return Merge(&NewLineWriter{w}, readers...)
}
