//go:build !solution

package jsonlist

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"reflect"
	"strings"
)

func Exit(err error) {
	if err != nil {
		_, _ = fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func GetMarshal(value reflect.Value) (marshal strings.Builder) {
	for i := 0; i < value.Len(); i++ {
		bytes, err := json.Marshal(value.Index(i).Interface())
		Exit(err)

		marshal.Write(bytes)
		if i < value.Len()-1 {
			marshal.WriteString(" ")
		}
	}
	return marshal
}

func Marshal(w io.Writer, slice interface{}) (err error) {
	if reflect.TypeOf(slice).Kind() != reflect.Slice {
		return &json.UnsupportedTypeError{Type: reflect.TypeOf(slice)}
	}

	marshal := GetMarshal(reflect.ValueOf(slice))
	_, err = w.Write([]byte(marshal.String()))
	return
}

func GetUnmarshal(value reflect.Value, decoder *json.Decoder) (unmarshal reflect.Value) {
	unmarshal = reflect.MakeSlice(value.Type(), 0, 2)
	for {
		response := reflect.New(value.Type().Elem()).Interface()
		err := decoder.Decode(response)
		if err == io.EOF {
			break
		}
		unmarshal = reflect.Append(unmarshal, reflect.ValueOf(response).Elem())
	}
	return unmarshal
}

func Unmarshal(r io.Reader, slice interface{}) (err error) {
	if reflect.ValueOf(slice).Kind() != reflect.Pointer {
		return &json.UnsupportedTypeError{Type: reflect.TypeOf(slice)}
	}

	value := reflect.ValueOf(slice).Elem()
	value.Set(GetUnmarshal(value, json.NewDecoder(r)))
	return nil
}
