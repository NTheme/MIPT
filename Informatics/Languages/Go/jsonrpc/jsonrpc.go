//go:build !solution

package jsonrpc

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"reflect"
)

func Exit(err error) {
	if err != nil {
		_, _ = fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func MakeHandler(service interface{}) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		byName := reflect.ValueOf(service).MethodByName(r.RequestURI[1:])
		request := reflect.New(byName.Type().In(1).Elem()).Interface()
		Exit(json.NewDecoder(r.Body).Decode(request))
		result := byName.Call([]reflect.Value{reflect.ValueOf(context.Background()), reflect.ValueOf(request)})

		if !result[1].IsNil() {
			w.WriteHeader(500)
			_, err := w.Write([]byte(result[1].Interface().(error).Error()))
			Exit(err)
		} else {
			w.WriteHeader(200)
			Exit(json.NewEncoder(w).Encode(result[0].Interface()))
		}
	})
}

func Call(ctx context.Context, endpoint string, method string, req, rsp interface{}) (err error) {
	encoded, err := json.Marshal(req)
	Exit(err)
	requestWithContext, err := http.NewRequestWithContext(ctx, "GET", endpoint+"/"+method, bytes.NewReader(encoded))
	Exit(err)

	client := http.Client{}
	do, err := client.Do(requestWithContext)
	Exit(err)

	if do.StatusCode != http.StatusOK {
		buf := new(bytes.Buffer)
		_, err := buf.ReadFrom(do.Body)
		Exit(err)
		return errors.New(buf.String())
	}

	Exit(json.NewDecoder(do.Body).Decode(rsp))
	return
}
