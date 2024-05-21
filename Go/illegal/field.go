//go:build !solution

package illegal

import (
	"reflect"
	"unsafe"
)

func SetPrivateField(obj interface{}, name string, value interface{}) {
	address := reflect.ValueOf(obj).Elem().FieldByName(name).Addr()
	reflect.NewAt(address.Elem().Type(), unsafe.Pointer(address.Pointer())).Elem().Set(reflect.ValueOf(value))
}
