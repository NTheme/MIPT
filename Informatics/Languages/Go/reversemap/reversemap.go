//go:build !solution

package reversemap

import "reflect"

func ReverseMap(forward interface{}) interface{} {
	res := reflect.MakeMap(reflect.MapOf(reflect.ValueOf(forward).Type().Elem(), reflect.ValueOf(forward).Type().Key()))
	for it := reflect.ValueOf(forward).MapRange(); it.Next(); {
		res.SetMapIndex(it.Value(), it.Key())
	}
	return res.Interface()
}
