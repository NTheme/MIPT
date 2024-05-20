//go:build !solution

package testequal

import "fmt"

func areEqualArray[T comparable](checkInstance []T, arrayInterface interface{}) bool {
	arrayInstance, err := arrayInterface.([]T)
	if !err || arrayInstance == nil || len(checkInstance) != len(arrayInstance) {
		return false
	}

	for i := range checkInstance {
		if !areEqual(checkInstance[i], arrayInstance[i]) {
			return false
		}
	}
	return true
}

func areEqualMap[K, V comparable](checkInstance map[K]V, mapInterface interface{}) bool {
	mapInstance, err := mapInterface.(map[K]V)
	if !err || mapInstance == nil {
		return false
	}

	for key := range checkInstance {
		checkValue, contains := mapInstance[key]
		if !contains || !areEqual(checkInstance[key], checkValue) {
			return false
		}
	}
	return true
}

func areEqual(expected, actual interface{}) bool {
	switch expectedType := expected.(type) {
	case int, int8, uint8, int16, uint16, int32, uint32, int64, uint64, string:
		return expected == actual
	case []int:
		return areEqualArray(expectedType, actual)
	case []byte:
		return areEqualArray(expectedType, actual)
	case map[string]string:
		return areEqualMap(expectedType, actual) && areEqualMap(actual.(map[string]string), expected)
	}
	return false
}

func AssertInternal(t T, expected, actual interface{}, equal bool, msgAndArgs ...interface{}) bool {
	t.Helper()

	if areEqual(expected, actual) != equal {
		if equal {
			t.Errorf("not ")
		}
		t.Errorf("equal:\nexpected: %v\nactual:   %v\nmessage:  ", expected, actual)
		if len(msgAndArgs) != 0 {
			t.Errorf("%s", fmt.Sprintf(msgAndArgs[0].(string), msgAndArgs[1:]...))
		}
		return false
	}
	return true
}

// AssertEqual checks that expected and actual are equal.
//
// Marks caller function as having failed but continues execution.
//
// Returns true iff arguments are equal.
func AssertEqual(t T, expected, actual interface{}, msgAndArgs ...interface{}) bool {
	t.Helper()
	return AssertInternal(t, expected, actual, true, msgAndArgs...)
}

// AssertNotEqual checks that expected and actual are not equal.
//
// Marks caller function as having failed but continues execution.
//
// Returns true iff arguments are not equal.
func AssertNotEqual(t T, expected, actual interface{}, msgAndArgs ...interface{}) bool {
	t.Helper()
	return AssertInternal(t, expected, actual, false, msgAndArgs...)
}

// RequireEqual does the same as AssertEqual but fails caller test immediately.
func RequireEqual(t T, expected, actual interface{}, msgAndArgs ...interface{}) {
	t.Helper()
	if !AssertEqual(t, expected, actual, msgAndArgs...) {
		t.FailNow()
	}
}

// RequireNotEqual does the same as AssertNotEqual but fails caller test immediately.
func RequireNotEqual(t T, expected, actual interface{}, msgAndArgs ...interface{}) {
	t.Helper()
	if !AssertNotEqual(t, expected, actual, msgAndArgs...) {
		t.FailNow()
	}
}
