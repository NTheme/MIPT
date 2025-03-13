//go:build !solution

package retryupdate

import (
	"errors"
	"github.com/gofrs/uuid"
	"gitlab.com/slon/shad-go/retryupdate/kvapi"
)

func SetupUpdate(c kvapi.Client, key string, updateFn func(oldValue *string) (newValue string, err error)) (*kvapi.SetRequest, error) {
	var err error
	var authError *kvapi.AuthError

	var oldValue *string
	var response *kvapi.GetResponse
	var request = kvapi.GetRequest{}
	var setRequest = kvapi.SetRequest{}

	request.Key = key
	setRequest.Key = key
	setRequest.NewVersion = uuid.Must(uuid.NewV4())

	for {
		response, err = c.Get(&request)
		if errors.Is(err, kvapi.ErrKeyNotFound) || errors.As(err, &authError) || err == nil {
			break
		}
	}

	if errors.As(err, &authError) {
		return nil, err
	}
	if !errors.Is(err, kvapi.ErrKeyNotFound) {
		oldValue = &response.Value
		setRequest.OldVersion = response.Version
	}

	setRequest.Value, err = updateFn(oldValue)
	if err != nil {
		return nil, err
	}

	return &setRequest, nil
}

func UpdateValue(c kvapi.Client, key string, updateFn func(oldValue *string) (newValue string, err error)) error {
	setRequest, err := SetupUpdate(c, key, updateFn)
	if err != nil {
		return err
	}

	for {
		_, setError := c.Set(setRequest)
		var authError *kvapi.AuthError
		var conflictError *kvapi.ConflictError

		if errors.As(setError, &authError) {
			return setError
		}
		if errors.As(setError, &conflictError) {
			if conflictError.ExpectedVersion == setRequest.NewVersion {
				return nil
			}
			setRequest, err = SetupUpdate(c, key, updateFn)
		}
		if errors.Is(setError, kvapi.ErrKeyNotFound) {
			setRequest.OldVersion = uuid.UUID{}
			setRequest.Value, err = updateFn(nil)
		}
		if setError == nil || err != nil {
			return nil
		}
	}
}
