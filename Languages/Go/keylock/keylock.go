//go:build !solution

package keylock

import (
	"sort"
	"sync"
)

type KeyLock struct {
	keys sync.Map
}

func New() *KeyLock {
	keyLock := KeyLock{}
	return &keyLock
}

func (l *KeyLock) LockKeys(keys []string, cancel <-chan struct{}) (canceled bool, unlock func()) {
	sortedKeys := append(make([]string, 0, len(keys)), keys...)
	sort.Strings(sortedKeys)

	exit := false
outer:
	for index := range sortedKeys {
		newChan := make(chan struct{}, 1)
		newChan <- struct{}{}

		value, _ := l.keys.LoadOrStore(sortedKeys[index], &newChan)
		select {
		case <-*value.(*chan struct{}):
		case <-cancel:
			for j := 0; j < index; j++ {
				val, _ := l.keys.Load(sortedKeys[j])
				*val.(*chan struct{}) <- struct{}{}
			}
			exit = true
			break outer
		}
	}

	if !exit {
		unlock = func() {
			for index := range sortedKeys {
				value, _ := l.keys.Load(sortedKeys[index])
				*value.(*chan struct{}) <- struct{}{}
			}
		}
	}
	return exit, unlock
}
