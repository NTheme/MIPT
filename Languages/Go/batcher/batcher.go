//go:build !solution

package batcher

import (
	"gitlab.com/slon/shad-go/batcher/slow"
	"time"
)

type TimestampedValue struct {
	value       any
	currentTime time.Time
}

type Batcher struct {
	slowValue        *slow.Value
	timestampedValue chan TimestampedValue
}

func (b *Batcher) Load() any {
	currentTime := time.Now()
	current := <-b.timestampedValue
	if current.currentTime.Before(currentTime) {
		for flag := false; !flag; flag = true {
			current.currentTime = time.Now()
			current.value = b.slowValue.Load()
		}
	}
	b.timestampedValue <- current
	return current.value
}

func NewBatcher(v *slow.Value) *Batcher {
	batcher := Batcher{v, make(chan TimestampedValue, 1)}

	var timestampedValue TimestampedValue
	for flag := false; !flag; flag = true {
		timestampedValue.currentTime = time.Now()
		timestampedValue.value = v.Load()
	}

	batcher.timestampedValue <- timestampedValue
	return &batcher
}
