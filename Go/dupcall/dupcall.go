dupcall.go
851 B
//go:build !solution
package dupcall
import (
	"context"
	"fmt"
	"sync"
)
type Call struct {
	contextFunc context.Context
	cancelFunc  context.CancelFunc
	mutex       sync.Mutex
	result      any
	counter     uint
	err         error
}
func (o *Call) Do(
	ctx context.Context,
	cb func(context.Context) (interface{}, error),
) (result interface{}, err error) {
	o.mutex.Lock()
	if o.counter == 0 {
		o.contextFunc, o.cancelFunc = context.WithCancel(context.Background())
		o.err = fmt.Errorf("")
		cancelScenery := func(cl *Call) {
			cl.result, cl.err = cb(cl.contextFunc)
			cl.cancelFunc()
		}
		go cancelScenery(o)
	}
	o.counter++
	o.mutex.Unlock()
	select {
	case <-ctx.Done():
	case <-o.contextFunc.Done():
	}
	o.mutex.Lock()
	o.counter--
	if o.counter == 0 {
		o.cancelFunc()
	}
	result, err = o.result, o.err
	o.mutex.Unlock()
	return
}
