//go:build !solution

package blowfish

// #cgo pkg-config: libcrypto
// #cgo CFLAGS: -Wno-deprecated-declarations
// #include <openssl/blowfish.h>
// #define key_len sizeof(BF_KEY)
import "C"
import "unsafe"

type Blowfish struct {
	key *C.BF_KEY
}

func New(key []byte) *Blowfish {
	res := Blowfish{(*C.BF_KEY)(C.malloc(C.key_len))}
	C.BF_set_key(res.key, (C.int)(len(key)), (*C.uchar)(unsafe.SliceData(key)))
	return &res
}

func (b *Blowfish) BlockSize() int {
	return 8
}

func (b *Blowfish) Common(dst, src []byte, enc C.int) {
	C.BF_ecb_encrypt((*C.uchar)(unsafe.SliceData(src)),
		(*C.uchar)(unsafe.SliceData(dst)), b.key, enc)
}

func (b *Blowfish) Encrypt(dst, src []byte) {
	b.Common(dst, src, C.BF_ENCRYPT)
}

func (b *Blowfish) Decrypt(dst, src []byte) {
	b.Common(dst, src, C.BF_DECRYPT)
}
