package configs

import (
	_ "embed"
)

//go:embed language_extensions.json
var LanguageMappingFile []byte
