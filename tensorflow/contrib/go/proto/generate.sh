#!/bin/bash

printf "// Code generated by go:generate\n// DO NOT EDIT\n\npackage tensorflow\n\nconst COpsDef = \`%s\`" "`cat ../../core/ops/ops.pbtxt | tr \\\` \'`"> proto/tf_ops_def.go
