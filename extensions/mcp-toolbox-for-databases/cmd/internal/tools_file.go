// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package internal

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"

	"github.com/goccy/go-yaml"
	"github.com/googleapis/genai-toolbox/internal/server"
)

// ToolsFile represents the structure of a tools configuration YAML file.
// Aggregates all resource types (sources, auth, tools, etc.) in one file.
type ToolsFile struct {
	Sources         server.SourceConfigs         `yaml:"sources"`         // database sources
	AuthServices    server.AuthServiceConfigs    `yaml:"authServices"`    // authentication configurations
	EmbeddingModels server.EmbeddingModelConfigs `yaml:"embeddingModels"` // embedding model definitions
	Tools           server.ToolConfigs           `yaml:"tools"`           // tool definitions
	Toolsets        server.ToolsetConfigs        `yaml:"toolsets"`        // groupings of tools
	Prompts         server.PromptConfigs         `yaml:"prompts"`         // prompt definitions
}

// parseEnv substitutes environment variable references in input.
// Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax.
// Returns error if referenced variable not found and no default provided.
func parseEnv(input string) (string, error) {
	re := regexp.MustCompile(`\$\{(\w+)(:([^}]*))?\}`)

	var err error
	output := re.ReplaceAllStringFunc(input, func(match string) string {
		parts := re.FindStringSubmatch(match)

		// extract the variable name
		variableName := parts[1]
		if value, found := os.LookupEnv(variableName); found {
			return value
		}
		if len(parts) >= 4 && parts[2] != "" {
			return parts[3]
		}
		err = fmt.Errorf("environment variable not found: %q", variableName)
		return ""
	})
	return output, err
}

// parseToolsFile parses and converts YAML bytes into a ToolsFile structure.
// Handles environment variable substitution and v1-to-v2 format conversion.
func parseToolsFile(ctx context.Context, raw []byte) (ToolsFile, error) {
	var toolsFile ToolsFile
	// Replace environment variables if found
	output, err := parseEnv(string(raw))
	if err != nil {
		return toolsFile, fmt.Errorf("error parsing environment variables: %s", err)
	}
	raw = []byte(output)

	raw, err = convertToolsFile(raw)
	if err != nil {
		return toolsFile, fmt.Errorf("error converting tools file: %s", err)
	}

	// Parse contents
	toolsFile.Sources, toolsFile.AuthServices, toolsFile.EmbeddingModels, toolsFile.Tools, toolsFile.Toolsets, toolsFile.Prompts, err = server.UnmarshalResourceConfig(ctx, raw)
	if err != nil {
		return toolsFile, err
	}
	return toolsFile, nil
}

// convertToolsFile converts tools configuration from v1 to v2 YAML format.
// Transforms v1 flat structure into v2 multi-document format with kind/name headers.
func convertToolsFile(raw []byte) ([]byte, error) {
	var input yaml.MapSlice
	decoder := yaml.NewDecoder(bytes.NewReader(raw), yaml.UseOrderedMap())

	// convert to tools file v2
	var buf bytes.Buffer
	encoder := yaml.NewEncoder(&buf)

	v1keys := []string{"sources", "authSources", "authServices", "embeddingModels", "tools", "toolsets", "prompts"}
	for {
		if err := decoder.Decode(&input); err != nil {
			if err == io.EOF {
				break
			}
			return nil, err
		}
		for _, item := range input {
			key, ok := item.Key.(string)
			if !ok {
				return nil, fmt.Errorf("unexpected non-string key in input: %v", item.Key)
			}
			// check if the key is config file v1's key
			if slices.Contains(v1keys, key) {
				// check if value conversion to yaml.MapSlice successfully
				// fields such as "tools" in toolsets might pass the first check but
				// fail to convert to MapSlice
				if slice, ok := item.Value.(yaml.MapSlice); ok {
					// Deprecated: convert authSources to authServices
					if key == "authSources" {
						key = "authServices"
					}
					transformed, err := transformDocs(key, slice)
					if err != nil {
						return nil, err
					}
					// encode per-doc
					for _, doc := range transformed {
						if err := encoder.Encode(doc); err != nil {
							return nil, err
						}
					}
				} else {
					// invalid input will be ignored
					// we don't want to throw error here since the config could
					// be valid but with a different order such as:
					// ---
					// tools:
					// - tool_a
					// kind: toolsets
					// ---
					continue
				}
			} else {
				// this doc is already v2, encode to buf
				if err := encoder.Encode(input); err != nil {
					return nil, err
				}
				break
			}
		}
	}
	return buf.Bytes(), nil
}

// transformDocs converts v1 resource entries to v2 YAML documents.
// Each entry becomes a document with kind and name fields. Preserves YAML field order.
func transformDocs(kind string, input yaml.MapSlice) ([]yaml.MapSlice, error) {
	var transformed []yaml.MapSlice
	for _, entry := range input {
		entryName, ok := entry.Key.(string)
		if !ok {
			return nil, fmt.Errorf("unexpected non-string key for entry in '%s': %v", kind, entry.Key)
		}
		entryBody := ProcessValue(entry.Value, kind == "toolsets")

		currentTransformed := yaml.MapSlice{
			{Key: "kind", Value: kind},
			{Key: "name", Value: entryName},
		}

		// Merge the transformed body into our result
		if bodySlice, ok := entryBody.(yaml.MapSlice); ok {
			currentTransformed = append(currentTransformed, bodySlice...)
		} else {
			return nil, fmt.Errorf("unable to convert entryBody to MapSlice")
		}
		transformed = append(transformed, currentTransformed)
	}
	return transformed, nil
}

// ProcessValue recursively transforms YAML values during v1-to-v2 format migration.
// Renames 'kind' fields to 'type' and wraps toolset lists in a 'tools' key for proper nesting.
func ProcessValue(v any, isToolset bool) any {
	switch val := v.(type) {
	case yaml.MapSlice:
		// creating a new MapSlice is safer for recursive transformation
		newVal := make(yaml.MapSlice, len(val))
		for i, item := range val {
			// Perform renaming
			if item.Key == "kind" {
				item.Key = "type"
			}
			// Recursive call for nested values (e.g., nested objects or lists)
			item.Value = ProcessValue(item.Value, false)
			newVal[i] = item
		}
		return newVal
	case []any:
		// Process lists: If it's a toolset top-level list, wrap it.
		if isToolset {
			return yaml.MapSlice{{Key: "tools", Value: val}}
		}
		// Otherwise, recurse into list items (to catch nested objects)
		newVal := make([]any, len(val))
		for i := range val {
			newVal[i] = ProcessValue(val[i], false)
		}
		return newVal
	default:
		return val
	}
}

// mergeToolsFiles combines multiple ToolsFile structs into a single structure.
// Returns error if any resource name is duplicated across files.
// All resource names must be globally unique.
func mergeToolsFiles(files ...ToolsFile) (ToolsFile, error) {
	merged := ToolsFile{
		Sources:         make(server.SourceConfigs),
		AuthServices:    make(server.AuthServiceConfigs),
		EmbeddingModels: make(server.EmbeddingModelConfigs),
		Tools:           make(server.ToolConfigs),
		Toolsets:        make(server.ToolsetConfigs),
		Prompts:         make(server.PromptConfigs),
	}

	var conflicts []string

	for fileIndex, file := range files {
		// Check for conflicts and merge sources
		for name, source := range file.Sources {
			if _, exists := merged.Sources[name]; exists {
				conflicts = append(conflicts, fmt.Sprintf("source '%s' (file #%d)", name, fileIndex+1))
			} else {
				merged.Sources[name] = source
			}
		}

		// Check for conflicts and merge authServices
		for name, authService := range file.AuthServices {
			if _, exists := merged.AuthServices[name]; exists {
				conflicts = append(conflicts, fmt.Sprintf("authService '%s' (file #%d)", name, fileIndex+1))
			} else {
				merged.AuthServices[name] = authService
			}
		}

		// Check for conflicts and merge embeddingModels
		for name, em := range file.EmbeddingModels {
			if _, exists := merged.EmbeddingModels[name]; exists {
				conflicts = append(conflicts, fmt.Sprintf("embedding model '%s' (file #%d)", name, fileIndex+1))
			} else {
				merged.EmbeddingModels[name] = em
			}
		}

		// Check for conflicts and merge tools
		for name, tool := range file.Tools {
			if _, exists := merged.Tools[name]; exists {
				conflicts = append(conflicts, fmt.Sprintf("tool '%s' (file #%d)", name, fileIndex+1))
			} else {
				merged.Tools[name] = tool
			}
		}

		// Check for conflicts and merge toolsets
		for name, toolset := range file.Toolsets {
			if _, exists := merged.Toolsets[name]; exists {
				conflicts = append(conflicts, fmt.Sprintf("toolset '%s' (file #%d)", name, fileIndex+1))
			} else {
				merged.Toolsets[name] = toolset
			}
		}

		// Check for conflicts and merge prompts
		for name, prompt := range file.Prompts {
			if _, exists := merged.Prompts[name]; exists {
				conflicts = append(conflicts, fmt.Sprintf("prompt '%s' (file #%d)", name, fileIndex+1))
			} else {
				merged.Prompts[name] = prompt
			}
		}
	}

	// If conflicts were detected, return an error
	if len(conflicts) > 0 {
		return ToolsFile{}, fmt.Errorf("resource conflicts detected:\n  - %s\n\nPlease ensure each source, authService, tool, toolset and prompt has a unique name across all files", strings.Join(conflicts, "\n  - "))
	}

	return merged, nil
}

// LoadAndMergeToolsFiles reads multiple YAML files, parses them, and merges into one config.
// Errors if files cannot be read, parsed, or contain duplicate resource names.
func LoadAndMergeToolsFiles(ctx context.Context, filePaths []string) (ToolsFile, error) {
	var toolsFiles []ToolsFile

	for _, filePath := range filePaths {
		buf, err := os.ReadFile(filePath)
		if err != nil {
			return ToolsFile{}, fmt.Errorf("unable to read tool file at %q: %w", filePath, err)
		}

		toolsFile, err := parseToolsFile(ctx, buf)
		if err != nil {
			return ToolsFile{}, fmt.Errorf("unable to parse tool file at %q: %w", filePath, err)
		}

		toolsFiles = append(toolsFiles, toolsFile)
	}

	mergedFile, err := mergeToolsFiles(toolsFiles...)
	if err != nil {
		return ToolsFile{}, fmt.Errorf("unable to merge tools files: %w", err)
	}

	return mergedFile, nil
}

// LoadAndMergeToolsFolder discovers all .yaml/.yml files in a directory, parses, and merges them.
// Errors if directory does not exist, is not a directory, or contains no YAML files.
func LoadAndMergeToolsFolder(ctx context.Context, folderPath string) (ToolsFile, error) {
	// Check if directory exists
	info, err := os.Stat(folderPath)
	if err != nil {
		return ToolsFile{}, fmt.Errorf("unable to access tools folder at %q: %w", folderPath, err)
	}
	if !info.IsDir() {
		return ToolsFile{}, fmt.Errorf("path %q is not a directory", folderPath)
	}

	// Find all YAML files in the directory
	pattern := filepath.Join(folderPath, "*.yaml")
	yamlFiles, err := filepath.Glob(pattern)
	if err != nil {
		return ToolsFile{}, fmt.Errorf("error finding YAML files in %q: %w", folderPath, err)
	}

	// Also find .yml files
	ymlPattern := filepath.Join(folderPath, "*.yml")
	ymlFiles, err := filepath.Glob(ymlPattern)
	if err != nil {
		return ToolsFile{}, fmt.Errorf("error finding YML files in %q: %w", folderPath, err)
	}

	// Combine both file lists
	allFiles := append(yamlFiles, ymlFiles...)

	if len(allFiles) == 0 {
		return ToolsFile{}, fmt.Errorf("no YAML files found in directory %q", folderPath)
	}

	// Use existing LoadAndMergeToolsFiles function
	return LoadAndMergeToolsFiles(ctx, allFiles)
}
