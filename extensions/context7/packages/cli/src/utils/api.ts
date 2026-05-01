import type {
  ListSkillsResponse,
  SingleSkillResponse,
  SearchResponse,
  SuggestResponse,
  DownloadResponse,
  LibrarySearchResponse,
  SkillQuestionsResponse,
  StructuredGenerateInput,
  GenerateStreamEvent,
  SkillQuotaResponse,
  LibraryResolveResponse,
  ContextResponse,
} from "../types.js";
import { downloadSkillFromGitHub } from "./github.js";

let baseUrl = "https://context7.com";

/**
 * Get the current base URL for API requests.
 *
 * @returns {string} The configured API base URL.
 */
export function getBaseUrl(): string {
  return baseUrl;
}

/**
 * Set the base URL for API requests.
 *
 * Useful for testing or using alternative API endpoints.
 *
 * @param {string} url - The new base URL (e.g., "https://api.context7.local").
 */
export function setBaseUrl(url: string): void {
  baseUrl = url;
}

/**
 * List all skills available in a project.
 *
 * @param {string} project - The project identifier.
 * @returns {Promise<ListSkillsResponse>} Response containing list of skills.
 */
export async function listProjectSkills(project: string): Promise<ListSkillsResponse> {
  const params = new URLSearchParams({ project });
  const response = await fetch(`${baseUrl}/api/v2/skills?${params}`);
  return (await response.json()) as ListSkillsResponse;
}

/**
 * Get details for a specific skill.
 *
 * @param {string} project - The project identifier.
 * @param {string} skillName - The skill name or identifier.
 * @returns {Promise<SingleSkillResponse>} Response with skill details.
 */
export async function getSkill(project: string, skillName: string): Promise<SingleSkillResponse> {
  const params = new URLSearchParams({ project, skill: skillName });
  const response = await fetch(`${baseUrl}/api/v2/skills?${params}`);
  return (await response.json()) as SingleSkillResponse;
}

/**
 * Search for skills by query.
 *
 * @param {string} query - Search query for skills.
 * @returns {Promise<SearchResponse>} Search results.
 */
export async function searchSkills(query: string): Promise<SearchResponse> {
  const params = new URLSearchParams({ query });
  const response = await fetch(`${baseUrl}/api/v2/skills?${params}`);
  return (await response.json()) as SearchResponse;
}

/**
 * Get skill suggestions based on project dependencies.
 *
 * Analyzes the provided dependencies and suggests relevant skills that may
 * be useful for the project.
 *
 * @param {string[]} dependencies - List of project dependencies.
 * @param {string} [accessToken] - Optional access token for authentication.
 * @returns {Promise<SuggestResponse>} Suggestions response.
 */
export async function suggestSkills(
  dependencies: string[],
  accessToken?: string
): Promise<SuggestResponse> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (accessToken) {
    headers["Authorization"] = `Bearer ${accessToken}`;
  }
  const response = await fetch(`${baseUrl}/api/v2/skills/suggest`, {
    method: "POST",
    headers,
    body: JSON.stringify({ dependencies }),
  });
  return (await response.json()) as SuggestResponse;
}

/**
 * Download a skill and its associated files from GitHub.
 *
 * Fetches skill metadata and clones the GitHub repository containing the skill
 * source code.
 *
 * @param {string} project - The project identifier.
 * @param {string} skillName - The skill name or identifier.
 * @returns {Promise<DownloadResponse>} Download result with skill info and files.
 */
export async function downloadSkill(project: string, skillName: string): Promise<DownloadResponse> {
  const skillData = await getSkill(project, skillName);

  if (skillData.error) {
    return {
      skill: { name: skillName, description: "", url: "", project },
      files: [],
      error: skillData.message || skillData.error,
    };
  }

  const skill = {
    name: skillData.name,
    description: skillData.description,
    url: skillData.url,
    project: skillData.project,
  };

  const { files, error } = await downloadSkillFromGitHub(skill);

  if (error) {
    return { skill, files: [], error };
  }

  return { skill, files };
}

/**
 * Response from skill generation with generated code and library information.
 */
export interface GenerateSkillResponse {
  content: string;
  libraryName: string;
  error?: string;
}

/**
 * Search for libraries matching a query.
 *
 * Uses semantic search to find documentation libraries relevant to the query.
 *
 * @param {string} query - Search query or topic.
 * @param {string} [accessToken] - Optional access token for authentication.
 * @returns {Promise<LibrarySearchResponse>} Search results.
 */
export async function searchLibraries(
  query: string,
  accessToken?: string
): Promise<LibrarySearchResponse> {
  const params = new URLSearchParams({ query });
  const headers: Record<string, string> = {};
  if (accessToken) {
    headers["Authorization"] = `Bearer ${accessToken}`;
  }
  const response = await fetch(`${baseUrl}/api/v2/libs/search?${params}`, { headers });
  return (await response.json()) as LibrarySearchResponse;
}

/**
 * Get the current skill generation quota for a user.
 *
 * Returns usage information including total quota, used amount, remaining quota,
 * and plan tier.
 *
 * @param {string} accessToken - Access token for authentication (required).
 * @returns {Promise<SkillQuotaResponse>} Quota information.
 */
export async function getSkillQuota(accessToken: string): Promise<SkillQuotaResponse> {
  const response = await fetch(`${baseUrl}/api/v2/skills/quota`, {
    headers: { Authorization: `Bearer ${accessToken}` },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    return {
      used: 0,
      limit: 0,
      remaining: 0,
      tier: "free",
      resetDate: null,
      error: (errorData as { message?: string }).message || `HTTP error ${response.status}`,
    };
  }

  return (await response.json()) as SkillQuotaResponse;
}

/**
 * Get guided questions for skill generation based on selected libraries.
 *
 * Generates contextual questions to help guide the user through the skill
 * generation process, tailored to the selected libraries and use case.
 *
 * @param {Array<{ id: string; name: string }>} libraries - Selected libraries.
 * @param {string} motivation - User motivation or use case description.
 * @param {string} [accessToken] - Optional access token for authentication.
 * @returns {Promise<SkillQuestionsResponse>} Questions and guidance.
 */
export async function getSkillQuestions(
  libraries: Array<{ id: string; name: string }>,
  motivation: string,
  accessToken?: string
): Promise<SkillQuestionsResponse> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (accessToken) {
    headers["Authorization"] = `Bearer ${accessToken}`;
  }

  const response = await fetch(`${baseUrl}/api/v2/skills/questions`, {
    method: "POST",
    headers,
    body: JSON.stringify({ libraries, motivation }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    return {
      questions: [],
      error: (errorData as { message?: string }).message || `HTTP error ${response.status}`,
    };
  }

  return (await response.json()) as SkillQuestionsResponse;
}

/**
 * Generate a skill using structured input from a guided form.
 *
 * Streams generation progress via optional event callback. Returns final
 * generated code when complete.
 *
 * @param {StructuredGenerateInput} input - Structured generation parameters.
 * @param {Function} [onEvent] - Optional callback for streaming events.
 * @param {string} [accessToken] - Optional access token for authentication.
 * @returns {Promise<GenerateSkillResponse>} Generated skill code and metadata.
 */
export async function generateSkillStructured(
  input: StructuredGenerateInput,
  onEvent?: (event: GenerateStreamEvent) => void,
  accessToken?: string
): Promise<GenerateSkillResponse> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (accessToken) {
    headers["Authorization"] = `Bearer ${accessToken}`;
  }

  const response = await fetch(`${baseUrl}/api/v2/skills/generate`, {
    method: "POST",
    headers,
    body: JSON.stringify(input),
  });

  const libraryName = input.libraries[0]?.name || "skill";
  return handleGenerateResponse(response, libraryName, onEvent);
}

async function handleGenerateResponse(
  response: Response,
  libraryName: string,
  onEvent?: (event: GenerateStreamEvent) => void
): Promise<GenerateSkillResponse> {
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    return {
      content: "",
      libraryName,
      error: (errorData as { message?: string }).message || `HTTP error ${response.status}`,
    };
  }

  const reader = response.body?.getReader();
  if (!reader) {
    return { content: "", libraryName, error: "No response body" };
  }

  const decoder = new TextDecoder();
  let content = "";
  let finalLibraryName = libraryName;
  let error: string | undefined;
  let buffer = ""; // Buffer for incomplete lines across chunks

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value, { stream: true });
    buffer += chunk;

    // Split by newline but keep track of incomplete lines
    const lines = buffer.split("\n");
    // Keep the last element (may be incomplete) in the buffer
    buffer = lines.pop() || "";

    for (const line of lines) {
      const trimmedLine = line.trim();
      if (!trimmedLine) continue;

      try {
        const data = JSON.parse(trimmedLine) as GenerateStreamEvent;

        if (onEvent) {
          onEvent(data);
        }

        if (data.type === "complete") {
          content = data.content || "";
          finalLibraryName = data.libraryName || libraryName;
        } else if (data.type === "error") {
          error = data.message;
        }
      } catch {
        // Ignore malformed JSON lines
      }
    }
  }

  // Process any remaining data in the buffer
  if (buffer.trim()) {
    try {
      const data = JSON.parse(buffer.trim()) as GenerateStreamEvent;
      if (onEvent) {
        onEvent(data);
      }
      if (data.type === "complete") {
        content = data.content || "";
        finalLibraryName = data.libraryName || libraryName;
      } else if (data.type === "error") {
        error = data.message;
      }
    } catch {
      // Ignore malformed JSON
    }
  }

  return { content, libraryName: finalLibraryName, error };
}

function getAuthHeaders(accessToken?: string): Record<string, string> {
  const headers: Record<string, string> = {};
  const apiKey = process.env.CONTEXT7_API_KEY;
  if (apiKey) {
    headers["Authorization"] = `Bearer ${apiKey}`;
  } else if (accessToken) {
    headers["Authorization"] = `Bearer ${accessToken}`;
  }
  return headers;
}

export async function resolveLibrary(
  libraryName: string,
  query?: string,
  accessToken?: string
): Promise<LibraryResolveResponse> {
  const params = new URLSearchParams({ libraryName });
  if (query) {
    params.set("query", query);
  }

  const response = await fetch(`${baseUrl}/api/v2/libs/search?${params}`, {
    headers: getAuthHeaders(accessToken),
  });

  if (!response.ok) {
    const errorData = (await response.json().catch(() => ({}))) as {
      error?: string;
      message?: string;
    };
    return {
      results: [],
      error: errorData.error || `HTTP error ${response.status}`,
      message: errorData.message,
    };
  }

  return (await response.json()) as LibraryResolveResponse;
}

export interface GetContextOptions {
  type?: "json" | "txt";
}

export async function getLibraryContext(
  libraryId: string,
  query: string,
  options?: GetContextOptions,
  accessToken?: string
): Promise<ContextResponse | string> {
  const params = new URLSearchParams({ libraryId, query });
  if (options?.type) {
    params.set("type", options.type);
  }
  const response = await fetch(`${baseUrl}/api/v2/context?${params}`, {
    headers: getAuthHeaders(accessToken),
  });

  if (!response.ok) {
    const errorData = (await response.json().catch(() => ({}))) as {
      error?: string;
      message?: string;
      redirectUrl?: string;
    };

    if (response.status === 301 && errorData.redirectUrl) {
      return {
        codeSnippets: [],
        infoSnippets: [],
        error: errorData.error || "library_redirected",
        message: errorData.message,
        redirectUrl: errorData.redirectUrl,
      };
    }

    return {
      codeSnippets: [],
      infoSnippets: [],
      error: errorData.error || `HTTP error ${response.status}`,
      message: errorData.message,
    };
  }

  if (options?.type === "txt") {
    return await response.text();
  }

  return (await response.json()) as ContextResponse;
}
