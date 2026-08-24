/**
 * Async HTTP client adapter to consume the Image-to-TikZ FastAPI backend.
 * Conforms to the Hexagonal Architecture adapters layer contract.
 */
export interface GenerationResult {
    tikzCode: string;
    previewUrl?: string;
    packages?: string[];
    compilationSuccess?: boolean;
}

export interface CompilationResult {
    success: boolean;
    previewUrl?: string;
    error?: string;
}

/**
 * Resolves the API base URL based on environment or local defaults.
 */
export function getApiBaseUrl(): string {
    if (typeof import.meta !== "undefined" && import.meta.env && import.meta.env.PUBLIC_API_URL) {
        return import.meta.env.PUBLIC_API_URL.replace(/\/+$/, "");
    }
    return "";
}

/**
 * Sends a multipart image payload to the generative neural inference endpoint.
 *
 * @param imageFile Uploaded image file (PNG, JPG, WebP).
 * @returns Parsed generation and preview result.
 */
export async function generateTikzFromImage(imageFile: File): Promise<GenerationResult> {
    const formData = new FormData();
    formData.append("image", imageFile);

    const baseUrl = getApiBaseUrl();
    const primaryUrl = baseUrl ? `${baseUrl}/api/v1/generate` : "/api/v1/generate";

    let response: Response;
    try {
        response = await fetch(primaryUrl, {
            method: "POST",
            body: formData,
        });
    } catch (primaryError) {
        // Fallback: If calling via relative proxy fails, attempt direct localhost CORS endpoint
        if (!baseUrl && window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
            try {
                response = await fetch("http://127.0.0.1:8000/api/v1/generate", {
                    method: "POST",
                    body: formData,
                });
            } catch {
                throw primaryError;
            }
        } else {
            throw primaryError;
        }
    }

    if (!response.ok) {
        let errorDetail = response.statusText;
        try {
            const errorJson = await response.json();
            if (errorJson && typeof errorJson.detail === "string") {
                errorDetail = errorJson.detail;
            }
        } catch {
            // Keep default statusText
        }
        throw new Error(`API Error (${response.status}): ${errorDetail}`);
    }

    const data = await response.json();
    if (typeof data.tikz_code !== "string" || data.tikz_code.trim().length === 0) {
        throw new Error("The API returned an empty TikZ code payload.");
    }

    return {
        tikzCode: data.tikz_code,
        previewUrl: typeof data.preview_url === "string" ? data.preview_url : undefined,
        packages: Array.isArray(data.packages) ? data.packages : [],
        compilationSuccess: Boolean(data.compilation_success),
    };
}

/**
 * Requests on-demand LaTeX compilation of modified TikZ markup.
 *
 * @param tikzCode Valid LaTeX TikZ markup text.
 * @param packages List of required packages.
 * @returns Compilation result containing status and preview URL.
 */
export async function compileTikzMarkup(
    tikzCode: string,
    packages: string[] = []
): Promise<CompilationResult> {
    const baseUrl = getApiBaseUrl();
    const primaryUrl = baseUrl ? `${baseUrl}/api/v1/compile` : "/api/v1/compile";

    const payload = JSON.stringify({
        tikz_code: tikzCode,
        packages: packages,
    });

    const headers = { "Content-Type": "application/json" };

    let response: Response;
    try {
        response = await fetch(primaryUrl, {
            method: "POST",
            headers,
            body: payload,
        });
    } catch (primaryError) {
        if (!baseUrl && (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1")) {
            try {
                response = await fetch("http://127.0.0.1:8000/api/v1/compile", {
                    method: "POST",
                    headers,
                    body: payload,
                });
            } catch {
                throw primaryError;
            }
        } else {
            throw primaryError;
        }
    }

    if (!response.ok) {
        throw new Error(`Compilation API Error (${response.status}): ${response.statusText}`);
    }

    const data = await response.json();
    return {
        success: Boolean(data.success),
        previewUrl: typeof data.preview_url === "string" ? data.preview_url : undefined,
        error: typeof data.error === "string" ? data.error : undefined,
    };
}
