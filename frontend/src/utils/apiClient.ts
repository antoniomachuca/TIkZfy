/**
 * Cliente HTTP asíncrono para consumir la API (FastAPI)
 * Cumple con la Capa de Adaptadores según la Arquitectura Hexagonal.
 */
export interface GenerationResult {
    tikzCode: string;
    previewUrl?: string;
}

export async function generateTikzFromImage(imageFile: File): Promise<GenerationResult> {
    const formData = new FormData();
    formData.append("image", imageFile);

    try {
        const response = await fetch("/api/v1/generate", {
            method: "POST",
            body: formData,
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }
        
        const data = await response.json();
        if (typeof data.tikz_code !== "string" || data.tikz_code.length === 0) {
            throw new Error("The API returned no TikZ code.");
        }

        return {
            tikzCode: data.tikz_code,
            previewUrl: typeof data.preview_url === "string" ? data.preview_url : undefined,
        };
    } catch (error) {
        console.error("Error connecting to the TikZ Generation API:", error);
        throw error;
    }
}
