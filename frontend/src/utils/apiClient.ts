/**
 * Cliente HTTP asíncrono para consumir la API (FastAPI)
 * Cumple con la Capa de Adaptadores según la Arquitectura Hexagonal.
 */
export async function generateTikzFromImage(imageFile: File): Promise<string> {
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
        return data.tikz_code;
    } catch (error) {
        console.error("Error connecting to the TikZ Generation API:", error);
        throw error;
    }
}
