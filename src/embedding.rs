use nalgebra::DVector;
use serde::Deserialize;

const EMBEDDING_API: &str = "http://localhost:1234/v1/embeddings";
const EMBEDDING_MODEL: &str = "text-embedding-qwen3-embedding-0.6b";

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

/// Get text embedding from local LM Studio
pub fn get_embedding(text: &str) -> Option<DVector<f32>> {
    if text.trim().is_empty() {
        return None;
    }

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .ok()?;

    let body = serde_json::json!({
        "input": text,
        "model": EMBEDDING_MODEL
    });

    let resp = client.post(EMBEDDING_API).json(&body).send().ok()?;

    if !resp.status().is_success() {
        return None;
    }

    let data: EmbeddingResponse = resp.json().ok()?;
    if data.data.is_empty() {
        return None;
    }

    Some(DVector::from_vec(data.data[0].embedding.clone()))
}
