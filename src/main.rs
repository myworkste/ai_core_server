use axum::{
    extract::Json,
    http::{Method, StatusCode},
    response::IntoResponse,
    routing::post,
    Router,
};
use serde::{Deserialize, Serialize};
use std::env;
use std::net::SocketAddr;
use tower_http::cors::{Any, CorsLayer};

#[derive(Deserialize)]
struct GenerateRequest {
    prompt: String,
    system_prompt: Option<String>,
    temperature: Option<f32>,
}

#[derive(Serialize)]
struct GenerateResponse {
    response: String,
}

#[tokio::main]
async fn main() {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::POST, Method::OPTIONS])
        .allow_headers(Any);

    let app = Router::new()
        .route("/api/v1/generate", post(handle_generate))
        .layer(cors);

    let port = env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let addr: SocketAddr = format!("0.0.0.0:{}", port)
        .parse()
        .expect("Invalid port or address");

    println!("🚀 Universal AI Backend listening on {}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind to port");
        
    axum::serve(listener, app)
        .await
        .expect("Failed to start server");
}

async fn handle_generate(Json(payload): Json<GenerateRequest>) -> impl IntoResponse {
    let api_key = match env::var("GEMINI_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "GEMINI_API_KEY environment variable is not set",
            )
                .into_response()
        }
    };

    let client = reqwest::Client::new();
    // Using gemini-2.5-flash as the "3.1 Flash" likely refers to a requested version or typo for 1.5 Flash
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={}",
        api_key
    );

    let mut body = serde_json::json!({
        "contents": [{
            "parts": [{"text": payload.prompt}]
        }]
    });

    if let Some(sys_prompt) = payload.system_prompt {
        body["system_instruction"] = serde_json::json!({
            "parts": [{"text": sys_prompt}]
        });
    }

    if let Some(temp) = payload.temperature {
        body["generationConfig"] = serde_json::json!({
            "temperature": temp
        });
    }

    let response = client
        .post(&url)
        .json(&body)
        .send()
        .await;

    match response {
        Ok(res) => {
            if res.status().is_success() {
                match res.json::<serde_json::Value>().await {
                    Ok(json) => {
                        let text = json["candidates"][0]["content"]["parts"][0]["text"]
                            .as_str()
                            .unwrap_or("Error: No text in response");
                        Json(GenerateResponse {
                            response: text.to_string(),
                        })
                        .into_response()
                    }
                    Err(_) => (StatusCode::BAD_GATEWAY, "Failed to parse Gemini response").into_response(),
                }
            } else {
                let status = res.status();
                let error_text = res.text().await.unwrap_or_default();
                (status, format!("Gemini API error: {}", error_text)).into_response()
            }
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to connect to Gemini API: {}", e),
        )
            .into_response(),
    }
}
