package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
)

func main() {
	modelPath := flag.String("model", "/data/model.json", "Path to XGBoost model file")
	normPath := flag.String("normalization", "/data/normalization.json", "Path to normalization.json")
	mccPath := flag.String("mcc-risk", "/data/mcc_risk.json", "Path to mcc_risk.json")
	port := flag.String("port", "8080", "HTTP port")
	flag.Parse()

	// Allow env var override (useful for Docker)
	if p := os.Getenv("PORT"); p != "" {
		*port = p
	}

	log.Println("Starting Rinha 2026 API...")

	// Load normalization config
	log.Println("Loading normalization config...")
	normData, err := os.ReadFile(*normPath)
	if err != nil {
		log.Fatalf("Failed to read normalization.json: %v", err)
	}
	if err := json.Unmarshal(normData, &normConfig); err != nil {
		log.Fatalf("Failed to parse normalization.json: %v", err)
	}

	// Load MCC risk map
	log.Println("Loading MCC risk map...")
	mccData, err := os.ReadFile(*mccPath)
	if err != nil {
		log.Fatalf("Failed to read mcc_risk.json: %v", err)
	}
	mccRiskMap = make(map[string]float64)
	if err := json.Unmarshal(mccData, &mccRiskMap); err != nil {
		log.Fatalf("Failed to parse mcc_risk.json: %v", err)
	}

	// Load XGBoost model
	log.Printf("Loading XGBoost model from %s...", *modelPath)
	predictor, err = NewPredictor(*modelPath)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	defer predictor.Close()

	// Mark as ready
	ready = true
	log.Println("Model loaded, API is ready!")

	// Register handlers
	http.HandleFunc("/ready", readyHandler)
	http.HandleFunc("/fraud-score", fraudScoreHandler)

	addr := fmt.Sprintf(":%s", *port)
	log.Printf("Listening on %s", addr)
	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
