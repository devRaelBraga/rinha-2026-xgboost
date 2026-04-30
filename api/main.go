package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"
)

var knnIndex *KNNIndex

func main() {
	runtime.GOMAXPROCS(1)
	modelPath := flag.String("model", "/data/model.json", "Path to XGBoost model file")
	normPath := flag.String("normalization", "/data/normalization.json", "Path to normalization.json")
	mccPath := flag.String("mcc-risk", "/data/mcc_risk.json", "Path to mcc_risk.json")
	vectorsPath := flag.String("vectors", "/data/refs_vectors.bin", "Path to KNN reference vectors")
	labelsPath := flag.String("labels", "/data/refs_labels.bin", "Path to KNN reference labels")
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

	InvMaxInstallments = 1.0 / normConfig.MaxInstallments
	InvMaxAmount = 1.0 / normConfig.MaxAmount
	InvAmountVsAvgRatio = 1.0 / normConfig.AmountVsAvgRatio
	InvMaxMinutes = 1.0 / normConfig.MaxMinutes
	InvMaxKm = 1.0 / normConfig.MaxKm
	InvMaxTxCount24h = 1.0 / normConfig.MaxTxCount24h
	InvMaxMerchantAvgAmount = 1.0 / normConfig.MaxMerchantAvgAmount

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

	// Load KNN index
	log.Printf("Loading KNN index from %s and %s...", *vectorsPath, *labelsPath)
	knnIndex, err = LoadKNNIndex(*vectorsPath, *labelsPath)
	if err != nil {
		log.Fatalf("Failed to load KNN index: %v", err)
	}

	// Mark as ready
	ready = true
	log.Println("Model loaded, API is ready!")

	// Pega o ID da instância pelo ENV (ex: "1", "2")
	instanceID := os.Getenv("INSTANCE_ID")
	if instanceID == "" {
		instanceID = "1" // fallback
	}

	// Define o caminho do socket no volume compartilhado
	sockPath := fmt.Sprintf("/tmp/sockets/api%s.sock", instanceID)

	// IMPORTANTE: Limpa o socket antigo se o container reiniciou e o arquivo ficou lá
	os.Remove(sockPath)

	mux := http.NewServeMux()
	mux.HandleFunc("/ready", readyHandler)
	mux.HandleFunc("/fraud-score", fraudScoreHandler)

	srv := &http.Server{
		Handler:      mux,
		ReadTimeout:  2 * time.Second,
		WriteTimeout: 2 * time.Second,
		IdleTimeout:  30 * time.Second,
	}

	// Cria o listener do tipo "unix"
	listener, err := net.Listen("unix", sockPath)
	if err != nil {
		log.Fatalf("Failed to listen on socket: %v", err)
	}

	// Garante que o Nginx tenha permissão para ler/escrever no socket
	os.Chmod(sockPath, 0777)

	log.Printf("Listening on Unix Socket: %s", sockPath)

	// Inicia o servidor no socket
	go func() {
		if err := srv.Serve(listener); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server failed: %v", err)
		}
	}()

	// Limpeza graciosa ao desligar o container
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	<-stop
	os.Remove(sockPath)
}
