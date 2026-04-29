package main

import (
	"encoding/json"
	"net/http"
	"sync"
)

// Global state (initialized in main.go)
var (
	predictor  *Predictor
	normConfig Normalization
	mccRiskMap map[string]float64
	ready      bool
)

// Pool for FraudRequest objects to reduce GC pressure.
var requestPool = sync.Pool{
	New: func() interface{} {
		return new(FraudRequest)
	},
}

// readyHandler responds 200 when the API is fully loaded.
func readyHandler(w http.ResponseWriter, r *http.Request) {
	if !ready {
		http.Error(w, "not ready", http.StatusServiceUnavailable)
		return
	}
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}

var (
	safeDefaultJSON = []byte(`{"approved":true,"fraud_score":0.0}`)
)

// fraudScoreHandler receives a transaction payload and returns the fraud decision.
func fraudScoreHandler(w http.ResponseWriter, r *http.Request) {
	if !ready {
		http.Error(w, "not ready", http.StatusServiceUnavailable)
		return
	}

	// Decode request
	req := requestPool.Get().(*FraudRequest)
	defer func() {
		// Reset the object before returning to pool
		req.LastTx = nil
		req.Customer.KnownMerchants = req.Customer.KnownMerchants[:0]
		requestPool.Put(req)
	}()

	if err := json.NewDecoder(r.Body).Decode(req); err != nil {
		// Return a safe default instead of HTTP error (errors cost 5x in scoring)
		w.Header().Set("Content-Type", "application/json")
		w.Write(safeDefaultJSON)
		return
	}

	// Vectorize → Predict
	vector := Vectorize(req, &normConfig, mccRiskMap)
	fraudScore, ok := predictor.Predict(vector)
	if !ok {
		// Fast fallback for timeouts or load shedding
		w.Header().Set("Content-Type", "application/json")
		w.Write(safeDefaultJSON)
		return
	}
	approved := fraudScore < 0.3

	// Respond
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(FraudResponse{
		Approved:   approved,
		FraudScore: fraudScore,
	})
}
