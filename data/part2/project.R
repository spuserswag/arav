rm(list=ls())

library(websocket)
library(jsonlite)
library(later)

# 1. Initialize Parameters & State
symbol <- "btcusdt" 
ws_url <- sprintf("wss://stream.binance.us:9443/ws/%s@depth10@100ms", symbol)
csv_file <- "advanced_orderflow_ws.csv"
target_count <- 1000000 # Stop after this many rows

# Create a shared environment to track the count across different scopes
state <- new.env()
state$count <- 0

# 2. Setup Persistent File Connection
file_exists <- file.exists(csv_file)
con <- file(csv_file, open = "a")

if (!file_exists) {
  cat("timestamp,mid_price,micro_price,fractional_price,spread,obi_1,obi_10,obi_diff\n", file = con)
}

# 3. Define the WebSocket Client
ws <- WebSocket$new(ws_url, autoConnect = FALSE)

# 4. Handle Incoming Data (The Event Loop)
ws$onMessage(function(event) {
  # Increment the counter immediately
  state$count <- state$count + 1
  
  # Parse the incoming JSON payload
  data <- fromJSON(event$data)
  
  bids_mat <- apply(data$bids, 2, as.numeric)
  asks_mat <- apply(data$asks, 2, as.numeric)
  
  best_bid <- bids_mat[1, 1]
  best_ask <- asks_mat[1, 1]
  bid_vol_1 <- bids_mat[1, 2]
  ask_vol_1 <- asks_mat[1, 2]
  
  mid_price <- (best_bid + best_ask) / 2
  spread <- best_ask - best_bid
  fractional_price <- mid_price - floor(mid_price)
  
  micro_price <- (best_bid * ask_vol_1 + best_ask * bid_vol_1) / (bid_vol_1 + ask_vol_1)
  obi_1 <- (bid_vol_1 - ask_vol_1) / (bid_vol_1 + ask_vol_1)
  
  bid_vol_10 <- sum(bids_mat[, 2])
  ask_vol_10 <- sum(asks_mat[, 2])
  obi_10 <- (bid_vol_10 - ask_vol_10) / (bid_vol_10 + ask_vol_10)
  
  obi_diff <- obi_10 - obi_1
  
  # --- SAVE TO CSV ---
  current_time <- format(Sys.time(), "%Y-%m-%d %H:%M:%OS3")
  row_str <- sprintf("%s,%.2f,%.2f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                     current_time, mid_price, micro_price, fractional_price, spread, obi_1, obi_10, obi_diff)
  
  cat(row_str, file = con)
  
  # Print status less frequently to save CPU (e.g., every 1,000 rows)
  if (state$count %% 1000 == 0) {
    cat(sprintf("[%s] Collected: %d / %d | Mid: %.2f | OBI(10): %+.2f\n", 
                current_time, state$count, target_count, mid_price, obi_10))
  }
})

# 5. Handle Connection Events
ws$onOpen(function(event) {
  cat("WebSocket connected! Listening for 100ms updates.\n")
})

ws$onClose(function(event) {
  cat("WebSocket closed gracefully.\n")
  # Check if connection is still open before closing to avoid errors
  if (isOpen(con)) {
    close(con)
    cat("File connection closed.\n")
  }
})

ws$onError(function(event) {
  cat("WebSocket error: ", event$message, "\n")
})

# 6. Start Connection and Keep R Alive
ws$connect()

# The loop now checks our environment counter instead of running infinitely
while(state$count < target_count) {
  later::run_now()
  Sys.sleep(0.01) 
}

# 7. Clean Shutdown
cat("Target count reached! Initiating shutdown...\n")
ws$close() # This will trigger the ws$onClose event, which closes the file.