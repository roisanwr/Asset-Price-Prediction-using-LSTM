import os
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from flask import Flask, render_template_string, request, jsonify
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# --- KONFIGURASI ---
app = Flask(__name__)
MODEL_DIR = 'models'

# --- FRONTEND (HTML, CSS, JS Chart.js) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YUI Stock Pro Chart</title>
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <!-- Chart.js (Library Grafik) -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <style>
        body { background-color: #0f172a; color: #cbd5e1; font-family: 'Segoe UI', sans-serif; }
        .card { background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; }
        .form-control, .form-select { background-color: #334155; border: 1px solid #475569; color: #fff; }
        .form-control:focus, .form-select:focus { background-color: #334155; color: #fff; border-color: #3b82f6; }
        .btn-primary { background-color: #3b82f6; border: none; font-weight: 600; }
        .btn-primary:hover { background-color: #2563eb; }
        
        /* Chart Container */
        .chart-container { 
            position: relative; 
            height: 400px; 
            width: 100%; 
            margin-top: 20px;
        }

        .prediction-badge {
            background: rgba(16, 185, 129, 0.2);
            border: 1px solid #10b981;
            color: #34d399;
            padding: 10px 20px;
            border-radius: 8px;
            text-align: center;
        }
        
        .loading-overlay {
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(15, 23, 42, 0.8);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 10;
            border-radius: 12px;
        }
    </style>
</head>
<body>
    <div class="container py-4">
        <!-- Header -->
        <div class="d-flex justify-content-between align-items-center mb-4">
            <h3 class="fw-bold text-white"><i class="fas fa-chart-line text-primary"></i> YUI Pro Dashboard</h3>
            <span class="badge bg-secondary">Model: LSTM v1.0</span>
        </div>
        
        <div class="row">
            <!-- Sidebar / Control Panel -->
            <div class="col-lg-3 mb-4">
                <div class="card p-3 h-100">
                    <form id="predict-form">
                        <label class="form-label text-white fw-bold">Pilih Aset</label>
                        <select class="form-select mb-3" id="ticker" required>
                            <option value="" selected disabled>-- Pilih --</option>
                            <option value="BBCA.JK">BBCA (BCA)</option>
                            <option value="BBRI.JK">BBRI (BRI)</option>
                            <option value="NVDA">NVDA (Nvidia)</option>
                            <option value="AAPL">AAPL (Apple)</option>
                            <option value="BTC-USD">BTC (Bitcoin)</option>
                        </select>
                        
                        <div class="mb-3">
                            <label class="form-label text-muted small">Periode Data</label>
                            <select class="form-select form-select-sm" disabled>
                                <option>6 Bulan Terakhir</option>
                            </select>
                        </div>

                        <button type="submit" class="btn btn-primary w-100 mb-3">
                            <i class="fas fa-play me-2"></i> Prediksi
                        </button>
                    </form>

                    <!-- Info Box -->
                    <div id="result-info" style="display:none;">
                        <hr class="border-secondary">
                        <div class="prediction-badge mb-2">
                            <div class="small text-uppercase">Prediksi Besok</div>
                            <div id="pred-price" class="fs-4 fw-bold">Rp 0</div>
                        </div>
                        <div class="text-center">
                            <small class="text-muted">Harga Terakhir:</small><br>
                            <span id="last-price" class="text-white fw-bold">Rp 0</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Main Chart Area -->
            <div class="col-lg-9">
                <div class="card p-3 position-relative">
                    <!-- Loading Spinner -->
                    <div id="loading" class="loading-overlay">
                        <div class="text-center">
                            <div class="spinner-border text-primary" role="status"></div>
                            <p class="mt-2 text-white">Connecting to Yahoo Finance...</p>
                        </div>
                    </div>

                    <!-- Canvas Grafik -->
                    <h5 class="text-white mb-3" id="chart-title">Menunggu Input...</h5>
                    <div class="chart-container">
                        <canvas id="stockChart"></canvas>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let myChart = null; // Variabel global buat nyimpen chart

        document.getElementById('predict-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            // UI Update
            document.getElementById('loading').style.display = 'flex';
            document.getElementById('result-info').style.display = 'none';
            
            const ticker = document.getElementById('ticker').value;

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ticker: ticker })
                });

                const data = await response.json();

                if (data.error) throw new Error(data.error);

                // Update Angka
                document.getElementById('pred-price').innerText = data.prediction_formatted;
                document.getElementById('last-price').innerText = data.last_price_formatted;
                document.getElementById('chart-title').innerText = `Analisa & Prediksi: ${ticker}`;
                document.getElementById('result-info').style.display = 'block';

                // --- RENDERING CHART ---
                renderChart(data.dates, data.history_prices, data.prediction_price, data.next_date);

            } catch (err) {
                alert("Error: " + err.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        });

        function renderChart(dates, historyData, predictedPrice, nextDate) {
            const ctx = document.getElementById('stockChart').getContext('2d');

            // Kalau chart udah ada sebelumnya, hancurkan dulu biar gak numpuk
            if (myChart) {
                myChart.destroy();
            }

            // Gabungkan label (Tanggal Histori + Tanggal Besok)
            const allLabels = [...dates, nextDate];
            
            // Data Histori: Angka asli, diakhiri null buat tempat prediksi
            // Kita ambil harga terakhir dari historyData
            const lastRealPrice = historyData[historyData.length - 1];

            // Data Prediksi: Isinya null semua, kecuali 2 titik terakhir (Hari ini & Besok)
            // Ini biar garis prediksinya nyambung dari titik terakhir
            const predictionData = new Array(dates.length - 1).fill(null);
            predictionData.push(lastRealPrice); // Titik sambung
            predictionData.push(predictedPrice); // Titik prediksi

            myChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: allLabels,
                    datasets: [
                        {
                            label: 'Data Historis',
                            data: historyData,
                            borderColor: '#3b82f6', // Biru
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            borderWidth: 2,
                            pointRadius: 0, // Titik histori gak usah digambar biar bersih
                            pointHoverRadius: 5,
                            tension: 0.1, // Garis agak melengkung dikit
                            fill: true
                        },
                        {
                            label: 'Prediksi AI',
                            data: predictionData, // Data khusus prediksi
                            borderColor: '#10b981', // Hijau
                            borderWidth: 2,
                            borderDash: [5, 5], // Garis putus-putus
                            pointRadius: 4,
                            pointBackgroundColor: '#10b981',
                            tension: 0
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false,
                    },
                    plugins: {
                        legend: { labels: { color: '#cbd5e1' } },
                        tooltip: { mode: 'index', intersect: false }
                    },
                    scales: {
                        x: {
                            ticks: { color: '#94a3b8', maxTicksLimit: 10 },
                            grid: { color: '#334155' }
                        },
                        y: {
                            ticks: { color: '#94a3b8' },
                            grid: { color: '#334155' }
                        }
                    }
                }
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        ticker = data.get('ticker')
        
        # 1. SETUP PATH FILE
        safe_ticker = ticker.replace('.', '_').replace('-', '_')
        model_path = os.path.join(MODEL_DIR, f"{safe_ticker}_model.h5")
        scaler_path = os.path.join(MODEL_DIR, f"{safe_ticker}_scaler.pkl")

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return jsonify({'error': f"Model {ticker} tidak ditemukan."}), 404

        # 2. AMBIL DATA HISTORIS (6 Bulan terakhir buat grafik)
        stock = yf.Ticker(ticker)
        # Ambil data agak banyak buat grafik, tapi minimal harus cover 60 hari buat LSTM
        df = stock.history(period="6mo")
        
        if len(df) < 60:
            return jsonify({'error': "Data Yahoo Finance kurang."}), 400

        # Siapkan Data Buat Grafik (Array Tanggal & Harga)
        # Format tanggal jadi string 'YYYY-MM-DD'
        dates = df.index.strftime('%Y-%m-%d').tolist()
        history_prices = df['Close'].tolist()

        # 3. PROSES PREDIKSI
        # Ambil 60 data terakhir khusus buat input model
        last_60_days = df['Close'].values[-60:].reshape(-1, 1)
        
        scaler = joblib.load(scaler_path)
        model = load_model(model_path)

        scaled_input = scaler.transform(last_60_days)
        x_input = np.reshape(scaled_input, (1, 60, 1))
        
        predicted_scaled = model.predict(x_input, verbose=0)
        predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

        # 4. HITUNG TANGGAL PREDIKSI (BESOK)
        last_date = df.index[-1]
        next_date = last_date + timedelta(days=1)
        # Kalau besok sabtu/minggu sebenernya pasar libur, tapi buat grafik kita plot aja H+1
        next_date_str = next_date.strftime('%Y-%m-%d')

        # 5. RESPONSE LENGKAP KE FRONTEND
        currency = "IDR" if ".JK" in ticker else "USD"
        
        return jsonify({
            'prediction_formatted': f"{currency} {predicted_price:,.2f}",
            'last_price_formatted': f"{currency} {history_prices[-1]:,.2f}",
            'raw_price': float(predicted_price),
            
            # Data Khusus Grafik
            'dates': dates,                # List tanggal histori
            'history_prices': history_prices, # List harga histori
            'prediction_price': float(predicted_price), # Harga prediksi
            'next_date': next_date_str     # Tanggal prediksi
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)