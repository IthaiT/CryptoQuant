# -*- coding: utf-8 -*-
"""
Real-time K-line chart with trade markers for Backtrader - TradingView (Lightweight Charts)
Serves a local web page with TradingView-style charts and polls data from Python.
"""
import webbrowser
import threading
import time
import socket
from collections import deque
from datetime import datetime
from typing import List, Dict

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from src.utils.logger import logger


class InteractiveRealtimeChartPlotter:
    """
    Realtime candlestick chart plotter using TradingView Lightweight Charts
    Displays via a local web server - no PyQt/desktop GUI, cross-platform, memory-efficient
    """
    def __init__(self, max_bars=60000, update_interval=1, default_visible_bars=100, port: int = 8765):
        self.max_bars = max_bars
        self.update_interval = update_interval
        self.default_visible_bars = default_visible_bars
        self.port = port
        self.bar_count = 0
        self.base_index = 0
        self._server_started = False
        self._server_thread = None

        # Data storage (sliding window)
        self.dates: deque = deque(maxlen=max_bars)
        self.opens: deque = deque(maxlen=max_bars)
        self.highs: deque = deque(maxlen=max_bars)
        self.lows: deque = deque(maxlen=max_bars)
        self.closes: deque = deque(maxlen=max_bars)
        self.volumes: deque = deque(maxlen=max_bars)

        # Signals (absolute indices)
        self.buy_signals: List = []
        self.sell_signals: List = []
        self.close_signals: List = []

        # FastAPI app
        self.app = FastAPI()
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self._mount_routes()

    def add_bar(self, dt, open_price, high, low, close, volume):
        """Add a candlestick bar"""
        self.dates.append(dt)
        self.opens.append(open_price)
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        self.volumes.append(volume)
        self.bar_count += 1
        self.base_index = self.bar_count - len(self.dates)
        self._prune_signals()

    def update_chart(self):
        """Compatibility method - web mode does not require explicit updates"""
        return

    def set_max_bars(self, max_bars):
        """
        Set maximum number of bars to display.
        Dynamically adjust deque sizes to accommodate all data.
        
        Args:
            max_bars: New maximum bar count
        """
        self.max_bars = max_bars
        # Recreate deques with new maxlen
        old_dates = list(self.dates)
        old_opens = list(self.opens)
        old_highs = list(self.highs)
        old_lows = list(self.lows)
        old_closes = list(self.closes)
        old_volumes = list(self.volumes)
        
        self.dates = deque(old_dates, maxlen=max_bars)
        self.opens = deque(old_opens, maxlen=max_bars)
        self.highs = deque(old_highs, maxlen=max_bars)
        self.lows = deque(old_lows, maxlen=max_bars)
        self.closes = deque(old_closes, maxlen=max_bars)
        self.volumes = deque(old_volumes, maxlen=max_bars)

    def start_live(self, interval_sec: float = 0.5):
        """Start web server and open chart in browser"""
        if not self._server_started:
            self._server_started = True
            # Start server in background thread (daemon so it exits with main)
            self._server_thread = threading.Thread(target=self._run_server, daemon=True)
            self._server_thread.start()
            # Wait for server to start (check port availability)
            self._wait_for_server(max_retries=30)
            logger.info('Web server is running at http://127.0.0.1:{}'.format(self.port))
            # Open browser
            time.sleep(0.5)  # Give server a bit more time to fully initialize
            webbrowser.open(f"http://127.0.0.1:{self.port}")
    
    def _wait_for_server(self, max_retries=30):
        """Wait for server to be ready by checking port"""
        for i in range(max_retries):
            try:
                sock = socket.create_connection(("127.0.0.1", self.port), timeout=0.5)
                sock.close()
                return True
            except (socket.timeout, ConnectionRefusedError, OSError):
                time.sleep(0.2)
        logger.warning('Warning: Web server may not have started properly')
        return False

    def _mount_routes(self):
        """Register FastAPI routes"""
        @self.app.get("/", response_class=HTMLResponse)
        def index():
            return self._html_page()

        @self.app.get("/data", response_class=JSONResponse)
        def data():
            return self._json_payload()

    def _run_server(self):
        """Run the FastAPI server"""
        try:
            # Use uvicorn.run directly - it handles event loop management
            uvicorn.run(
                self.app,
                host="127.0.0.1",
                port=self.port,
                log_level="error",
                access_log=False
            )
        except Exception as e:
            logger.error('Server error: {}'.format(e))

    def _html_page(self) -> str:
        return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Real-time Backtest Chart</title>
  <script src="https://unpkg.com/lightweight-charts@4/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    * { box-sizing: border-box; }
    html, body { height: 100%; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1e1e1e; color: #fff; }
    #container { display: flex; flex-direction: column; height: 100%; }
    #chart { flex: 3; background: #1e1e1e; position: relative; border-bottom: 1px solid #333; }
    #volume-chart { flex: 1; background: #1e1e1e; position: relative; border-bottom: 1px solid #333; }
    #footer { padding: 12px 16px; background: #0a0a0a; border-top: 1px solid #333; font-size: 12px; color: #888; }
    #status { position: fixed; bottom: 8px; right: 8px; padding: 12px 16px; background: #2a2a2a; border: 1px solid #444; border-radius: 4px; z-index: 100; max-width: 360px; color: #fff; pointer-events: none; }
    #status.error { background: #3a2222; border-color: #ff6b6b; color: #ff9999; }
    #status.loading { color: #6db3f2; }
    #status.ok { background: #223a22; border-color: #4caf50; color: #90ee90; }
  </style>
</head>
<body>
  <div id="container">
    <div id="status" class="loading">Loading chart... Please wait</div>
    <div id="controls" style="position:fixed; bottom:8px; left:8px; z-index:101; background:#2a2a2a; border:1px solid #444; border-radius:4px; padding:8px 12px; color:#ddd; font-size:12px; display:flex; gap:8px; align-items:center;">
      <label style="display:flex; align-items:center; gap:6px;">
        <input id="toggle-playback" type="checkbox" /> Playback
      </label>
      <button id="btn-play" disabled>Play</button>
      <button id="btn-pause" disabled>Pause</button>
      <label>Speed
        <select id="speed-select" disabled>
          <option value="50">Fast</option>
          <option value="150" selected>Medium</option>
          <option value="300">Slow</option>
        </select>
      </label>
    </div>
    <div id="chart"></div>
    <div id="volume-chart"></div>
    <div id="footer">TradingView Lightweight Charts - Live from Python Backtest (Dark Mode)</div>
  </div>

  <script>
    (function() {
      console.log('Chart script starting...');
      
      const chartContainer = document.getElementById('chart');
      const volumeChartContainer = document.getElementById('volume-chart');
      const statusEl = document.getElementById('status');
      
      let chart = null;
      let candleSeries = null;
      let volumeChart = null;
      let volumeSeries = null;
      let lastDataCount = 0;
      let cachedBars = [];
      let retryCount = 0;
      const maxRetries = 10;

      // Playback state
      let playbackEnabled = false;
      let playing = false;
      let speedMs = 150;
      let renderTimer = null;
      let refreshTimer = null;
      let fullBars = [];
      let fullMarkers = [];
      let renderedCount = 0;

      async function initChart() {
        try {
          console.log('Initializing chart...');
          if (!LightweightCharts) {
            throw new Error('LightweightCharts library not loaded');
          }
          
          // Create main candlestick chart
          chart = LightweightCharts.createChart(chartContainer, {
            width: chartContainer.clientWidth,
            height: chartContainer.clientHeight,
            layout: { 
              background: { color: '#1e1e1e' },
              textColor: '#d1d1d1'
            },
            grid: { 
              vertLines: { color: '#333333' }, 
              horzLines: { color: '#333333' } 
            },
            timeScale: { 
              secondsVisible: false, 
              rightOffset: 10, 
              fixLeftEdge: false,
              timeVisible: true
            },
            watermark: {
              color: 'rgba(255, 255, 255, 0.05)',
              visible: true,
              text: 'Backtest'
            }
          });

          candleSeries = chart.addCandlestickSeries({
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderVisible: true,
            wickUpColor: '#26a69a',
            wickDownColor: '#ef5350'
          });

          // Create separate volume chart
          volumeChart = LightweightCharts.createChart(volumeChartContainer, {
            width: volumeChartContainer.clientWidth,
            height: volumeChartContainer.clientHeight,
            layout: { 
              background: { color: '#1e1e1e' },
              textColor: '#d1d1d1'
            },
            grid: { 
              vertLines: { color: '#333333' }, 
              horzLines: { color: '#333333' } 
            },
            timeScale: { 
              secondsVisible: false, 
              rightOffset: 10, 
              fixLeftEdge: false,
              timeVisible: true
            }
          });

          volumeSeries = volumeChart.addHistogramSeries({
            color: '#26a69a',
            priceFormat: { type: 'volume' }
          });

          // Sync time scales between charts
          const syncToChart = (sourceChart) => {
            sourceChart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
              if (!range) return;
              volumeChart.timeScale().setVisibleLogicalRange(range);
            });
          };
          syncToChart(chart);

          console.log('Chart initialized successfully');
          statusEl.textContent = 'Initializing... fetching data';
          statusEl.className = 'loading';
          
          await refresh();
          setupControls();
        } catch (e) {
          console.error('Chart init error:', e);
          statusEl.textContent = 'Error: ' + e.message;
          statusEl.className = 'error';
        }
      }

      async function refresh() {
        try {
          const resp = await fetch('/data', { cache: 'no-store' });
          if (!resp.ok) throw new Error('HTTP ' + resp.status);
          
          const payload = await resp.json();
          if (!payload.bars) throw new Error('No bars in response');
          
          // Ensure markers sorted by time
          if (payload.markers && payload.markers.length) {
            payload.markers.sort((a, b) => a.time - b.time);
          }

          const bars = payload.bars;
          if (bars.length === 0) {
            statusEl.textContent = 'Waiting for backtest data...';
            statusEl.className = 'loading';
            lastDataCount = 0;
          } else {
            if (!playbackEnabled) {
              // Batch mode: 使用增量更新避免闪烁
              if (lastDataCount === 0) {
                // 首次加载：全量设置
                candleSeries.setData(bars);
                const volData = bars.map(b => ({ time: b.time, value: b.volume, color: b.close >= b.open ? '#26a69a' : '#ef5350' }));
                volumeSeries.setData(volData);
                chart.timeScale().fitContent();
                volumeChart.timeScale().fitContent();
                cachedBars = bars;
              } else if (bars.length > lastDataCount) {
                // 增量更新：只添加新的 bars
                for (let i = lastDataCount; i < bars.length; i++) {
                  const bar = bars[i];
                  candleSeries.update(bar);
                  const volBar = { time: bar.time, value: bar.volume, color: bar.close >= bar.open ? '#26a69a' : '#ef5350' };
                  volumeSeries.update(volBar);
                }
                cachedBars = bars;
              }
              // 更新交易信号（仅当有变化时）
              if (payload.markers && payload.markers.length > 0) {
                candleSeries.setMarkers(payload.markers);
              }
              lastDataCount = bars.length;
            } else {
              // Playback mode: keep full arrays and let render loop add bars
              fullBars = bars;
              fullMarkers = payload.markers || [];
              // Initialize renderedCount on first enable
              if (renderedCount === 0 && fullBars.length > 0) {
                renderedCount = Math.max(1, Math.min(50, fullBars.length)); // start with some bars for context
                candleSeries.setData(fullBars.slice(0, renderedCount));
                const volData0 = fullBars.slice(0, renderedCount).map(b => ({ time: b.time, value: b.volume, color: b.close >= b.open ? '#26a69a' : '#ef5350' }));
                volumeSeries.setData(volData0);
                const lastTime0 = fullBars[renderedCount - 1].time;
                const visMarkers0 = fullMarkers.filter(m => m.time <= lastTime0);
                candleSeries.setMarkers(visMarkers0);
                chart.timeScale().fitContent();
                volumeChart.timeScale().fitContent();
              }
            }

            const msg = 'Chart updated: ' + bars.length + ' bars';
            const sig = (payload.markers ? payload.markers.length : 0);
            statusEl.textContent = msg + ' | Signals: ' + sig + (playbackEnabled ? ' | Playback' : ' | Batch');
            statusEl.className = 'ok';
            retryCount = 0;
          }
        } catch (e) {
          console.error('Refresh error:', e);
          retryCount++;
          
          if (retryCount >= maxRetries) {
            statusEl.textContent = 'Error: Connection lost (' + retryCount + '/' + maxRetries + ')';
            statusEl.className = 'error';
          } else {
            statusEl.textContent = 'Connecting... (' + retryCount + '/' + maxRetries + ')';
            statusEl.className = 'loading';
          }
        }
      }

      window.addEventListener('resize', () => {
        if (chart && volumeChart) {
          const w = chartContainer.clientWidth;
          const h = chartContainer.clientHeight;
          const vh = volumeChartContainer.clientHeight;
          chart.applyOptions({ width: w, height: h });
          volumeChart.applyOptions({ width: w, height: vh });
        }
      });

      console.log('Setting up page...');
      initChart();
      refreshTimer = setInterval(refresh, 1000);

      function setupControls() {
        const toggle = document.getElementById('toggle-playback');
        const btnPlay = document.getElementById('btn-play');
        const btnPause = document.getElementById('btn-pause');
        const speedSel = document.getElementById('speed-select');

        const updateButtons = () => {
          btnPlay.disabled = !playbackEnabled || playing;
          btnPause.disabled = !playbackEnabled || !playing;
          speedSel.disabled = !playbackEnabled;
        };

        toggle.addEventListener('change', () => {
          playbackEnabled = toggle.checked;
          if (!playbackEnabled) {
            // Stop render loop and reset
            playing = false;
            if (renderTimer) { clearInterval(renderTimer); renderTimer = null; }
            renderedCount = 0;
            // Force full refresh next tick
          } else {
            // Prepare for playback mode
            if (fullBars.length > 0) {
              renderedCount = Math.max(1, Math.min(50, fullBars.length));
              candleSeries.setData(fullBars.slice(0, renderedCount));
              const volData0 = fullBars.slice(0, renderedCount).map(b => ({ time: b.time, value: b.volume, color: b.close >= b.open ? '#26a69a' : '#ef5350' }));
              volumeSeries.setData(volData0);
              const lastTime0 = fullBars[renderedCount - 1].time;
              const visMarkers0 = fullMarkers.filter(m => m.time <= lastTime0);
              candleSeries.setMarkers(visMarkers0);
              chart.timeScale().fitContent();
              volumeChart.timeScale().fitContent();
            }
          }
          updateButtons();
        });

        btnPlay.addEventListener('click', () => {
          if (!playbackEnabled) return;
          if (playing) return;
          playing = true;
          if (renderTimer) clearInterval(renderTimer);
          renderTimer = setInterval(() => {
            if (!fullBars.length) return;
            if (renderedCount < fullBars.length) {
              const nextBar = fullBars[renderedCount];
              candleSeries.update(nextBar);
              const volBar = { time: nextBar.time, value: nextBar.volume, color: nextBar.close >= nextBar.open ? '#26a69a' : '#ef5350' };
              volumeSeries.update(volBar);
              renderedCount += 1;
              const lastTime = nextBar.time;
              const visMarkers = fullMarkers.filter(m => m.time <= lastTime);
              candleSeries.setMarkers(visMarkers);
            } else {
              // reached end
              playing = false;
              clearInterval(renderTimer);
              renderTimer = null;
            }
          }, speedMs);
          updateButtons();
        });

        btnPause.addEventListener('click', () => {
          if (!playbackEnabled) return;
          playing = false;
          if (renderTimer) { clearInterval(renderTimer); renderTimer = null; }
          updateButtons();
        });

        speedSel.addEventListener('change', () => {
          speedMs = parseInt(speedSel.value, 10) || 150;
          if (playing) {
            // restart timer with new speed
            clearInterval(renderTimer);
            renderTimer = setInterval(() => {
              if (!fullBars.length) return;
              if (renderedCount < fullBars.length) {
                const nextBar = fullBars[renderedCount];
                candleSeries.update(nextBar);
                const volBar = { time: nextBar.time, value: nextBar.volume, color: nextBar.close >= nextBar.open ? '#26a69a' : '#ef5350' };
                volumeSeries.update(volBar);
                renderedCount += 1;
                const lastTime = nextBar.time;
                const visMarkers = fullMarkers.filter(m => m.time <= lastTime);
                candleSeries.setMarkers(visMarkers);
              } else {
                playing = false;
                clearInterval(renderTimer);
                renderTimer = null;
              }
            }, speedMs);
          }
        });

        updateButtons();
      }
    })();
  </script>
</body>
</html>
"""

    def _json_payload(self) -> Dict:
        """Convert sliding window to Lightweight Charts format"""
        bars = []
        for i in range(len(self.dates)):
            dt = pd.to_datetime(self.dates[i])
            bars.append({
                "time": int(dt.timestamp()),
                "open": float(self.opens[i]),
                "high": float(self.highs[i]),
                "low": float(self.lows[i]),
                "close": float(self.closes[i]),
                "volume": float(self.volumes[i]),
            })
        # Generate trade signal markers
        markers = []
        def idx_to_time(abs_idx):
            rel = abs_idx - self.base_index
            if rel < 0 or rel >= len(self.dates):
                return None
            return int(pd.to_datetime(self.dates[rel]).timestamp())
        for abs_idx, price in self.buy_signals:
          t = idx_to_time(abs_idx)
          if t:
            markers.append({"time": t, "position": "belowBar", "shape": "arrowUp", "color": "#00ffff", "text": f"BUY {price:.2f}"})
        for abs_idx, price in self.sell_signals:
          t = idx_to_time(abs_idx)
          if t:
            markers.append({"time": t, "position": "aboveBar", "shape": "arrowDown", "color": "#ff00ff", "text": f"SELL {price:.2f}"})
        for abs_idx, price in self.close_signals:
          t = idx_to_time(abs_idx)
          if t:
            markers.append({"time": t, "position": "inBar", "shape": "square", "color": "#ffcc00", "text": f"CLOSE {price:.2f}"})
        # Ensure markers sorted by time to avoid rendering glitches
        try:
          markers.sort(key=lambda m: m["time"])  # type: ignore
        except Exception:
          pass
        return {"bars": bars, "markers": markers}

    def add_buy_signal(self, price):
        """Record buy signal"""
        abs_idx = self.bar_count - 1
        self.buy_signals.append((abs_idx, price))

    def add_sell_signal(self, price):
        """Record sell signal"""
        abs_idx = self.bar_count - 1
        self.sell_signals.append((abs_idx, price))

    def add_close_signal(self, price):
        """Record close signal"""
        abs_idx = self.bar_count - 1
        self.close_signals.append((abs_idx, price))

    def _prune_signals(self):
        """Remove signals that have slid out of the window"""
        self.buy_signals = [(i, p) for i, p in self.buy_signals if i >= self.base_index]
        self.sell_signals = [(i, p) for i, p in self.sell_signals if i >= self.base_index]
        self.close_signals = [(i, p) for i, p in self.close_signals if i >= self.base_index]

    def save_chart(self, filename='backtest_result.png'):
        """Web version does not directly export images"""
        pass

    def show(self):
        """Web version does not require blocking display"""
        pass

    def show_async(self):
        """Web version does not need this"""
        pass


# Compatibility alias
RealtimeChartPlotter = InteractiveRealtimeChartPlotter
