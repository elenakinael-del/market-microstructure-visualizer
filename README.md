
# Market Microstructure Visualizer

A high-performance visualization suite designed to map real-time market microstructure dynamics. This tool converts raw limit order book (LOB) data into interactive visual representations, enabling the identification of liquidity clusters and order flow imbalances that precede price shifts.

## Overview

This visualizer is engineered to bridge the gap between abstract order book data and intuitive market analysis. By rendering high-frequency data, it allows researchers to observe how liquidity evaporates or accumulates at specific price levels, providing a clearer picture of market sentiment in real-time.

## Core Capabilities

* **Real-Time LOB Mapping:** Processes streaming order book updates to visualize live market depth.


* **Liquidity Cluster Identification:** Highlights visual "hotspots" where significant buying or selling pressure is concentrated.
* **State Transition Tracking:** Specifically built to capture "pre-threshold" state transitions, helping to visualize market exhaustion before a move occurs.


* **Interactive Interface:** Built for exploration, allowing for granular inspection of microstructure events across varied timeframes.



## Technical Stack

* **Core Processing:** Python


* **Visualization Engine:** Plotly / Matplotlib (optimized for high-frequency data updates)


* **Geometric Mapping:** SciPy (used for processing order book spatial transitions)



## Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/elenakinael-del/market-microstructure-visualizer

```


2. **Install requirements:**
```bash
pip install -r requirements.txt

```


3. **Execute the visualizer:**
```bash
python main.py  # Or specify your primary entry point

```



## Research Context

This tool serves as the visual component of the broader **Neuro-Market Intelligence** initiative. By combining microstructure analysis with behavioral modeling, this visualizer helps validate the impact of herding and cognitive bias on observed market liquidity.

---

*This repository is part of the Neuro-Market Intelligence research suite. Contributions and experimental forks focused on advanced microstructure modeling are welcome.*
