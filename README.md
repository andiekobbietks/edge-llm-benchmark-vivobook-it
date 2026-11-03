# Benchmark di inferenza LLM su hardware consumer: LFM2.5-1.2B su un portatile del 2021

## Introduzione

Questo documento descrive una misurazione reale di throughput di un modello linguistico (LLM) eseguito interamente su un portatile consumer del 2021, senza GPU dedicata, senza cloud, senza server. L'obiettivo era verificare un'affermazione specifica: "LFM2.5-1.2B gira a 40-70 token al secondo su un Intel i5-1135G7 con Iris Xe via Vulkan, out-of-the-box."

La risposta, misurata e non stimata, è più articolata di quanto si potesse prevedere.

## Manuale in PDF

Il rapporto completo è anche un manuale impaginato per la lettura su Kindle e per la stampa: formato 6x9 pollici, corpo 12 pt in Atkinson Hyperlegible, codice in Consolas su sfondo chiaro con numeri di riga, figure con didascalia, note e segnalibri PDF per ogni sezione.

- File: [`edge_llm_benchmark_handbook_it.pdf`](edge_llm_benchmark_handbook_it.pdf) (15 pagine)
- Contiene: metodologia, risultati, analisi, comandi di riproduzione, script completi, versioni dei pacchetti.

## Hardware e ambiente

| Componente | Specifica |
|---|---|
| Portatile | ASUS VivoBook (2021) |
| CPU | Intel Core i5-1135G7, 4 core / 8 thread, Tiger Lake |
| GPU | Intel Iris Xe Graphics, 80 EUs, integrata, memoria condivisa |
| RAM | 8 GB DDR4 (7.89 GB utilizzabili) |
| Sistema operativo | Windows 10 Home 19045 (22H2) |
| Vulkan | 1.2.198, driver Intel proprietario 30.0.101.1404 |
| Python | 3.12.10 |

## Modello testato

LiquidAI LFM2.5-1.2B-Instruct: architettura ibrida (convoluzioni a corto raggio + attenzione), 1.2 miliardi di parametri, progettato per esecuzione on-device. Testato in tre quantizzazioni:

- Q8_0 (1.19 GB) via Ollama, importato da GGUF
- Q4_0 QAD (0.66 GB) via Ollama, importato da GGUF
- int4 group-quantized (0.64 GB) via OpenVINO, pre-convertito da HuggingFace

## Risultati misurati

| Runtime | Quantizzazione | Dispositivo | tok/s (decode) | Prompt eval (tok/s) |
|---|---|---|---|---|
| Ollama 0.40.13 | Q8_0 | CPU (AVX2) | 28.0 | 118.5 |
| Ollama 0.40.13 | Q8_0 | GPU (Vulkan) | 14.5 | 99.2 |
| Ollama 0.40.13 | Q4_0 (QAD) | CPU (AVX2) | 38.3 | 121.5 |
| OpenVINO 2026.4.1 | int4 (gq) | CPU (8 thread) | 40.96 | (non misurato) |
| OpenVINO 2026.4.1 | int4 (gq) | iGPU (Iris Xe) | 15.28 | (non misurato) |

Misurazione: ~301 token generati per run, warmup di 16 token escluso, processore verificato via `ollama ps` o stringa device.

## Risultati chiave

**1. La CPU batte la GPU integrata per modelli piccoli.**

Su questo hardware, sia Vulkan (Ollama) che OpenVINO su Iris Xe sono circa 2 volte più lenti dello stesso modello su CPU. L'assunzione comune "scaricare il calcolo sulla iGPU = più veloce" non vale per modelli da 1.2B su Tiger Lake. Gli 80 EUs della iGPU non riescono a superare l'overhead di trasferimento memoria e il collo di bottiglia della banda condivisa.

**2. La soglia di 40 tok/s è raggiungibile solo con OpenVINO int4 su CPU.**

Ollama Q4 su CPU dà 38.3 (vicino ma sotto). La scelta del runtime conta tanto quanto la quantizzazione.

**3. La quantizzazione scala in modo approssimativamente lineare.**

Q4 = 1.4x Q8 in throughput su CPU (38.3 vs 28.0).

**4. La conversione locale è fragile per architetture ibride.**

`optimum-cli export openvino` crasha con access violation (0xC0000005) su LFM2.5-1.2B su Windows, sia con la versione stabile (2.2.0) che dev (2.3.0.dev0) di optimum-intel. Il crash è nel backend C++ (torch parallel materialization via ThreadPoolExecutor). Il modello pre-convertito da HuggingFace (OpenVINO org) funziona senza problemi.

**5. La NPU non è un'opzione su Tiger Lake.**

Il modello "ottimizzato per NPU" funziona su CPU qui. La NPU richiede Intel Core Ultra (Meteor Lake+).

**6. Raccomandazione pratica per questa classe di hardware:**

OpenVINO int4 su CPU (40 tok/s, 0.64 GB, 8 thread). Non forzare l'offload GPU.

## Budget RAM e tab del browser

Con il modello caricato e Chrome chiuso: 2.23 GB liberi su 7.89 GB totali.

| Tipo di tab | Dimensione media | Tab disponibili |
|---|---|---|
| Leggere (testo, documenti) | ~100 MB | ~14 |
| Medie (news, blog) | ~150 MB | ~9 |
| Pesanti (YouTube, web app) | ~300 MB | ~4 |

Il modello occupa ~0.64 GB fissi mentre gira. Con Chrome aperto (~1.7 GB), i tab Firefox disponibili si riducono drasticamente.

## Metodologia

Ogni configurazione: (1) run di warmup di 16 token per escludere il caricamento del modello dalla misurazione, (2) run misurata di ~300 token, (3) processore verificato via `ollama ps` o stringa device. RAM misurata via `Win32_OperatingSystem.FreePhysicalMemory` dopo il caricamento del modello. Nessuna altra applicazione pesante in esecuzione durante la misurazione (Chrome chiuso per liberare RAM).

## Guida alla riproduzione

### OpenVINO (il più veloce, 40.96 tok/s)

```
pip install openvino-genai huggingface_hub
python -c "import huggingface_hub as h; h.snapshot_download('OpenVINO/LFM2.5-1.2B-Instruct-int4-gq-ov', local_dir='ov_model')"
python -c "
from openvino_genai import LLMPipeline
import time
p = LLMPipeline('ov_model', 'CPU')
p.generate('warmup', max_new_tokens=16)
t0 = time.perf_counter()
out = p.generate('Il tuo prompt', max_new_tokens=300)
tok = p.get_tokenizer()
print(f'{tok.encode(out).input_ids.shape[1]/(time.perf_counter()-t0):.2f} tok/s')
"
```

### Ollama (CPU, Q4, 38.3 tok/s)

```
# Scarica il GGUF Q4 da HuggingFace, poi:
ollama create lfm2.5-q4 -f ModelfileQ4
$env:OLLAMA_NUM_GPU = "0"
ollama run lfm2.5-q4 "Il tuo prompt" --verbose
```

## Issue upstream

Issue #2102 aperta su huggingface/optimum-intel con i dettagli del crash. La causa radice è il parallel materialization in torch (spawn_materialize con ThreadPoolExecutor) che crasha su LFM2.5 su Windows. Il workaround è HF_DEACTIVATE_ASYNC_LOAD=1 (caricamento sincrono) più un pagefile >= 12 GB.

## Limitazioni

- Singola configurazione hardware (i5-1135G7 + Iris Xe). I risultati possono variare su AMD, Apple Silicon, o Intel più recente (Meteor Lake+ con NPU).
- Singolo modello (LFM2.5-1.2B). Modelli più grandi possono mostrare tradeoff CPU/GPU diversi.
- Nessun test di significatività statistica (singola run per configurazione). La varianza è tipicamente +/- 2-5%.
- Solo Windows 10. Linux (WSL) potrebbe dare risultati diversi, specialmente per la conversione.

## Autore

andiekobbietks (AFJ / AK Build). Ricerca su inferenza AI on-device su hardware consumer.

## Licenza

Note di ricerca e dati di benchmark: MIT. Licenze dei modelli: vedi LiquidAI LFM2.5 e OpenVINO model card.

## Riferimenti

- Repo benchmark: https://github.com/andiekobbietks/edge-llm-benchmark-vivobook
- Fork optimum-intel: https://github.com/andiekobbietks/optimum-intel/tree/edge-research-lfm25-2026
- Issue upstream: https://github.com/huggingface/optimum-intel/issues/2102
- Modello: https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct
- Modello OpenVINO: https://huggingface.co/OpenVINO/LFM2.5-1.2B-Instruct-int4-gq-ov
