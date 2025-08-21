# TikTok Video Feature Extractor

A comprehensive Python toolkit for extracting multimodal features from TikTok videos, including audio, visual, and textual analysis. Supports both research and production workflows, with modular design and robust test coverage.

### Quickstart (Docker + Jupyter Notebook)

1) Prerequisites
- Install Docker and Docker Compose
- For GPU acceleration: install NVIDIA drivers and NVIDIA Container Toolkit

2) Start the container with Jupyter
```bash
docker compose up --build
```
Open `http://localhost:8888` in your browser. Jupyter runs without a token for local development.

3) Configure TikTok MS Token inside the container
```bash
python auto_configure_tiktok.py
```
Follow the prompts to add your `ms_token` into `config/config.yml` under the `tiktok` section. You can update it later with `python check_tiktok_config.py`.

4) Run a batch analysis (CSV of creators)
```bash
python main.py --csv data/creators.csv
```
The entrypoint delegates to `src/main_batch.py` and will crawl/analyze creators listed in the CSV.

5) Use the notebook (recommended for exploration)
Open `tiktok_video_project.ipynb` in Jupyter and run the cells to process a single video or a batch.

Notes
- On Windows, Docker + Jupyter is the recommended workflow to avoid local dependency issues.
- Your `ms_token` is sensitive. Do not commit it; it is stored in `config/config.yml` on your machine.

### Run locally (alternative)
Local runs require ffmpeg and Playwright dependencies in addition to Python packages.
```bash
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # or CPU wheels
pip install -r requirements.txt
python -m playwright install-deps chromium
python -m playwright install chromium
```
Then configure your TikTok token and run:
```bash
python auto_configure_tiktok.py
python main.py --csv data/creators.csv
```

### Common commands
```bash
# Validate TikTok configuration
python check_tiktok_config.py

# Demo and smoke tests
python demo_tiktok_integration.py
python test_tiktok_integration.py

# Real crawler sanity check (requires valid ms_token)
python test_real_crawler.py

# Batch CLI (see flags with -h)
python main.py --csv data/creators.csv --config config/config.yml --batch-size 5 --concurrency 2
```

### Notebook workflow
- Open `tiktok_video_project.ipynb` to run step-by-step examples for:
  - Single video analysis from `data/tiktok_videos/`
  - Batch analysis using a creators CSV
  - Inspecting intermediate artifacts (frames, transcripts, detections)
- For quick tests, you can also use `test_notebook.ipynb`.

### What gets extracted
- Video metadata: duration, resolution, frame rate, file size
- Audio: transcription (Whisper), voice activity, speech rate scoring, music recognition (Shazamio), audio events
- Visual: keyframes, sampled frames, SSIM dedup, blur/black filtering, YOLO detections
- Multimodal: BLIP captions, CLIP similarities, optional GPT‑4o analysis (configurable)
- Outputs: CSV/JSON/TXT summaries plus per-creator reports under `data/analysis_results/`

### Troubleshooting
- TikTokApi session fails to create: ensure a valid `ms_token`, try a fresh token from browser cookies, and confirm Playwright browsers are installed. In Docker these are preinstalled.
- Empty or zero videos returned: rate limiting or token expiry; reconfigure `ms_token` and reduce concurrency.
- GPU not used in Docker: install NVIDIA Container Toolkit and ensure Compose starts with GPU resources; verify `nvidia-smi` works on host.
- ffmpeg errors: ffmpeg is preinstalled in the Docker image. For local runs, install ffmpeg and ensure it is on PATH.
- Cost control: GPT‑4o analysis is configurable and may be disabled by default to avoid API costs; enable in `config/config.yml` if needed.

### Data layout and outputs
- `data/tiktok_videos/`: input and downloaded videos
- `data/tiktok_frames/<video_name>/`: extracted frames and analysis artifacts
- `data/analysis_results/<creator>/`: integrated crawl + analysis outputs per creator
- `data/results/`: consolidated CSV/JSON outputs for batch runs
- `logs/`: processing logs and error reports
- `models/`: model caches and weights (Torch/HF/YOLO)

### Configuration and overrides
- Primary config: `config/config.yml`.
- TikTok section requires a valid `ms_token`. Use `python auto_configure_tiktok.py` to set it.
- Important pipeline knobs (with environment overrides):
  - `processing.batch_size` (env: `BATCH_SIZE`)
  - `features.audio.sample_rate` (env: `SAMPLE_RATE`)
  - `models.whisper.model_size` (env: `WHISPER_MODEL_SIZE`)
  - `logging.level` (env: `LOG_LEVEL`)

Environment variables used by the app and Docker setup:
- `TIKTOK_MS_TOKEN`: fallback for `tiktok.ms_token` if not set in YAML
- `TIKTOK_BROWSER`: Playwright browser type (default `chromium`)
- `CUDA_VISIBLE_DEVICES`: GPU selection inside container
- `PYTORCH_CUDA_ALLOC_CONF`: CUDA memory behavior tuning
- `TORCH_HOME`, `HF_HOME`: model cache directories (mapped to `/app/models`)
- `YOLO_CONFIG_DIR`: YOLO cache/config directory
- `PYTHONPATH`: set to `/app` in Docker
- `DISPLAY`: X display for headless browser (Docker)

Windows note: prefer Docker + Notebook for running tests and demos to avoid incomplete dependencies in PowerShell.

### CLI reference
`main.py` delegates to `src/main_batch.py`. View flags:
```bash
python main.py -h | cat
```
Common flags:
- `--csv PATH`: creators CSV with column `username` (required)
- `--config PATH`: path to `config/config.yml`
- `--batch-id ID`: custom batch identifier
- `--resume`: resume a previous batch by `--batch-id`
- `--dry-run`: validate inputs and config without crawling/analysis
- `--batch-size N`: override batch size
- `--concurrency N`: max concurrent creators
- `--limit-per-creator N`: cap videos per creator

You can also run the integration module directly:
```bash
python src/tiktok_integration.py
```
It will process `data/creators.csv` when present.


---

## Project Overview
This project implements automatic multimodal feature extraction for TikTok short videos. It processes batches of videos to extract:
- Video metadata (duration, resolution, frame rate, file size)
- Audio features (speech, music, events, transcription with noise tolerance)
- Visual features (keyframes, representative frames, object detection, blur/black frame filtering)
- Multimodal features (YOLO, BLIP, CLIP, GPT4O-powered analysis)
- AI-powered categorization and video description with enhanced analysis dimensions

**Use Cases:**
- Product and content categorization for e-commerce
- Video search and recommendation
- Dataset creation for machine learning
- Research on multimodal video understanding
- Content analysis and creative insights

**Key Features:**
- **Scalable Processing**: Batch processing with configurable pipelines and memory management
- **Production Ready**: Comprehensive error handling, logging, and monitoring capabilities
- **GPU Acceleration**: CUDA support for faster processing with YOLO, CLIP, and multimodal models
- **Modular Architecture**: Independent components for audio, video, and multimodal analysis
- **Extensive Testing**: 90%+ test coverage with automated test suites for all modules
- **Docker Support**: Containerized deployment with GPU acceleration and Jupyter notebook access
- **GPT4O Integration**: Advanced multimodal analysis with OpenAI's GPT4O model
- **Image Compression**: Optimized token usage with configurable image compression
- **Enhanced Analysis**: Content creative ideas, emotional value, selling points, target demographics

---

## Workflow
1. **Video Collection**: Place `.mp4` files in `data/tiktok_videos/`. The system supports various video formats and automatically validates file integrity.

2. **Metadata Extraction**: Use FFmpeg to extract duration, resolution, frame rate, and file size. This provides baseline information for processing optimization.

3. **Audio Processing**:
   - Extract audio as 16kHz mono WAV for optimal speech recognition
   - Separate vocals/non-vocals using Demucs for enhanced audio analysis
   - Transcribe speech using OpenAI Whisper with language auto-detection
   - **Enhanced Noise Tolerance**: Keep garbled text and noisy transcriptions for data preservation
   - **Simple Garbage Filter**: Remove only non-English characters while preserving numbers and symbols
   - Recognize music using Shazamio API for commercial music identification
   - Detect audio events (sound, music, speech, ambient) with temporal segmentation
   - Save results as JSON, TXT, and CSV with structured data format

4. **Keyframe Extraction**:
   - Extract I-frames and sampled frames (1 each second) for comprehensive coverage
   - Extract highlight frames from speech segments for content-aware selection
   - Save all frames in `data/tiktok_frames/<video_name>/` with organized structure

5. **Frame Filtering & Selection**:
   - Remove similar frames using SSIM similarity threshold (> 0.8) to reduce redundancy
   - Filter black/blur frames using computer vision techniques for quality assurance
   - Detect objects using YOLO-World for comprehensive object recognition
   - **Optimized Frame Selection**: Select 2-5 representative frames using CLIP and product prompts
   - **Image Compression**: Compress frames to reduce GPT4O token usage while maintaining quality
   - Save representative frames and timestamps for downstream analysis

6. **Multimodal Analysis**:
   - Extract YOLO, BLIP, and CLIP features for each representative frame
   - **GPT4O Integration**: Advanced multimodal analysis with OpenAI's latest model
   - **Enhanced Analysis Dimensions**:
     - Content creative ideas and concepts
     - Emotional value and appeal
     - Selling points and product features
     - Target user demographics
     - Video description and categorization
   - **Prompt Monitoring**: Log actual prompts sent to GPT4O for debugging
   - **Robust JSON Parsing**: Handle code block responses and truncated outputs
   - Aggregate results across frames to create comprehensive video understanding
   - Generate structured output with confidence scores and category hierarchies

7. **Result Saving**:
   - Save all features and analysis to `video_features_results.csv` with standardized format
   - Generate additional output files for specific use cases (JSON, TXT)
   - Maintain processing logs and error reports for debugging

8. **Batch & Single Video Support**: Process single videos or entire folders with configurable batch sizes and progress tracking.

---

## Project Structure
```
tiktok/
├── src/
│   ├── audio_processor.py         # Audio extraction, separation, speech/music/event analysis
│   ├── video_processor.py         # Video metadata and keyframe extraction
│   ├── frame_analyzer.py          # Frame filtering, blur/black detection, representative selection
│   ├── multimodal_extractor.py    # YOLO, BLIP, CLIP, GPT4O analysis with enhanced prompts
│   ├── tiktok_feature_extractor.py# Main pipeline controller
│   ├── config_manager.py          # Configuration management and YAML parsing
│   ├── shared_models.py           # Shared model loading/utilities/constants
│   └── image_compressor.py        # Image compression for token optimization
├── data/
│   ├── tiktok_videos/             # Input videos (MP4 format)
│   └── tiktok_frames/             # Output frames and analysis results
├── models/                        # Model weights (Qwen, YOLO, Whisper, etc.)
├── config/                        # Configuration files (YAML format)
├── test/                          # Comprehensive test suite with 90%+ coverage
├── logs/                          # Processing logs and error reports
├── requirements.txt               # Python dependencies with version specifications
├── Dockerfile                     # Docker build with CUDA support
├── docker-compose.yml             # Docker Compose with Jupyter and GPU acceleration
├── tiktok_video_project.ipynb     # Interactive Jupyter notebook for exploration
├── test_notebook.ipynb            # Testing and validation notebook
└── README.md                      # This documentation file
```

---

## Installation and Setup

**System Requirements:**
- Python 3.8 or higher with pip package manager
- FFmpeg for video and audio processing (system dependency)
- CUDA 11.8+ for GPU acceleration (optional but recommended)
- 8GB+ RAM for processing large video batches
- 10GB+ disk space for models, outputs, and temporary files
- OpenAI API key for GPT4O multimodal analysis

**Python Dependencies:**
The system requires several specialized libraries including PyTorch for deep learning models, OpenCV for computer vision, Whisper for speech recognition, Demucs for audio separation, Transformers for multimodal models, and OpenAI for GPT4O integration. All dependencies are specified in requirements.txt with compatible versions.

**Docker Deployment:**
For containerized deployment, the system provides Docker and Docker Compose configurations with GPU support. The Docker setup includes Jupyter notebook access for interactive development and testing.

**Configuration Management:**
The system uses YAML-based configuration files for flexible parameter tuning. Configuration covers processing parameters, model settings, output formats, logging preferences, and GPT4O API settings.

---

## Configuration

**Key Configuration Options:**
- **GPT4O Settings**: API key, model parameters, max tokens (2048), temperature
- **Image Compression**: Max dimension (1024px), quality (85%), compression algorithm
- **Frame Selection**: Number of representative frames (2-5), selection criteria
- **Audio Processing**: Whisper model size, noise tolerance, garbage text filtering
- **Analysis Dimensions**: Content creative ideas, emotional value, selling points, demographics

**Example Configuration:**
```yaml
gpt4o:
  api_key: "your-openai-api-key"
  max_tokens: 2048
  temperature: 0.7

image_compression:
  max_dimension: 1024
  quality: 85

frame_selection:
  max_frames: 5
  similarity_threshold: 0.8

audio_processing:
  whisper_model: "base"
  keep_garbled_text: true
  garbage_filter: true
```

---

## Jupyter Notebook
Use the provided notebook for interactive exploration:
```bash
jupyter notebook tiktok_video_project.ipynb
```

The notebook provides step-by-step examples for single video processing and batch analysis, with detailed explanations of each processing stage and output interpretation.

---

## Output Features
The extractor generates the following for each video:
- **Video Metadata**: duration, resolution, frame rate, file size with validation
- **Audio Analysis**: speech detection with confidence scores, transcription with timestamps (including noisy/garbled text), music recognition with artist/title information, event detection with temporal segmentation
- **Speech Rate Analysis**: Comprehensive speech quality evaluation with 1-10 scoring system
- **Visual Analysis**: keyframe count with quality metrics, representative frames with selection criteria, object detection with bounding boxes and confidence scores
- **Multimodal Analysis**: YOLO/BLIP/CLIP features with embeddings, GPT4O video description with enhanced analysis dimensions
- **Enhanced Analysis Dimensions**:
  - Content creative ideas and concepts
  - Emotional value and appeal
  - Selling points and product features
  - Target user demographics
  - Video description and categorization
- **Structured Output**: CSV/JSON/TXT formats with standardized schemas for easy integration

**Example Output Files:**
- `video_features_results.csv`: Comprehensive feature matrix for all processed videos
- `speech_transcription.txt`: Time-stamped speech text with confidence scores (including noisy text)
- `audio_analysis_results.json`: Detailed audio analysis with event timelines and speech rate analysis
- `speech_rate_analysis.json`: Detailed speech rate analysis with scoring breakdown
- `representative_timestamps.json`: Selected frame timestamps with selection rationale
- `processing_logs.txt`: Detailed processing logs with performance metrics and GPT4O prompts

**Data Quality:**
The system implements multiple quality assurance measures including frame similarity filtering, blur detection, audio quality assessment, confidence scoring for all AI-generated outputs, and robust error handling for API calls.

---

## Speech Rate Analysis

The system includes comprehensive speech rate analysis that evaluates speech quality and provides a 1-10 scoring system for each video.

### Scoring System

The overall score (1-10) is calculated using weighted components:

- **Speech Rate (40%)**: How well the speaking pace matches optimal ranges
- **Consistency (25%)**: How consistent the speech rate is throughout the video  
- **Clarity (20%)**: Based on transcription confidence
- **Pause Analysis (15%)**: Quality and timing of pauses

### Speech Rate Categories

| Category | WPM Range | Score Range | Description |
|----------|-----------|-------------|-------------|
| Very Slow | 0-120 | 1-6 | Too slow for engagement |
| Slow | 120-150 | 6-8 | Slightly slow |
| Normal | 150-180 | 8-10 | Optimal range |
| Fast | 180-220 | 7-9 | Slightly fast |
| Very Fast | 220-300 | 4-7 | Too fast |
| Extremely Fast | 300+ | 1-4 | Very difficult to follow |

### Consistency Scoring

| Coefficient of Variation | Score | Description |
|-------------------------|-------|-------------|
| < 0.1 | 9-10 | Excellent consistency |
| 0.1-0.2 | 7-9 | Good consistency |
| 0.2-0.3 | 5-7 | Moderate consistency |
| > 0.3 | 1-5 | Poor consistency |

### Pause Analysis

- **Optimal pause percentage**: 10-20% of total duration
- **Optimal pause duration**: 0.5-1.5 seconds
- **Scoring based on**: Both percentage and duration

### Output Format

The speech rate analysis returns comprehensive metrics:

```python
{
    "speech_rate_wpm": 165.5,           # Words per minute
    "syllables_per_minute": 248.2,      # Syllables per minute
    "rate_category": "normal",           # Category name
    "rate_score": 9.2,                 # Rate-specific score (1-10)
    "consistency_score": 8.5,          # Consistency score (1-10)
    "pause_score": 7.8,                # Pause quality score (1-10)
    "clarity_score": 8.9,              # Clarity score (1-10)
    "overall_score": 8.6,              # Final weighted score (1-10)
    "detailed_report": "...",           # Human-readable report
    "metrics": {                        # Raw metrics
        "total_duration": 45.2,
        "word_count": 124,
        "syllable_count": 187,
        "speech_segments": [...],
        ...
    }
}
```

### Use Cases

- **Content Creators**: Optimize speech rate for better audience engagement
- **Content Analysis**: Identify high-quality and low-quality speech content
- **Education**: Evaluate presentation and communication skills
- **Research**: Study speech patterns in social media content

---

## Module and Class Descriptions

**audio_processor.py**: 
Comprehensive audio processing module handling extraction, separation using Demucs for vocal isolation, speech recognition with OpenAI Whisper supporting multiple languages, music recognition via Shazamio API, and event detection with temporal segmentation. Includes noise reduction, voice activity detection, audio quality assessment, enhanced noise tolerance for garbled text preservation, and **speech rate analysis** with 1-10 scoring system based on rate, consistency, clarity, and pause analysis.

**video_processor.py**: 
Video metadata extraction using FFmpeg with format validation, keyframe extraction with configurable sampling rates, and highlight frame detection based on speech segments. Supports multiple video formats and implements efficient frame extraction algorithms.

**frame_analyzer.py**: 
Advanced frame analysis including similarity filtering using SSIM algorithm, blur and black frame detection with quality thresholds, YOLO object detection with multiple model support, and CLIP-based representative frame selection with product-aware prompts.

**multimodal_extractor.py**: 
Multimodal feature extraction integrating YOLO for object detection, BLIP for image captioning, CLIP for similarity analysis, and GPT4O for advanced video understanding. Generates comprehensive video descriptions and enhanced analysis dimensions with confidence scoring, prompt monitoring, and robust JSON parsing.

**image_compressor.py**: 
Image compression utility for optimizing GPT4O token usage. Implements configurable compression algorithms with quality and dimension controls, following OpenAI's token calculation guidelines.

**tiktok_feature_extractor.py**: 
Main pipeline controller orchestrating all processing stages with error handling, progress tracking, and batch management. Supports both single video and folder processing with configurable parameters and output formats.

**config_manager.py**: 
Configuration management system with YAML parsing, environment variable support, and validation. Provides centralized access to all system parameters and model configurations.

**shared_models.py**: 
Shared model loading utilities with memory management, model caching, and GPU optimization. Includes constants, product definitions, and category hierarchies for consistent analysis.

---

## Performance and Optimization

**Processing Speed:**
- Single video (15 seconds): approximately 45 seconds processing time
- Batch processing: approximately 2.5 seconds per video with GPU acceleration
- Memory usage: peak usage of 4GB with automatic cleanup
- GPU utilization: 85-95% during intensive processing stages
- **Token Optimization**: Image compression reduces GPT4O token usage by 60-80%

**Scalability Features:**
- Concurrent processing support for up to 4 videos simultaneously
- Optimal batch size of 10 videos for memory efficiency
- Automatic memory management with garbage collection
- Configurable processing parameters for different hardware configurations
- **API Cost Optimization**: Image compression and frame reduction minimize GPT4O API costs

**Quality Assurance:**
- Comprehensive error handling with detailed logging
- Input validation for video formats and file integrity
- Output validation with confidence scoring
- Performance monitoring and resource usage tracking
- **Prompt Monitoring**: Log actual GPT4O prompts for debugging and optimization

---

## Testing and Validation

**Test Coverage:**
The system includes comprehensive test suites covering all modules with 90%+ code coverage. Tests include unit tests for individual components, integration tests for pipeline workflows, and performance tests for optimization validation.

**Test Categories:**
- Audio processing tests with various audio formats and quality levels
- Video processing tests with different resolutions and frame rates
- Frame analysis tests with synthetic and real-world images
- Multimodal extraction tests with diverse content types
- GPT4O integration tests with prompt validation
- Image compression tests with quality assessment
- Integration tests for end-to-end pipeline validation
- Performance tests for memory usage and processing speed

**Validation Procedures:**
- Automated test execution with continuous integration support
- Manual validation procedures for quality assurance
- Performance benchmarking with standardized test datasets
- Error scenario testing for robust error handling
- API response validation for GPT4O integration

---

## Recent Improvements

**Latest Enhancements:**
- **Speech Rate Analysis**: Comprehensive speech quality evaluation with 1-10 scoring system based on rate, consistency, clarity, and pause analysis
- **GPT4O Integration**: Advanced multimodal analysis with OpenAI's latest model
- **Enhanced Analysis Dimensions**: Content creative ideas, emotional value, selling points, target demographics
- **Image Compression**: Configurable compression to optimize token usage and reduce API costs
- **Noise Tolerance**: Preserve garbled text and noisy transcriptions for data completeness
- **Simple Garbage Filter**: Remove only non-English characters while preserving numbers and symbols
- **Prompt Monitoring**: Log actual prompts sent to GPT4O for debugging and optimization
- **Robust JSON Parsing**: Handle code block responses and truncated outputs from GPT4O
- **Configuration-Driven**: All settings configurable via YAML files
- **Error Handling**: Improved error handling for API calls and JSON parsing
- **Token Optimization**: Reduced representative frames and image compression for cost efficiency

**Removed Features:**
- Competitive analysis (due to hallucination issues)
- Strict speech filtering (to preserve more transcription data)
- Token estimation code (rely on API-reported usage)

---

## Requirements
- Python 3.8+ with pip package manager
- FFmpeg for video and audio processing (system dependency)
- CUDA 11.8+ for GPU acceleration (optional but recommended for production use)
- 8GB+ RAM for processing large video batches
- 10GB+ disk space for models, outputs, and temporary files
- OpenAI API key for GPT4O multimodal analysis
- See `requirements.txt` for complete Python dependency list with version specifications

**Optional Dependencies:**
- NVIDIA GPU with CUDA support for accelerated processing
- Additional storage for large model weights and extensive video datasets
- Network connectivity for music recognition API calls and GPT4O API access