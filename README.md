
# ShobdoTori: Regional to Standard Bangla Speech Recognition

## Overview

**ShobdoTori** is an advanced Automatic Speech Recognition (ASR) system that transcribes regional Bangladeshi dialects into standard Bangla text. This project was developed as part of a prestigious AI hackathon organized by the Department of Electronics and Telecommunication Engineering (ETE), Chittagong University of Engineering & Technology (CUET).

The system leverages fine-tuned Whisper models to handle the unique phonetic challenges of Bangladesh's 20 regional dialects, converting dialectal speech into formal Bangla text with high accuracy.

---

## Competition Details

### Event Information
- **Event Name:** শব্দতরী: Where Dialects Flow into Bangla
- **Organizer:** Department of Electronics and Telecommunication Engineering (ETE), Chittagong University of Engineering & Technology (CUET)
- **Part of:** Televerse 1.0
- **Sponsor:** DOREEN POWER
- **Co-Host:** IEEE CS, CUET Student Branch Chapter

### Competition Timeline
- **Start Date:** November 6, 2025
- **End Date:** November 17, 2025
- **Prize Pool:** 70,000 BDT

### Challenge Description
Participants developed ASR models to transcribe regional Bangladeshi dialects into standard Bangla text. The competition featured:
- **Dataset:** 3,800 audio recordings
- **Regional Coverage:** 20 regional dialects across Bangladesh
- **Evaluation Metric:** Normalized Levenshtein Similarity
- **Evaluation Method:** Hybrid evaluation with top 20 teams required to submit notebooks

---

## Dataset

The dataset comprises audio recordings from 20 regional dialects across Bangladesh:

**Covered Regions:**
Barisal, Bhola, Bogura, Brahmanbaria, Chittagong, Comilla, Dhaka, Feni, Jessore, Jhenaidah, Khulna, Kushtia, Lakshmipur, Mymensingh, Natore, Noakhali, Pabna, Rajshahi, Rangpur, Sylhet

**Dataset Structure:**
- **Training Set:** 3,015 audio samples with transcriptions
- **Test Set:** 335 audio samples
- **Audio Format:** WAV files at 16 kHz sampling rate
- **Transcriptions:** Standard Bangla text in UTF-8 encoding

---

## Evaluation Metric

### Normalized Levenshtein Similarity

The competition uses Normalized Levenshtein Similarity to measure transcription accuracy:

```
Similarity = 1.0 - (Levenshtein Distance / Max(Reference Length, Prediction Length))
Final Score = (Σ Similarity for each sample) / Total number of samples
```

**Score Interpretation:**
- **Score = 1.0:** Perfect transcription (no errors)
- **Score > 0.8:** Excellent transcription quality
- **Score < 0.5:** Poor transcription quality

**Example Calculation:**
```
Sample 1: Reference = "আমি ভাত খাই" | Prediction = "আমি ভাত খাই"
Distance = 0 | Similarity = 1.0000

Sample 2: Reference = "সে স্কুলে যায়" | Prediction = "সে স্কুলে যায"
Distance = 1 | Similarity = 0.9286

Sample 3: Reference = "ঢাকা রাজধানী" | Prediction = "ঢাকা শহর"
Distance = 6 | Similarity = 0.5385

Final Score = (1.0000 + 0.9286 + 0.5385) / 3 = 0.8224
```

---

## Project Architecture

### Model Selection
The project utilizes **Whisper Medium** model, specifically the pre-trained Bengali variant:
- **Base Model:** `openai/whisper-medium`
- **Fine-tuned Version:** `bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium`
- **Original Training:** Ben10 Kaggle Competition dataset

### Key Components

#### 1. **Data Preparation** (`BITWISEMINDS_Regional_WhisperMedium_FineTuning_Train.ipynb`)
- Dataset loading from regional dialect folders
- Audio preprocessing and validation
- Train/test split (90/10 ratio)
- Transcription normalization

#### 2. **Feature Extraction**
- Log-Mel spectrogram computation
- 16 kHz audio resampling
- Standardized feature dimensions

#### 3. **Model Fine-tuning**
- Transfer learning from pre-trained Whisper
- Language-specific tokenization (Bengali)
- Gradient accumulation for memory efficiency
- Mixed precision (FP16) training

#### 4. **Inference** (`BITWISEMINDS_Regional_WhisperMedium_FineTuning_Infer.ipynb`)
- Batch prediction on test set
- Transcription post-processing
- CSV output generation

---

## Training Configuration

### Hyperparameters
```
Learning Rate:              1e-5
Batch Size (per GPU):       4
Gradient Accumulation:      4
Effective Batch Size:       16
Max Training Steps:         4000
Warmup Steps:              200
```

### Memory Optimization
```
Gradient Checkpointing:     Enabled
Mixed Precision (FP16):     Enabled
Optimization Level:         O1
Max Gradient Norm:          1.0
```

### Evaluation Settings
```
Evaluation Strategy:        Every 1000 steps
Prediction Generation:      Enabled
Max Generation Length:      225 tokens
Metric for Best Model:      WER (Word Error Rate)
Save Strategy:              Keep latest checkpoint only
```

---

## Installation & Setup

### Requirements
```
Python >= 3.8
PyTorch >= 2.0 (with CUDA support)
transformers >= 4.45.0
datasets >= 2.16.1
accelerate >= 1.0.0
evaluate >= 0.4.3
jiwer >= 3.0.3
librosa
soundfile
tensorboard
```

### Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/SambasBoyyyy/ShobdoTori-Regional-to-Standard-Bangla-Speech-Recognition.git
cd ShobdoTori-Regional-to-Standard-Bangla-Speech-Recognition
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Prepare dataset:**
   - Obtain the competition dataset
   - Extract audio files and annotations
   - Update paths in the notebooks

---

## Usage

### Training

Open and run `BITWISEMINDS_Regional_WhisperMedium_FineTuning_Train.ipynb`:

1. **Environment Setup:** Verify GPU availability and library versions
2. **Data Preparation:** Load and preprocess regional dialect audio
3. **Model Loading:** Initialize Whisper model and processor
4. **Training:** Fine-tune on regional dialect data
5. **Evaluation:** Monitor WER on validation set

**Expected Training Time:** 3.5-4 hours on Tesla T4 GPU

### Inference

Open and run `BITWISEMINDS_Regional_WhisperMedium_FineTuning_Infer.ipynb`:

1. Load fine-tuned model
2. Process test audio files
3. Generate transcriptions
4. Export predictions to CSV

### Output Format

Predictions should be submitted as CSV with the following format:
```csv
audio,text
test_001.wav,আজ আকাশে মেঘ করেছে
test_002.wav,তুমি কোথায় যাচ্ছ
test_003.wav,আমি বই পড়তে ভালোবাসি
```

**Requirements:**
- UTF-8 encoding
- All 500 test audio files must have predictions
- Transcriptions in standard Bangla script
- Proper Bangla Unicode characters

---

## Key Features

✅ **Regional Dialect Coverage:** Handles 20 distinct Bangladeshi dialects

✅ **Transfer Learning:** Leverages pre-trained Whisper model for faster convergence

✅ **Memory Efficient:** Gradient checkpointing and mixed precision training

✅ **Robust Evaluation:** Normalized Levenshtein Similarity metric

✅ **Production Ready:** Inference pipeline with batch processing

✅ **Comprehensive Logging:** TensorBoard integration for training monitoring

---

## Results & Performance

### Model Performance
- **Training Samples:** 3,015 audio recordings
- **Validation Samples:** 335 audio recordings
- **Evaluation Metric:** Word Error Rate (WER)
- **Best Model:** Automatically selected based on validation WER

### Optimization Techniques
- Gradient accumulation for larger effective batch sizes
- Mixed precision training (FP16) for memory efficiency
- Gradient checkpointing to reduce memory footprint
- Learning rate warmup for stable training
- Early stopping based on validation WER

---

## File Structure

```
ShobdoTori-Regional-to-Standard-Bangla-Speech-Recognition/
├── README.md                                              # Project documentation
├── requirements.txt                                       # Python dependencies
├── BITWISEMINDS_Regional_WhisperMedium_FineTuning_Train.ipynb
│   └── Complete training pipeline with data prep and fine-tuning
├── BITWISEMINDS_Regional_WhisperMedium_FineTuning_Infer.ipynb
│   └── Inference pipeline for test set predictions
└── whisper-medium-bengali-regional/                      # Output directory
    ├── checkpoint-*/                                      # Training checkpoints
    ├── config.json                                        # Model configuration
    └── pytorch_model.bin                                  # Fine-tuned weights
```

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| **Deep Learning Framework** | PyTorch 2.6.0 |
| **Model Architecture** | OpenAI Whisper (Medium) |
| **NLP Library** | Hugging Face Transformers 4.45.0 |
| **Dataset Management** | Hugging Face Datasets 2.16.1 |
| **Training Framework** | Hugging Face Accelerate 1.0.0 |
| **Evaluation Metrics** | jiwer 3.0.3, evaluate 0.4.3 |
| **Audio Processing** | librosa, soundfile |
| **Monitoring** | TensorBoard |
| **Hardware** | NVIDIA Tesla T4 GPU (14.74 GB VRAM) |

---

## Challenges & Solutions

### Challenge 1: Dialectal Variation
**Problem:** Regional dialects have significant phonetic and vocabulary differences
**Solution:** Fine-tuned pre-trained model on dialect-specific data with 20 regional samples

### Challenge 2: Memory Constraints
**Problem:** Limited GPU memory for large batch sizes
**Solution:** Implemented gradient accumulation, mixed precision training, and gradient checkpointing

### Challenge 3: Data Imbalance
**Problem:** Uneven distribution across regional dialects
**Solution:** Loaded all available samples and used stratified evaluation

### Challenge 4: Transcription Accuracy
**Problem:** Dialectal speech recognition requires understanding phonetic nuances
**Solution:** Used pre-trained Bengali model as base and fine-tuned on competition dataset

---

## Future Improvements

- [ ] Ensemble methods combining multiple Whisper variants
- [ ] Language model integration for post-processing
- [ ] Dialect-specific model variants
- [ ] Real-time inference optimization
- [ ] Multi-GPU distributed training
- [ ] Confidence scoring for predictions
- [ ] Interactive web interface for testing

---

## References

### Models & Frameworks
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [bengaliAI Whisper Model](https://huggingface.co/bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium)

### Evaluation Metrics
- [Normalized Levenshtein Distance](https://en.wikipedia.org/wiki/Levenshtein_distance)
- [Word Error Rate (WER)](https://en.wikipedia.org/wiki/Word_error_rate)

### Related Work
- Ben10 Kaggle Competition - Bengali ASR
- Televerse 1.0 - CUET's Technology Festival

---

## License

This project is provided for educational and research purposes. Please refer to the original competition guidelines for usage restrictions.

---

## Acknowledgments

- **CUET Department of ETE** for organizing the competition
- **Televerse 1.0** for providing the platform
- **DOREEN POWER** for sponsorship
- **IEEE CS, CUET Student Branch** for co-hosting
- **bengaliAI** for the pre-trained Whisper model
- **OpenAI** for the Whisper architecture

---

## Contact & Support

For questions or issues related to this project, please refer to the competition guidelines or contact the organizing committee through the official Televerse 1.0 platform.

---

**Last Updated:** November 2025

**Status:** Competition Submission ✅
=======
# ShobdoTori-Regional-to-Standard-Bangla-Speech-Recognition
#Code and Demo will be published soon
>>>>>>> 855a0d13135d135e391a249680cda23a6125d8bd 
