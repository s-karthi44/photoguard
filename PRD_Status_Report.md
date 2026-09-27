# PhotoGuard AI - Product Requirements Document & Status Report

## 1. Project Overview & Objective

PhotoGuard AI is a multi-modal forensic detection pipeline designed to distinguish between Real and AI-generated/manipulated images. The core pipeline utilizes a Spatial & Frequency CNN combined with specialized forensic modules (ELA, PRNU, FFT, and Metadata analysis) to produce a unified, explainable verdict via a Flutter mobile application.

## 2. Project Analysis: Implemented vs. Planned

The project originally followed a compressed 12-week schedule. Here is the analysis of the current implementation state against the PRD requirements:

### ✅ Successfully Implemented Core Features

- **Infrastructure & Datasets:** Dockerized environment, virtual environments, and dataset downloading scripts (`scripts/`) are fully set up. Data splitting and metadata extraction are complete.
- **Forensic Sub-Detectors:**
  - CNN Architecture (`modules/cnn`) trained and weights saved (`best_model.pth`).
  - Error Level Analysis (`modules/ela.py`)
  - PRNU Camera Noise (`modules/prnu.py`)
  - FFT Artifact Detection (`modules/fft.py`)
  - Metadata Verifier (`modules/metadata.py`)
- **Fusion Engine:** `modules/fusion.py` is actively weighting the scores from all sub-detectors based on predefined logic (e.g., handling PNGs differently than JPEGs).
- **Explainability:** Grad-CAM (`modules/cnn/gradcam.py`) is fully implemented.
- **Backend & Frontend:** FastAPI (`api.py`) is fully functional. The Flutter app (`photoguard_app`) has been initialized with significant UI development.

### ❌ Pending/Missing Requirements

- **Uncertainty Handling:** The fusion engine currently forces a binary "AI" or "Real" decision. The required "Uncertain" fallback logic (e.g., for confidence scores around 50%) is missing.
- **Ablation & Robustness Studies:** Scripts to test degradation (JPEG compression, resizing) and module combinations are not yet developed.
- **CLIP Integration:** The semantic CLIP module, mentioned as an optional "nice-to-have", has not been implemented.
- **Documentation:** Formal logging of known limitations (e.g., PRNU degradation under compression) is missing.

---

## 3. Plan Deviation: Given Plan vs. Actual Changes

During development, the team has deviated from the strict chronological order of the 12-week plan. Below are the key differences between the **Given Plan** and the **Actual Implementation**:

### 1. Accelerated Backend & API Integration

- **Given Plan:** The FastAPI backend, Grad-CAM API integration, and full system endpoint were scheduled for **Month 3 (Weeks 10-11)**.
- **Actual Change:** The backend (`api.py`) is already fully developed. Grad-CAM generation and base64 encoding are already integrated into the prediction pipeline. The team prioritized having a working end-to-end prototype API over completing all testing phases first.

### 2. Delay in Formal Testing & Reporting

- **Given Plan:** Ablation studies and robustness testing (JPEG compression, unseen generators) were supposed to be completed in **Weeks 7 and 8**, followed by a consolidated decision report.
- **Actual Change:** These testing scripts have not been created yet. The team has built the entire core pipeline and fusion script without formally running or documenting the ablation study results.

### 3. Module Prioritization (FFT included, CLIP skipped)

- **Given Plan:** Week 7 required preparing prototype modules for both CLIP and DWT/FFT. Week 9 involved deciding whether to include them based on latency/accuracy gains.
- **Actual Change:** `fft.py` was implemented and actively integrated into the `fusion.py` engine with a 15% weight. However, **CLIP has been entirely omitted** from the codebase. The team implicitly decided to skip CLIP (likely due to latency/overhead concerns) and move forward with FFT.

### 4. Flutter App Workload Split Execution

- **Given Plan:** The original plan had Vinay soloing the Flutter app. The revised plan split this between Vinay (Backend/API integration) and Karthikeyan (UI Screens), while Jeeva focused on the Core Pipeline.
- **Actual Change:** This strategy has been successfully executed. The `photoguard_app/lib/main.dart` is heavily populated, and the core pipeline is thoroughly built out. The team successfully managed this workload change to avoid bottlenecks and deliver the app prototype faster.

---

## 4. Next Actions to Fulfill PRD

To completely align the current codebase with the final project goals, the following actions are recommended:

1.  **Update `fusion.py`:** Add the 'Uncertain' fallback logic for edge-case predictions (e.g., when the fusion score is between 0.45 and 0.55).
2.  **Create Testing Suite:** Write scripts in `scripts/` to automate JPEG compression and resize robustness tests to prove the pipeline's reliability.
3.  **Formalize the Decision:** Officially document the decision to drop the CLIP module to save latency, fulfilling the Week 9 decision requirement.
