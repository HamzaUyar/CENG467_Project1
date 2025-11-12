# CENG467 Project 1: CoMAT, Shapley Value Analysis & GRPO Fine-tuning

This project implements a comprehensive pipeline for evaluating and fine-tuning large language models (LLMs) using CoMAT (Chain-of-Thought Mechanism Analysis Tool), Shapley value decomposition, and GRPO (Generalized Reward-based Policy Optimization) techniques on college mathematics problems from the MMLU-Redux dataset.

## Project Overview

**Course:** CENG467 - Advanced Topics in Machine Learning  
**Focus Areas:**
- Instruction-guided model evaluation (CoMAT prompt engineering)
- Feature importance analysis via Shapley values
- Reinforcement learning-based fine-tuning (GRPO)

## File Descriptions

### Core Implementation Files

#### `main.py`
Main experiment runner orchestrating the complete evaluation pipeline. Handles:
- Loading MMLU-Redux college mathematics dataset
- Configuring model parameters (temperature, max tokens, etc.)
- Iterating through questions with CoMAT prompt instructions
- Logging predictions and evaluating model responses
- Organizing results by experiment variant

**Key Functions:**
- Dataset loading and preprocessing
- Experiment execution loop with progress tracking
- Result serialization to JSON and CSV formats

#### `mmlu_redux.py`
MMLU-Redux dataset handling and answer extraction utilities:
- Loads college mathematics questions from CSV dataset
- Implements regex-based answer parsing to extract multiple-choice answers (A-E) from model outputs
- Provides question-answer pair management
- Handles edge cases in response parsing (e.g., invalid formats, malformed outputs)

**Key Functions:**
- `load_mmlu_dataset()`: Loads questions from CSV
- `extract_answer()`: Extracts multiple-choice answer from model response using regex patterns

#### `utils.py`
Low-level model interfacing utilities:
- **`predict_model(messages, configuration)`**: Main inference function
  - Inputs: 
    - `messages`: List of role-content dictionaries for chat template
    - `configuration`: Dict with `temperature` and `max_token_limit`
  - Outputs: Model-generated response string
  - Supports: QWEN2/QWEN3 Instruct models via Hugging Face transformers
  - Implements: Chat template application, device placement (CUDA/CPU), generation with specified parameters
  
**Implementation Details:**
- Uses `transformers` library for model loading and inference
- Applies chat templates for prompt formatting
- Handles device management (GPU/CPU fallback)
- Respects configuration parameters for generation quality/diversity

#### `shapley_value_evaluation.py`
Shapley value analysis for feature importance decomposition:
- Analyzes contribution of individual problem-solving steps to final accuracy
- Uses game theory framework to compute fair feature attribution
- Loads pre-computed step combinations and accuracy labels from `evaluation_with_steps.csv`
- Implements algorithms for Shapley value calculation

**Key Functions:**
- Shapley value computation for step-based feature importance
- Coalition value computation across step combinations
- Statistical analysis of feature contributions

#### `CoMAT_Instruction.py`
Chain-of-Thought prompt instructions management:
- Loads and stores the CoMAT instruction template
- Provides structured prompts guiding models to show reasoning steps
- Reads instruction content from `prompt-instruction.txt`
- Supplies `INSTRUCTION` variable for use in main evaluation loop

#### `grpo_finetune.py`
GRPO (Generalized Reward-based Policy Optimization) fine-tuning pipeline:
- Implements reinforcement learning-based model fine-tuning
- Uses reward signal derived from MMLU-Redux accuracy
- Integrates with `trl` library for RLHF/GRPO algorithms
- Manages training loops, policy updates, and model checkpointing

**Key Components:**
- Reward model based on answer correctness
- Policy model initialization and optimization
- Training configuration and logging

### Data Files

#### `mmlu-redux-college_mathematics_dataset.csv`
Primary evaluation dataset containing:
- College-level mathematics questions
- Multiple-choice options (A-E)
- Ground truth answers
- Question metadata

#### `evaluation_with_steps.csv`
Shapley value analysis dataset:
- Problem-solving step combinations (coalition scenarios)
- Accuracy labels (0/1) for each step combination
- Used for computing Shapley value contributions

#### `prompt-instruction.txt`
CoMAT prompt instruction template:
- Detailed instructions for model to generate step-by-step reasoning
- Formatted for integration into model messages
- Enhances model performance through Chain-of-Thought prompting

### Output Directories

#### `cevaplar/` (Answers)
Organized results from different experimental configurations:
- Individual answer files (`.txt`) for specific questions
- Subdirectories for different model/temperature combinations:
  - `comat_qwen2_temp0.1_v1/`: QWEN2 model with low temperature (deterministic)
  - `comat_qwen2_temp0.7_v1/`: QWEN2 model with higher temperature (diverse)
  - `comat_qwen3_mt2000_v1/`: QWEN3 model with 2000 max tokens
  - `comat_qwen3_mt4000_v1/`: QWEN3 model with 4000 max tokens
- JSON logs and detailed experiment results

#### `final_results/`
Aggregated and processed results:
- Performance metrics per configuration
- Comparative analysis outputs
- Visualization-ready data formats

#### `MMLU-Redux-college_mathematics_prompts/`
Processed prompt files:
- `comat.txt`: CoMAT-formatted prompts for all questions

### Configuration & Documentation

#### `requirements.txt`
Project dependencies:
- `transformers`: Model loading and inference
- `torch`: Deep learning framework with CUDA support
- `datasets`: Dataset utilities
- `huggingface_hub`: Model repository access
- `trl`: Transformer Reinforcement Learning library
- `pandas`, `numpy`: Data processing
- `tqdm`: Progress bars
- `sympy`: Symbolic mathematics
- `python-dotenv`: Environment configuration

#### `CENG467_Proje1_Plan.md`
Detailed project planning document (Turkish):
- Step-by-step implementation guide
- STUB completion instructions
- Experiment execution plan
- Hardware/environment setup guidance

#### `REPORT_TASLAK.md`
Report template and outline for project documentation

## Getting Started

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (optional but recommended)
- 16GB+ RAM

### Installation

1. **Clone/Setup workspace:**
   ```bash
   cd /path/to/CENG467_code_template
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify GPU (if available):**
   ```python
   python -c "import torch; print('GPU Available:', torch.cuda.is_available())"
   ```

### Quick Start

Run the main evaluation pipeline:
```bash
python main.py
```

This will:
- Load the MMLU-Redux dataset
- Apply CoMAT instructions via QWEN models
- Generate predictions with configured parameters
- Save results to `cevaplar/` directory
- Compute evaluation metrics

## Project Workflow

1. **CoMAT Evaluation (Q2.a-Q2.g):** Generate model responses with Chain-of-Thought prompting
2. **Shapley Analysis (Q2.h-Q2.j):** Decompose feature contributions using Shapley values
3. **GRPO Fine-tuning (Q4):** Optimize model using reinforcement learning
4. **Result Analysis:** Evaluate improvements and generate comprehensive report

## Results Organization

- Each experiment variant produces:
  - Individual question answer files
  - Aggregated JSON results
  - CSV evaluation logs with accuracy metrics
  - Step-by-step reasoning traces

## Notes

- For running on resource-limited systems, consider Google Colab with GPU support
- Experiment results are saved with timestamps and configuration identifiers
- CoMAT prompts enforce Chain-of-Thought reasoning for improved performance
- Shapley analysis requires pre-computed step combinations

