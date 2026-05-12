# BirdCLEF 2023 2nd Place Solution

This project is a deep learning-based solution for the [BirdCLEF 2023 Kaggle competition](https://www.kaggle.com/competitions/birdclef-2023), which focused on identifying bird species in audio recordings. This solution achieved 2nd place and utilizes both CNN and Sound Event Detection (SED) architectures.

## Project Overview

-   **Domain**: Bioacoustics / Audio Classification
-   **Main Technologies**: Python, PyTorch, PyTorch Lightning, `timm` (PyTorch Image Models), `librosa`, `audiomentations`, `wandb`, OpenVINO.
-   **Architecture**:
    -   **Models**: Supports multiple backbones (EfficientNet-V2, ResNet, SeresNext) within either a standard CNN classifier or a SED (Sound Event Detection) framework.
    -   **Pipeline**: Multi-stage training process (Pretraining -> Training -> Finetuning) using Cross-Entropy (CE) and Binary Cross-Entropy (BCE) losses.
    -   **Data Augmentation**: Heavy use of audio-specific augmentations including pitch shifting, noise injection, and Mixup.
    -   **Optimization**: Integrated with Weights & Biases (W&B) for experiment tracking and Model Checkpointing.

## Directory Structure

-   `train.py`: Main entry point for training models.
-   `convert.py`: Script to convert trained PyTorch models to ONNX and OpenVINO formats.
-   `configs/`: Contains configuration files for different model architectures and common settings.
    -   `common.py`: Global configurations (SR, labels, etc.).
    -   `<model_name>.py`: Model-specific hyperparameters.
-   `modules/`: Core logic components.
    -   `model.py`: Model definitions (CNN, SED, Attention blocks).
    -   `dataset.py`: PyTorch Dataset and DataLoader implementation.
    -   `preprocess.py`: Data loading and label preparation.
    -   `augmentations.py`: Custom audio augmentation wrappers.
    -   `utils.py`: Utility functions.
-   `inputs/`: (Expected) Data directory for training audios, metadata, and labels.
-   `outputs/`: Directory where checkpoints and converted models are saved.

## Building and Running

### Prerequisites

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Configuration

Before training, you must set your Weights & Biases API key in `configs/common.py`:

```python
cfg.WANDB_API_KEY = 'your_key_here'
```

### Training

Run the training script by specifying the stage and model name:

```bash
# Example: Training an EfficientNet-B0 CNN model
python3 train.py --stage train_ce --model_name cnn_b0ns
```

**Available Stages**: `pretrain_ce`, `pretrain_bce`, `train_ce`, `train_bce`, `finetune`.
**Available Models**: `sed_v2s`, `sed_b3ns`, `sed_seresnext26t`, `cnn_v2s`, `cnn_resnet34d`, `cnn_b3ns`, `cnn_b0ns`.

### Model Conversion

To convert a trained model to ONNX and OpenVINO:

```bash
python3 convert.py --model_name cnn_b0ns
```

## Development Conventions

-   **Modular Configs**: Hyperparameters are managed through inheritance from `common.py` into model-specific files. Avoid hardcoding values in `modules/`.
-   **PyTorch Lightning**: The project follows PL conventions. The `BirdClefModelBase` in `modules/model.py` handles the training loop logic.
-   **Audio Processing**: All audio is processed at 32kHz (`cfg.SR`).
-   **Reproducibility**: Seeds are defined per stage in the model configurations.
-   **Paths**: Some scripts contain hardcoded paths (e.g., `/content/...`) that may need adjustment for local environments. Check `train.py` and `configs/` if issues arise.
