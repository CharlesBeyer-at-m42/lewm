# LeWM

LeWM is a C++20 rewrite, on ggml, of the LpWM world model. This glossary fixes the vocabulary shared by the C++ code, the Python exporter, and the reference implementation.

## Language

### Projects

**LeWM**:
The C++20 world model built in this workspace. Names the codebase and the model it implements.
_Avoid_: the port, the C version, lpwm-cpp
_Caution_: the reference implementation's source comments also say "LeWM" for the upstream PyTorch project (lucas-maes/le-wm) whose transformer blocks LpWM descends from. When that project is meant, write "le-wm (Maes)".

**LpWM**:
The world model described in the paper "A Case for Sparse Representations in World Models". LeWM implements this model.
_Avoid_: LeWM (when meaning the paper's model), JEPA (too broad)

**Reference implementation**:
The upstream PyTorch codebase `lpworldmodel`, cloned beside LeWM. It is the source of truth for what LeWM must compute and is kept as close to pristine as possible.
_Avoid_: the Python version, upstream (ambiguous with ggml upstream)

**Baseline**:
The DINO-WM comparison model that also lives in the reference implementation: a frozen DINOv2 encoder with a separate predictor and pixel decoder. Out of scope for LeWM.
_Avoid_: concat model, DINO path

### Model

**World model**:
The composition of encoder, action encoder, predictor, and link that predicts the next code from past codes and actions.

**Encoder**:
The vision transformer that maps one observation to one or more tokens. Trained from scratch; not a pretrained backbone.
_Avoid_: backbone, ViT (as a name for the component)

**Action encoder**:
The small network that maps one chunked action to an action code the predictor conditions on.
_Avoid_: embedder

**Predictor**:
The causal transformer that maps a history window of codes plus action codes to the next code. Conditioned on actions through adaptive layer normalisation.
_Avoid_: dynamics model, transition model

**Predictor ladder**:
The family of linear-dynamics predictors used for the paper's ablations. Out of scope for LeWM.

**Link**:
The fixed, parameter-free nonlinearity applied to both encoder and predictor outputs before they are compared. It is what makes codes sparse and non-negative.
_Avoid_: activation, rectifier, h

**Code**:
The vector the world model reasons in: the link applied to an encoder or predictor output. One code per token.
_Avoid_: latent, embedding, representation, feature

**RDMReg**:
The regulariser that pushes codes toward a rectified generalised Gaussian distribution. Training only.
_Avoid_: the reg, SWD loss

**Proprio**:
Proprioceptive state from the environment. The reference implementation encodes and checkpoints it but never uses it in the LpWM path. LeWM is vision-and-action only and has no proprio.

### Data

**Observation**:
One rendered image from an environment at one time step.
_Avoid_: obs, visual, image (when meaning the model input)

**Frame**:
One time step in a sequence fed to the world model. A frame holds one observation and one chunked action.

**Token**:
One encoder output position for one frame. In CLS mode a frame has one token; in patch mode it has one per image patch.

**CLS mode / Patch mode**:
The two encoder output modes. CLS mode emits the class token only. Patch mode emits every patch token. LeWM supports both; CLS mode is the default.

**Chunked action**:
The action vector for one frame, formed by concatenating the environment's raw actions across the skipped steps between frames.
_Avoid_: frameskip action, flattened action

**History window**:
The number of past frames the predictor sees when predicting the next code.
_Avoid_: num_hist, context length

**Rollout**:
Repeated application of the predictor from one starting observation and a sequence of chunked actions, producing predicted codes without further observations.
_Avoid_: imagination, dream

**Environment**:
A simulator that produces observations and accepts actions. Two exist: Wall and PushT. Environments stay in Python; LeWM does not implement them.

### Training

**Gradient parity**:
A test that runs one fixed batch, with identical weights and identical random numbers, through LeWM and the reference implementation and fails if any parameter's gradient differs by more than a tolerance. The first gate of the training milestone.

**Trajectory parity**:
A test that trains both implementations for a short, deterministic run on identical batches and fails if the loss curves diverge. The second gate.

**Batch plan**:
A file listing, per optimizer step, which episode-and-start windows form the batch. Produced by the reference implementation so LeWM can consume identical batches; without one LeWM shuffles for itself.

**Optimizer sidecar**:
The GGUF file saved beside a model file during training that holds the AdamW moments and step count needed to resume.

**Frame file**:
The exported training data: every observation as uint8 pixels in one memory-mapped file, with chunked actions in a companion file. Normalisation and frameskip slicing happen in LeWM.

### Files and tests

**Model file**:
A GGUF file holding LeWM's weights and hyperparameters, produced by the exporter from a reference checkpoint.
_Avoid_: checkpoint (reserved for the reference implementation's files), weights file

**Exporter**:
The Python tool that reads a reference checkpoint and writes a model file.
_Avoid_: converter

**Reference dump**:
A file of inputs and intermediate and final outputs produced by the reference implementation for one fixed input and one model file.

**Parity test**:
A test that runs LeWM on a reference dump's inputs and fails if any output differs from the dump by more than a tolerance. Parity is defined against weights, not against trained behaviour, so randomly initialised weights are valid.
