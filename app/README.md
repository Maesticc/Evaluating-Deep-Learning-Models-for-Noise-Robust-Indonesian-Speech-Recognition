# Local demo UI (later phase)

Deferred by design: the demo (Streamlit, Gradio or FastAPI) is built only after model inference and the
clean-vs-noisy evaluation pipeline are stable. It will reuse `src.model_loader.load_model` and
`src.noise_mixing.mix_at_snr`.
