"""Comprehensive unit tests for the unified Diarizers v1.0.0 package."""

import json
import os
import tempfile
import unittest

import numpy as np

from diarizers import alignment
from diarizers import asr
from diarizers import cli
from diarizers import configs
from diarizers import dataprep
from diarizers import diarization
from diarizers import evaluation
from diarizers import llm
from diarizers import pipeline
from diarizers import spectral
from diarizers import types

if diarization.torch is not None:
  diarization.torch.set_num_threads(4)


class TestTypesAndConfigs(unittest.TestCase):
  """Tests data structures and export formats in types.py and configs.py."""

  def test_presets_and_prompt_config(self) -> None:
    self.assertEqual(configs.FISHER_TELEPHONY_PRESET.min_clusters, 2)
    self.assertEqual(configs.FISHER_TELEPHONY_PRESET.max_clusters, 2)
    self.assertEqual(configs.CALLHOME_TELEPHONY_PRESET.max_clusters, 6)
    self.assertEqual(configs.MEETING_MULTISPK_PRESET.max_clusters, 10)
    self.assertTrue(configs.DEFAULT_PROMPT_CONFIG.use_gemma_turn_template)
    self.assertTrue(configs.DEFAULT_PROMPT_CONFIG.use_locality_gate)
    ws = types.WordSegment(word="hi", start=0.1, end=0.5, speaker="1")
    self.assertEqual(ws.to_dict()["speaker"], "1")

  def test_diarized_transcript_exports(self) -> None:
    words = [
        types.DiarizedWord(
            word="hello", start=0.0, end=0.5, acoustic_speaker="1", speaker="1"
        ),
        types.DiarizedWord(
            word="world", start=0.5, end=1.0, acoustic_speaker="1", speaker="1"
        ),
        types.DiarizedWord(
            word="hi", start=1.2, end=1.6, acoustic_speaker="2", speaker="2"
        ),
    ]
    dt = types.DiarizedTranscript(
        utterance_id="utt_01",
        words=words,
        hyp_text="hello world hi",
        hyp_spk="1 1 2",
        hyp_spk_llm_raw="1 1 2",
        hyp_spk_llm="1 1 2",
        diarized_before_llm="<speaker:1> hello world <speaker:2> hi",
        diarized_after_llm="<speaker:1> hello world <speaker:2> hi",
    )
    self.assertEqual(dt.word_list(), ["hello", "world", "hi"])
    self.assertEqual(dt.speaker_list(), ["1", "1", "2"])
    self.assertIn("<speaker:1> hello world <speaker:2> hi", dt.to_text())
    self.assertIn(
        "<speaker:1> hello world \n <speaker:2> hi",
        dt.to_text(use_new_line=True),
    )
    rttm = dt.to_rttm()
    self.assertIn("SPEAKER utt_01 1 0.000 1.000", rttm)
    self.assertIn("SPEAKER utt_01 1 1.200 0.400", rttm)
    srt = dt.to_srt()
    self.assertIn("00:00:00,000 --> 00:00:01,000", srt)
    self.assertIn("[Speaker 1] hello world", srt)
    vtt = dt.to_vtt()
    self.assertTrue(vtt.startswith("WEBVTT"))
    self.assertIn("<v Speaker 2>hi", vtt)
    parsed = json.loads(dt.to_json())
    self.assertEqual(parsed["utterance_id"], "utt_01")
    self.assertEqual(parsed["hyp_text"], "hello world hi")
    self.assertEqual(parsed["hyp_spk_llm"], "1 1 2")

  def test_empty_transcript(self) -> None:
    dt = types.DiarizedTranscript(utterance_id="empty")
    self.assertEqual(dt.to_text(), "")
    self.assertEqual(dt.to_rttm(), "")
    self.assertEqual(dt.to_srt(), "")
    self.assertEqual(dt.to_vtt(), "WEBVTT")


class TestAlignmentAndLGTPST(unittest.TestCase):
  """Tests word-speaker alignment, TPST, LG-TPST, and chunk reconciliation."""

  def test_normalize_text_and_speakers_transform(self) -> None:
    self.assertEqual(
        alignment.normalize_text("Hello, World! Don't stop..."),
        "hello world dont stop",
    )
    self.assertEqual(alignment.normalize_text(""), "")
    self.assertEqual(
        alignment.speakers_transform(["spk_B", "spk_B", "spk_A", "spk_C"]),
        ["1", "1", "2", "3"],
    )

  def test_assign_speakers_fixes_all_2024_bugs(self) -> None:
    asr_words = [
        {"word": "hello", "timestamp": (0.1, 0.4)},
        {"word": "gapword", "timestamp": (1.05, 1.10)},
        {"word": "inverted", "timestamp": (2.5, 2.2)},
        {"word": "none_end", "timestamp": (2.6, None)},
    ]
    diar_turns = [
        {"segment": {"start": 0.0, "end": 1.0}, "speaker": "SPEAKER_09"},
        {"segment": {"start": 2.0, "end": 3.0}, "speaker": "SPEAKER_02"},
    ]
    spks = alignment.assign_speakers(asr_words, diar_turns)
    self.assertEqual(spks, ["1", "1", "2", "2"])
    self.assertEqual(alignment.assign_speakers([], diar_turns), [])
    self.assertEqual(alignment.assign_speakers(asr_words, []), ["1"] * 4)

  def test_diarized_text_roundtrip_and_malformed_tags(self) -> None:
    words = ["hello", "there", "general", "kenobi"]
    spks = ["1", "1", "2", "2"]
    diar_str = alignment.create_diarized_text(words, spks)
    self.assertEqual(
        diar_str, "<speaker:1> hello there <speaker:2> general kenobi"
    )
    ext_w, ext_s = alignment.extract_text_and_spk(diar_str)
    self.assertEqual(ext_w, "hello there general kenobi")
    self.assertEqual(ext_s, "1 1 2 2")

    malformed = "<speaker:1> hello <speaker:abc> world <speaker:99> friend"
    mw, ms = alignment.extract_text_and_spk(malformed)
    self.assertEqual(mw, "hello world friend")
    self.assertEqual(ms, "1 1 1")

  def test_tpst_preserves_word_count_under_hallucinations(self) -> None:
    tgt_text = "one two three four five six"
    tgt_spk = "1 1 1 2 2 2"
    src_text = "one two three extra four five"
    src_spk = "1 1 2 2 2 2"
    transferred = alignment.transcript_preserving_speaker_transfer(
        src_text=src_text,
        src_spk=src_spk,
        tgt_text=tgt_text,
        tgt_spk=tgt_spk,
    )
    self.assertEqual(len(transferred.split()), 6)
    self.assertEqual(transferred, "1 1 2 2 2 2")

  def test_lg_tpst_absorbs_ghost_and_gates_long_monologue_flips(self) -> None:
    hyp_spks = ["1"] * 20 + ["2"] * 20 + ["3"] * 20 + ["4", "4"]
    llm_spks = (
        ["1"] * 18 + ["2"] * 2 + ["2"] * 5 + ["1"] * 15 + ["3"] * 22
    )
    gated = alignment.apply_locality_gated_speaker_transfer(
        hyp_spks,
        llm_spks,
        max_local_span_words=5,
        max_ghost_cluster_words=12,
    )
    self.assertEqual(len(gated), len(hyp_spks))
    self.assertEqual(gated[18:20], ["2", "2"])
    self.assertEqual(gated[25:40], ["2"] * 15)
    self.assertEqual(gated[60:62], ["3", "3"])

  def test_chunker_and_hungarian_reconciliation(self) -> None:
    words = [f"w{i}" for i in range(30)]
    spks = ["1"] * 15 + ["2"] * 15
    chunker = alignment.UtteranceChunker(
        max_input_length=60, max_target_length=60
    )
    chunks = list(chunker.chunk_utterance_fields(words, spks, spks, "u1"))
    self.assertGreater(len(chunks), 1)
    completions = [c["target"] for c in chunks]
    raw_s, gated_s = alignment.reconcile_chunk_speakers_hungarian(
        words, spks, completions, chunker=chunker
    )
    self.assertEqual(len(raw_s.split()), 30)
    self.assertEqual(gated_s.split(), spks)


class TestSpectralAndDiarization(unittest.TestCase):
  """Tests clean-room SpectralClusterer and ECAPASpectralDiarizer."""

  def test_spectral_clusterer_separates_orthogonal_clusters(self) -> None:
    rng = np.random.default_rng(42)
    c1 = np.tile([1.0, 0.0, 0.0, 0.0], (25, 1)) + 0.03 * rng.normal(
        size=(25, 4)
    )
    c2 = np.tile([0.0, 1.0, 0.0, 0.0], (25, 1)) + 0.03 * rng.normal(
        size=(25, 4)
    )
    embs = np.vstack([c1, c2])
    clusterer = spectral.SpectralClusterer(min_clusters=2, max_clusters=4)
    labels = clusterer.predict(embs, num_clusters=2)
    self.assertEqual(len(labels), 50)
    self.assertEqual(len(set(labels[:25])), 1)
    self.assertEqual(len(set(labels[25:])), 1)
    self.assertNotEqual(labels[0], labels[25])

    auto_labels = clusterer.predict(embs)
    self.assertEqual(len(auto_labels), 50)

    self.assertEqual(len(clusterer.predict(np.zeros((0, 4)))), 0)
    self.assertEqual(list(clusterer.predict(np.ones((1, 4)))), [0])

  def test_ecapa_diarizer_with_mock_encoder_and_audio_io(self) -> None:
    def mock_encoder(batch: np.ndarray) -> np.ndarray:
      b = batch.shape[0]
      out = np.zeros((b, 16), dtype=np.float32)
      half = max(1, b // 2)
      out[:half, 0] = 1.0
      out[half:, 1] = 1.0
      return out

    preset = configs.DiarizationPreset(
        window_frames=20, step_frames=10, min_clusters=2, max_clusters=2
    )
    diar = diarization.ECAPASpectralDiarizer(
        preset=preset, encoder=mock_encoder
    )
    sr = 16000
    t = np.linspace(0, 1.5, int(1.5 * sr), endpoint=False, dtype=np.float32)
    wav = 0.2 * np.sin(2 * np.pi * 440.0 * t)
    turns = diar(wav, num_speakers=2)
    self.assertGreaterEqual(len(turns), 1)
    turns_auto = diar(wav, min_speakers=2, max_speakers=3)
    self.assertGreaterEqual(len(turns_auto), 1)
    self.assertEqual(diar(np.array([], dtype=np.float32)), [])

    # Test load_audio_16k resampling from 8000 Hz tuple and WAV file
    wav_8k = wav[::2]
    resampled = diarization.load_audio_16k((wav_8k, 8000), target_sr=16000)
    self.assertEqual(len(resampled), len(wav))

    if diarization.torch is not None:
      model = diarization.SpeechBrainECAPATDNN(in_mels=80, emb_dim=192)
      model.eval()
      dummy_feats = diarization.torch.zeros((2, 30, 80))
      with diarization.torch.no_grad():
        emb_out = model(dummy_feats)
      self.assertEqual(tuple(emb_out.shape), (2, 192))

      # Test standalone PyanNetSegmentation (pyannote/segmentation-3.0)
      seg_net = diarization.PyanNetSegmentation()
      seg_net.eval()
      dummy_wav = diarization.torch.zeros((2, 1, 16000))
      with diarization.torch.no_grad():
        log_probs = seg_net(dummy_wav)
      self.assertEqual(log_probs.shape[0], 2)
      self.assertEqual(log_probs.shape[2], 7)
      probs = np.exp(log_probs.numpy())
      ml_probs = diarization.powerset_probs_to_multilabel(probs[0])
      self.assertEqual(ml_probs.shape[1], 3)

  def test_pyannote_segmentation_vs_sliding_window_diarizer(self) -> None:
    def mock_encoder(batch: np.ndarray) -> np.ndarray:
      b = batch.shape[0]
      out = np.zeros((b, 16), dtype=np.float32)
      for i in range(b):
        # Switch speaker between windows 0..1 and 2..5
        spk = 0 if i < 2 else 1
        out[i, spk] = 1.0
      return out

    def mock_seg_fn(waveform: np.ndarray):
      del waveform
      return [
          (0.0, 1.45, np.ones(50, dtype=np.float32)),
          (1.45, 3.2, np.ones(50, dtype=np.float32)),
      ]

    sr = 16000
    t = np.linspace(0, 3.2, int(3.2 * sr), endpoint=False, dtype=np.float32)
    wav = 0.2 * np.sin(2 * np.pi * 440.0 * t)

    # 1. Default mode with pyannote/segmentation-3.0 + curve snapping
    diar_pyannote = diarization.ECAPASpectralDiarizer(
        encoder=mock_encoder,
        segmentation_fn=mock_seg_fn,
        segmentation_model="pyannote/segmentation-3.0",
    )
    self.assertEqual(
        diar_pyannote.segmentation_model, "pyannote/segmentation-3.0"
    )
    ml_curves = diar_pyannote.extract_segmentation_curves(wav, sample_rate=sr)
    self.assertIsNotNone(ml_curves)
    assert ml_curves is not None
    self.assertEqual(ml_curves.shape[1], 3)
    turns_pyannote = diar_pyannote(wav, num_speakers=2)
    self.assertEqual(len(turns_pyannote), 2)
    for turn in turns_pyannote:
      self.assertLessEqual(turn.start, turn.end)

    # 2. Pure sliding-window mode ("none", None, "energy_vad")
    for seg_flag in ("none", None, "energy_vad"):
      diar_sliding = diarization.ECAPASpectralDiarizer(
          encoder=mock_encoder,
          segmentation_model=seg_flag,
      )
      self.assertIsNone(diar_sliding.segmentation_model)
      self.assertIsNone(
          diar_sliding.extract_segmentation_curves(wav, sample_rate=sr)
      )
      turns_sliding = diar_sliding(wav, num_speakers=2)
      self.assertEqual(len(turns_sliding), 2)

    # Verify that segmentation_fn snapped the turn boundary near 1.45s
    # whereas pure sliding window used the unsnapped midpoint (1.20s)
    self.assertAlmostEqual(turns_pyannote[0].end, 1.45, delta=0.04)
    self.assertAlmostEqual(turns_sliding[0].end, 1.20, delta=0.01)

    # 3. Monotonicity check under rapid alternating speaker transitions
    times_rapid = [
        (0.0, 1.2),
        (0.4, 1.6),
        (0.8, 2.0),
        (1.2, 2.4),
    ]
    labels_rapid = np.array([0, 1, 0, 1], dtype=np.int32)
    frame_dt = 10.0 / 589.0
    ml_adversarial = np.zeros((200, 3), dtype=np.float32)
    # Place high-flux transition late for idx=1 and early for idx=2
    f_late = int(round(1.12 / frame_dt))
    f_early = int(round(0.88 / frame_dt))
    ml_adversarial[:f_early, 0] = 1.0
    ml_adversarial[f_early:f_late, 1] = 1.0
    ml_adversarial[f_late:, 2] = 1.0
    snapped_bounds = diar_pyannote.snap_turn_boundaries(
        times_rapid, labels_rapid, ml_adversarial
    )
    for b_prev, b_next in zip(snapped_bounds, snapped_bounds[1:]):
      self.assertLessEqual(b_prev, b_next)

    # 4. Direct PyanNetSegmentation module execution when torch is available
    if diarization.torch is not None:
      seg_net = diarization.PyanNetSegmentation()
      seg_net.eval()
      diar_direct = diarization.ECAPASpectralDiarizer(
          encoder=mock_encoder,
          segmentation_model=seg_net,
      )
      ml_direct = diar_direct.extract_segmentation_curves(wav, sample_rate=sr)
      self.assertIsNotNone(ml_direct)
      turns_direct = diar_direct(wav, num_speakers=2)
      self.assertGreaterEqual(len(turns_direct), 1)
      for turn in turns_direct:
        self.assertLessEqual(turn.start, turn.end)

    # 5. Verify ASRConfig & DiarizersConfig defaults
    self.assertEqual(
        configs.DEFAULT_ASR_MODEL, "openai/whisper-large-v3-turbo"
    )
    self.assertEqual(
        configs.DEFAULT_SEGMENTATION_MODEL, "pyannote/segmentation-3.0"
    )
    self.assertEqual(
        configs.DEFAULT_ECAPA_MODEL, "speechbrain/spkrec-ecapa-voxceleb"
    )
    self.assertEqual(
        configs.DEFAULT_DIARIZERS_CONFIG.asr.model_name,
        "openai/whisper-large-v3-turbo",
    )
    self.assertEqual(
        configs.DEFAULT_DIARIZERS_CONFIG.diarization.segmentation_model,
        "pyannote/segmentation-3.0",
    )
    self.assertEqual(
        configs.DEFAULT_DIARIZERS_CONFIG.llm_model_or_path,
        "diarizers-community/DiarizationLM-Gemma-4-E4B-v1",
    )


class TestASRrefinerPipelineAndEval(unittest.TestCase):
  """Tests ASR adapters, DiarizationLMRefiner, DiarizersPipeline, and Eval."""

  def test_turns_to_words_and_precomputed_asr(self) -> None:
    turns = [
        {
            "start": 0.0,
            "end": 1.0,
            "speaker": "1",
            "transcript": "Hello world!",
        },
        {
            "start": 1.2,
            "end": 2.0,
            "speaker": "2",
            "transcript": "Good morning",
        },
    ]
    w_segs = asr.turns_to_word_segments(turns)
    self.assertEqual(
        [w.word for w in w_segs], ["hello", "world", "good", "morning"]
    )
    pre_asr = asr.PrecomputedASR(turns)
    self.assertEqual(len(pre_asr()), 4)
    dict_asr = asr.PrecomputedASR(
        [{"word": "Test!", "start": 0.0, "end": 0.5, "speaker": "1"}]
    )
    self.assertEqual(dict_asr()[0].word, "test")

  def test_whisper_asr_with_mock_processor_and_model(self) -> None:
    class _MockInputs:

      def __init__(self) -> None:
        self.input_features = np.zeros((1, 80, 3000), dtype=np.float32)
        self.attention_mask = np.ones((1, 3000), dtype=np.int32)

    class _MockProc:

      def __call__(self, *args, **kwargs) -> _MockInputs:
        del args, kwargs
        return _MockInputs()

      def batch_decode(self, *args, **kwargs):
        del args, kwargs
        return [
            "<|startoftranscript|><|0.00|> Hello world!<|1.20|>"
            "<|1.20|> How are you?<|2.50|><|endoftext|>"
        ]

    class _MockModel:

      def generate(self, *args, **kwargs):
        del args, kwargs
        return [[1, 2, 3]]

    w_asr = asr.WhisperWordASR()
    w_asr.processor = _MockProc()
    w_asr.model = _MockModel()
    words = w_asr(np.zeros(16000 * 2, dtype=np.float32))
    self.assertEqual(
        [w.word for w in words], ["hello", "world", "how", "are", "you"]
    )

  def test_llm_refiner_and_pipeline(self) -> None:
    refiner = llm.DiarizationLMRefiner()
    formatted = refiner.format_model_input("hello --> ")
    self.assertIn("<|turn>user\n", formatted)
    # Identity pass-through when no model is loaded
    self.assertEqual(
        refiner.generate_chunk_completion("<speaker:1> hi --> "),
        "<speaker:1> hi [eod]",
    )

    def mock_completion(prompt: str) -> str:
      self.assertIn("<speaker:1>", prompt)
      return "<speaker:1> hello world <speaker:2> good morning [eod]"

    refiner_mock = llm.DiarizationLMRefiner(completion_fn=mock_completion)
    pre_asr = asr.PrecomputedASR([
        types.WordSegment("hello", 0.0, 0.4),
        types.WordSegment("world", 0.4, 0.8),
        types.WordSegment("good", 1.1, 1.5),
        types.WordSegment("morning", 1.5, 1.9),
    ])

    def mock_diarizer(
        audio_in,
        num_speakers=None,
        min_speakers=None,
        max_speakers=None,
    ):
      del audio_in, num_speakers, min_speakers, max_speakers
      return [
          types.SpeakerTurn(0.0, 1.0, "1"),
          types.SpeakerTurn(1.0, 2.0, "2"),
      ]

    pipe = pipeline.DiarizersPipeline(
        asr_backend=pre_asr,
        diarizer_backend=mock_diarizer,
        llm_refiner=refiner_mock,
    )
    res = pipe(np.zeros(1600, dtype=np.float32), utterance_id="demo_01")
    self.assertEqual(res.utterance_id, "demo_01")
    self.assertEqual(res.word_list(), ["hello", "world", "good", "morning"])
    self.assertEqual(res.speaker_list(), ["1", "1", "2", "2"])

    refined_dict = refiner_mock.refine_json_dict({
        "utterances": [{
            "utterance_id": "u1",
            "hyp_text": "hello world good morning",
            "hyp_spk": "1 2 2 2",
        }]
    })
    self.assertEqual(
        refined_dict["utterances"][0]["hyp_spk_llm"], "1 1 2 2"
    )

  def test_evaluation_and_bootstrap_ci(self) -> None:
    synth = dataprep.generate_synthetic_dataset(num_utterances=6, seed=7)
    base_eval = evaluation.evaluate_json_dict(
        synth, hyp_spk_field="hyp_spk"
    )
    ora_eval = evaluation.evaluate_json_dict(
        synth, hyp_spk_field="hyp_spk_oracle"
    )
    self.assertLessEqual(ora_eval["WDER"], base_eval["WDER"])
    ci = evaluation.bootstrap_paired_ci(
        base_eval["utterances"],
        ora_eval["utterances"],
        num_resamples=200,
        seed=7,
    )
    self.assertIn("WDER", ci)
    self.assertIn("cpWER", ci)

  def test_cli_prepare_evaluate_and_infer(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      synth_path = os.path.join(tmpdir, "synth.json")
      sft_path = os.path.join(tmpdir, "sft.json")
      eval_path = os.path.join(tmpdir, "eval.json")
      infer_path = os.path.join(tmpdir, "infer.json")
      cli.main([
          "prepare-data",
          "--generate_synthetic",
          "--num_synthetic",
          "4",
          "--output_json",
          synth_path,
      ])
      self.assertTrue(os.path.isfile(synth_path))
      cli.main([
          "prepare-data",
          "--input_json",
          synth_path,
          "--output_json",
          sft_path,
      ])
      with open(sft_path, "r", encoding="utf-8") as f:
        sft_rows = json.load(f)
      self.assertGreater(len(sft_rows), 0)
      cli.main([
          "evaluate",
          "--input_json",
          synth_path,
          "--output_json",
          eval_path,
      ])
      self.assertTrue(os.path.isfile(eval_path))
      cli.main([
          "infer",
          "--input_json",
          synth_path,
          "--output_json",
          infer_path,
      ])
      self.assertTrue(os.path.isfile(infer_path))

  def test_readme_quickstart_and_cli_aliases(self) -> None:
    import diarizers

    # 1. Verify eigen_threshold=0.45 default in presets and SpectralClusterer
    self.assertAlmostEqual(
        configs.FISHER_TELEPHONY_PRESET.eigen_threshold, 0.45
    )
    self.assertAlmostEqual(
        configs.MEETING_MULTISPK_PRESET.eigen_threshold, 0.45
    )
    sc_default = spectral.SpectralClusterer(min_clusters=2, max_clusters=6)
    self.assertAlmostEqual(sc_default.eigen_threshold, 0.45)
    # Eigenvalues [0.001, 0.12, 0.28, 0.39, 0.68, 0.85] -> 4 clusters
    eigs = np.array([0.001, 0.12, 0.28, 0.39, 0.68, 0.85])
    self.assertEqual(sc_default.estimate_num_clusters(eigs), 4)

    # 2. Verify DiarizationPipeline / Pipeline kwargs and DiarizedTranscript
    pre_asr = asr.PrecomputedASR([
        types.WordSegment("hello", 0.0, 0.4),
        types.WordSegment("world", 0.4, 0.8),
        types.WordSegment("good", 1.1, 1.5),
        types.WordSegment("morning", 1.5, 1.9),
    ])

    def mock_diar(
        audio_in,
        num_speakers=None,
        min_speakers=None,
        max_speakers=None,
    ):
      del audio_in, num_speakers, min_speakers, max_speakers
      return [
          types.SpeakerTurn(0.0, 1.0, "1"),
          types.SpeakerTurn(1.0, 2.0, "2"),
      ]

    pipe = diarizers.DiarizationPipeline(
        asr=pre_asr,
        diarizer=mock_diar,
        llm_refiner=None,
        use_locality_gate=True,
    )
    self.assertTrue(pipe.prompt_config.use_locality_gate)
    res = pipe(np.zeros(1600, dtype=np.float32), num_speakers=2)
    self.assertIn(
        "<speaker:1> hello world <speaker:2> good morning",
        res.to_diarized_text(),
    )
    self.assertIn(
        "SPEAKER conversation 1",
        res.to_rttm(recording_id="conversation"),
    )

    # 3. Verify DiarizationLMRefiner(model_path=...), refine_transcript, LG-TPST
    refiner = diarizers.DiarizationLMRefiner(
        model_path="diarizers-community/DiarizationLM-Gemma-4-E4B-v1",
        completion_fn=lambda p: (
            "<speaker:1> hello how are <speaker:2> you i am doing great "
            "thanks [eod]"
        ),
    )
    self.assertEqual(
        refiner.model_path_or_repo,
        "diarizers-community/DiarizationLM-Gemma-4-E4B-v1",
    )
    hyp_diar = (
        "<speaker:1> hello how are you <speaker:2> i am doing great thanks"
    )
    llm_text, llm_spk = refiner.refine_transcript(hyp_diar)
    hyp_text, hyp_spk = diarizers.extract_text_and_spk(hyp_diar)
    transferred = diarizers.locality_gated_speaker_transfer(
        hyp_text=hyp_text,
        hyp_spk=hyp_spk,
        llm_text=llm_text,
        llm_spk=llm_spk,
    )
    self.assertEqual(len(transferred.split()), len(hyp_text.split()))
    self.assertIs(diarizers.ChunkedPromptBuilder, alignment.UtteranceChunker)
    self.assertIs(
        diarizers.tpst_align, alignment.transcript_preserving_speaker_transfer
    )

    # 4. Verify CLI hyphenated flags and subcommand aliases
    with tempfile.TemporaryDirectory() as tmpdir:
      synth_p = os.path.join(tmpdir, "synth.json")
      sft_p = os.path.join(tmpdir, "sft.json")
      run_p = os.path.join(tmpdir, "run.json")
      cli.main([
          "prepare-sft-data",
          "--generate-synthetic",
          "--num-synthetic",
          "4",
          "--output-json",
          synth_p,
      ])
      cli.main([
          "prepare-sft-data",
          "--input-json",
          synth_p,
          "--output-json",
          sft_p,
      ])
      cli.main([
          "run",
          "--input-json",
          synth_p,
          "--output-json",
          run_p,
      ])
      cli.main([
          "evaluate",
          "--input-json",
          run_p,
          "--bootstrap-samples",
          "50",
      ])
      self.assertTrue(os.path.isfile(sft_p))
      self.assertTrue(os.path.isfile(run_p))

  def test_chunker_overlap_words_and_permuted_reconciliation(self) -> None:
    words = [f"word{i}" for i in range(20)]
    spks = ["1"] * 10 + ["2"] * 10
    builder = alignment.ChunkedPromptBuilder(
        max_input_length=70, max_target_length=70, overlap_words=4
    )
    chunks = list(builder.chunk_utterance_fields(words, spks, spks, "u_ov"))
    self.assertGreater(len(chunks), 1)
    # Verify consecutive chunks overlap by up to 4 words
    self.assertLess(chunks[1]["start_idx"], chunks[0]["end_idx"])

    # Simulate speaker permutation in later chunks (1 <-> 2 swapped)
    comps = []
    for idx, ch in enumerate(chunks):
      cw = ch["words"].split()
      cs = ch["input_spk"].split()
      if idx % 2 == 1:
        cs = ["2" if s == "1" else "1" for s in cs]
      comps.append(alignment.create_diarized_text(cw, cs) + " [eod]")

    raw_s, gated_s = alignment.reconcile_chunk_speakers_hungarian(
        words, spks, comps, chunker=builder
    )
    self.assertEqual(len(raw_s.split()), len(words))
    self.assertEqual(gated_s.split(), spks)

  def test_dp_and_myers_levenshtein_parity(self) -> None:
    w1 = ["hello", "world", "how", "are", "you", "doing", "today"]
    w2 = ["hello", "there", "world", "how", "you", "doing", "now"]
    dp_dist, dp_align = alignment._dp_levenshtein(w1, w2)
    myers_dist, myers_align = alignment.levenshtein_with_edits(
        " ".join(w1), " ".join(w2)
    )
    fast_dist = alignment.levenshtein_distance(w1, w2)
    self.assertEqual(dp_dist, myers_dist)
    self.assertEqual(dp_dist, fast_dist)
    self.assertEqual(len(dp_align), len(myers_align))

  def test_audiocpp_asr_with_mock_cli(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      cli_script = os.path.join(tmpdir, "mock_audiocpp.sh")
      with open(cli_script, "w", encoding="utf-8") as f:
        f.write(
            "#!/bin/sh\n"
            "while [ $# -gt 0 ]; do\n"
            '  if [ "$1" = "--text-out" ]; then\n'
            '    echo "Hello from Qwen3 ASR!" > "$2"\n'
            "    shift 2\n"
            "  else\n"
            "    shift\n"
            "  fi\n"
            "done\n"
        )
      os.chmod(cli_script, 0o755)
      acpp = asr.AudioCppASR(
          gguf_model_path="dummy.gguf",
          audiocpp_cli_path=cli_script,
          chunk_seconds=5,
      )
      self.assertEqual(acpp(np.array([], dtype=np.float32)), [])
      segs = acpp(np.zeros(8000, dtype=np.float32))
      self.assertEqual(
          [s.word for s in segs], ["hello", "from", "qwen3", "asr"]
      )

  def test_train_config_and_prepare_sft_records(self) -> None:
    from diarizers import train

    cfg = train.TrainingConfig(lora_rank=128, max_seq_length=2048)
    self.assertEqual(cfg.lora_rank, 128)
    self.assertEqual(cfg.max_seq_length, 2048)
    with tempfile.TemporaryDirectory() as tmpdir:
      synth_p = os.path.join(tmpdir, "synth.json")
      synth = dataprep.generate_synthetic_dataset(num_utterances=2, seed=1)
      with open(synth_p, "w", encoding="utf-8") as f:
        json.dump(synth, f)
      recs = train.prepare_sft_records_from_json(synth_p)
      self.assertGreater(len(recs), 0)
      list_p = os.path.join(tmpdir, "list.json")
      with open(list_p, "w", encoding="utf-8") as f:
        json.dump(recs, f)
      self.assertEqual(
          len(train.prepare_sft_records_from_json(list_p)), len(recs)
      )

  def test_export_gguf_skip_and_mock_quantize(self) -> None:
    from diarizers import export

    with tempfile.TemporaryDirectory() as tmpdir:
      out_gguf = os.path.join(tmpdir, "model.gguf")
      res_skip = export.export_gguf_q4_k_m(
          hf_model_dir=tmpdir,
          output_gguf_path=out_gguf,
          llama_cpp_dir=os.path.join(tmpdir, "nonexistent"),
      )
      self.assertEqual(res_skip["status"], "skipped")

      llama_dir = os.path.join(tmpdir, "llama.cpp")
      bin_dir = os.path.join(llama_dir, "build", "bin")
      os.makedirs(bin_dir, exist_ok=True)
      conv_py = os.path.join(llama_dir, "convert_hf_to_gguf.py")
      with open(conv_py, "w", encoding="utf-8") as f:
        f.write(
            "import sys\n"
            "idx = sys.argv.index('--outfile')\n"
            "open(sys.argv[idx + 1], 'wb').write(b'GGUF_F16')\n"
        )
      quant_sh = os.path.join(bin_dir, "llama-quantize")
      with open(quant_sh, "w", encoding="utf-8") as f:
        f.write("#!/bin/sh\ncp \"$1\" \"$2\"\n")
      os.chmod(quant_sh, 0o755)

      res_ok = export.export_gguf_q4_k_m(
          hf_model_dir=tmpdir,
          output_gguf_path=out_gguf,
          llama_cpp_dir=llama_dir,
      )
      self.assertEqual(res_ok["status"], "completed")
      self.assertTrue(os.path.isfile(out_gguf))

      # Also test path without .gguf suffix to ensure output is not deleted
      out_bin = os.path.join(tmpdir, "model_q4_bin")
      res_bin = export.export_gguf_q4_k_m(
          hf_model_dir=tmpdir,
          output_gguf_path=out_bin,
          llama_cpp_dir=llama_dir,
      )
      self.assertEqual(res_bin["status"], "completed")
      self.assertTrue(os.path.isfile(out_bin))

  def test_review_fixes_regression(self) -> None:
    import diarizers

    # 1. AMI_MEETING_PRESET and PromptConfig defaults
    self.assertEqual(diarizers.AMI_MEETING_PRESET.min_clusters, 3)
    self.assertEqual(diarizers.AMI_MEETING_PRESET.max_clusters, 5)
    self.assertEqual(
        diarizers.DEFAULT_PROMPT_CONFIG.max_two_spk_span_words, 15
    )
    self.assertEqual(
        diarizers.DEFAULT_PROMPT_CONFIG.max_ghost_cluster_words, 15
    )

    # 2. _cosine_kmeans updates centroid when sample 0 is the only member
    pts = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.1, 0.99]], dtype=np.float64
    )
    labels_km = spectral._cosine_kmeans(pts, n_clusters=2, max_iter=10)
    self.assertEqual(len(labels_km), 3)
    self.assertNotEqual(labels_km[0], labels_km[1])
    self.assertEqual(labels_km[1], labels_km[2])

    # 3. extract_text_and_spk preserves attached words (<speaker:2>hello)
    att_w, att_s = diarizers.extract_text_and_spk(
        "<speaker:1> hi <speaker:2>hello world"
    )
    self.assertEqual(att_w, "hi hello world")
    self.assertEqual(att_s, "1 2 2")

    # 4. _deduplicate_repetitions preserves 3-fold repeats and collapses >=5
    self.assertEqual(
        asr._deduplicate_repetitions(["yeah", "yeah", "yeah"]),
        ["yeah", "yeah", "yeah"],
    )
    self.assertEqual(
        len(asr._deduplicate_repetitions(["um"] * 10)),
        4,
    )

    # 5. compute_utterance_metrics and compute_corpus_metrics + validation
    m_utt = diarizers.compute_utterance_metrics(
        ref_text="hello world good morning",
        ref_spk="1 1 2 2",
        hyp_text="hello world good morning",
        hyp_spk="1 1 2 2",
    )
    self.assertAlmostEqual(m_utt["WER"], 0.0)
    self.assertAlmostEqual(m_utt["WDER"], 0.0)
    self.assertAlmostEqual(m_utt["cpWER"], 0.0)

    synth = dataprep.generate_synthetic_dataset(num_utterances=4, seed=9)
    m_corp = diarizers.compute_corpus_metrics(
        synth["utterances"], hyp_spk_field="hyp_spk_oracle"
    )
    self.assertIn("WER", m_corp)
    self.assertIn("WDER", m_corp)
    self.assertIn("cpWER", m_corp)

    with self.assertRaises(ValueError):
      evaluation.evaluate_utterance("a b", "1", "a b", "1 1")
    with self.assertRaises(ValueError):
      evaluation.bootstrap_paired_ci(
          [{"utterance_id": "u1"}],
          [{"utterance_id": "u2"}],
      )

    # 6. flavor filter in build_sft_dataset ("hyp2ora", "deg2ref", "both")
    rows_both = dataprep.build_sft_dataset(synth, flavor="both")
    rows_h2o = dataprep.build_sft_dataset(synth, flavor="hyp2ora")
    rows_d2r = dataprep.build_sft_dataset(synth, flavor="deg2ref")
    self.assertGreater(len(rows_h2o), 0)
    self.assertGreater(len(rows_d2r), 0)
    self.assertEqual(len(rows_both), len(rows_h2o) + len(rows_d2r))
    self.assertTrue(all(r["flavor"] == "hyp2ora" for r in rows_h2o))
    self.assertTrue(all(r["flavor"] == "deg2ref" for r in rows_d2r))


if __name__ == "__main__":
  unittest.main()
