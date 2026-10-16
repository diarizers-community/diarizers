"""Core data structures for Diarizers transcripts, segments, and metrics."""

import dataclasses
import json
from typing import Any, List, Optional

from diarizers import configs


def _format_timestamp_srt(seconds: float) -> str:
  """Formats seconds as HH:MM:SS,mmm for SRT subtitles."""
  total_ms = max(0, int(round(seconds * 1000)))
  ms = total_ms % 1000
  total_s = total_ms // 1000
  s = total_s % 60
  total_m = total_s // 60
  m = total_m % 60
  h = total_m // 60
  return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
  """Formats seconds as HH:MM:SS.mmm for WebVTT subtitles."""
  return _format_timestamp_srt(seconds).replace(",", ".")


@dataclasses.dataclass
class WordSegment:
  """A transcribed word with start and end timestamps in seconds."""

  word: str
  start: float
  end: float
  speaker: Optional[str] = None

  def to_dict(self) -> dict[str, Any]:
    out: dict[str, Any] = {
        "word": self.word,
        "start": round(float(self.start), 3),
        "end": round(float(self.end), 3),
    }
    if self.speaker is not None:
      out["speaker"] = str(self.speaker)
    return out


@dataclasses.dataclass
class SpeakerTurn:
  """A contiguous speaker turn with start and end timestamps in seconds."""

  start: float
  end: float
  speaker: str
  text: str = ""

  @property
  def duration(self) -> float:
    return max(0.0, float(self.end) - float(self.start))

  def to_dict(self) -> dict[str, Any]:
    return {
        "start": round(float(self.start), 3),
        "end": round(float(self.end), 3),
        "speaker": str(self.speaker),
        "text": self.text,
    }


@dataclasses.dataclass
class DiarizedWord:
  """A word aligned with both initial acoustic and LLM-refined speakers."""

  word: str
  start: float
  end: float
  acoustic_speaker: str = "1"
  speaker: str = "1"

  def to_dict(self) -> dict[str, Any]:
    return {
        "word": self.word,
        "start": round(float(self.start), 3),
        "end": round(float(self.end), 3),
        "acoustic_speaker": str(self.acoustic_speaker),
        "speaker": str(self.speaker),
    }


@dataclasses.dataclass
class DiarizedTranscript:
  """Complete output of the 4-stage Diarizers pipeline."""

  utterance_id: str = "utt"
  words: List[DiarizedWord] = dataclasses.field(default_factory=list)
  hyp_text: str = ""
  hyp_spk: str = ""
  hyp_spk_llm_raw: str = ""
  hyp_spk_llm: str = ""
  diarized_before_llm: str = ""
  diarized_after_llm: str = ""

  def word_list(self) -> List[str]:
    """Returns list of transcribed words."""
    if self.words:
      return [w.word for w in self.words]
    return self.hyp_text.split()

  def speaker_list(self, use_llm_speaker: bool = True) -> List[str]:
    """Returns list of per-word speaker labels."""
    if self.words:
      return [
          w.speaker if use_llm_speaker else w.acoustic_speaker
          for w in self.words
      ]
    return (self.hyp_spk_llm if use_llm_speaker else self.hyp_spk).split()

  def get_turns(self, use_llm_speaker: bool = True) -> List[SpeakerTurn]:
    """Groups contiguous words from the same speaker into SpeakerTurns."""
    if not self.words:
      return []
    turns: List[SpeakerTurn] = []
    first = self.words[0]
    curr_spk = first.speaker if use_llm_speaker else first.acoustic_speaker
    curr_start = first.start
    curr_end = first.end
    curr_words = [first.word]

    for w in self.words[1:]:
      spk = w.speaker if use_llm_speaker else w.acoustic_speaker
      if spk == curr_spk:
        curr_end = max(curr_end, w.end)
        curr_words.append(w.word)
      else:
        turns.append(
            SpeakerTurn(
                start=curr_start,
                end=curr_end,
                speaker=curr_spk,
                text=" ".join(curr_words),
            )
        )
        curr_spk = spk
        curr_start = w.start
        curr_end = w.end
        curr_words = [w.word]

    turns.append(
        SpeakerTurn(
            start=curr_start,
            end=curr_end,
            speaker=curr_spk,
            text=" ".join(curr_words),
        )
    )
    return turns

  def to_text(
      self,
      use_llm_speaker: bool = True,
      speaker_prefix: str = configs.SPEAKER_PREFIX,
      speaker_suffix: str = configs.SPEAKER_SUFFIX,
      use_new_line: bool = False,
  ) -> str:
    """Formats transcript with speaker tags."""
    if not self.words:
      return (
          self.diarized_after_llm
          if use_llm_speaker
          else self.diarized_before_llm
      )
    parts: List[str] = []
    prev_spk = ""
    for w in self.words:
      spk = w.speaker if use_llm_speaker else w.acoustic_speaker
      if spk != prev_spk:
        if prev_spk and use_new_line:
          parts.append("\n")
        parts.append(f"{speaker_prefix}{spk}{speaker_suffix}")
        prev_spk = spk
      parts.append(w.word)
    return " ".join(parts)

  def to_diarized_text(
      self,
      use_llm_speaker: bool = True,
      speaker_prefix: str = configs.SPEAKER_PREFIX,
      speaker_suffix: str = configs.SPEAKER_SUFFIX,
      use_new_line: bool = False,
  ) -> str:
    """Alias for to_text() formatting transcript with speaker tags."""
    return self.to_text(
        use_llm_speaker=use_llm_speaker,
        speaker_prefix=speaker_prefix,
        speaker_suffix=speaker_suffix,
        use_new_line=use_new_line,
    )

  def to_rttm(
      self,
      use_llm_speaker: bool = True,
      recording_id: Optional[str] = None,
  ) -> str:
    """Formats speaker turns as NIST RTTM lines."""
    rec_id = recording_id or self.utterance_id
    lines: List[str] = []
    for turn in self.get_turns(use_llm_speaker=use_llm_speaker):
      dur = max(0.01, turn.duration)
      line = (
          f"SPEAKER {rec_id} 1 {turn.start:.3f} {dur:.3f} "
          f"<NA> <NA> speaker_{turn.speaker} <NA> <NA>"
      )
      lines.append(line)
    return "\n".join(lines)

  def to_srt(self, use_llm_speaker: bool = True) -> str:
    """Formats speaker turns as SubRip (.srt) subtitles."""
    blocks: List[str] = []
    for idx, turn in enumerate(
        self.get_turns(use_llm_speaker=use_llm_speaker), start=1
    ):
      t_s = _format_timestamp_srt(turn.start)
      t_e = _format_timestamp_srt(max(turn.start + 0.05, turn.end))
      blocks.append(
          f"{idx}\n{t_s} --> {t_e}\n[Speaker {turn.speaker}] {turn.text}"
      )
    return "\n\n".join(blocks)

  def to_vtt(self, use_llm_speaker: bool = True) -> str:
    """Formats speaker turns as WebVTT (.vtt) subtitles."""
    lines = ["WEBVTT", ""]
    for turn in self.get_turns(use_llm_speaker=use_llm_speaker):
      t_s = _format_timestamp_vtt(turn.start)
      t_e = _format_timestamp_vtt(max(turn.start + 0.05, turn.end))
      lines.append(f"{t_s} --> {t_e}")
      lines.append(f"<v Speaker {turn.speaker}>{turn.text}")
      lines.append("")
    return "\n".join(lines).strip()

  def to_dict(self) -> dict[str, Any]:
    """Serializes transcript to a JSON-compatible dictionary."""
    return {
        "utterance_id": self.utterance_id,
        "hyp_text": self.hyp_text,
        "hyp_spk": self.hyp_spk,
        "hyp_spk_llm_raw": self.hyp_spk_llm_raw,
        "hyp_spk_llm": self.hyp_spk_llm,
        "diarized_before_llm": self.diarized_before_llm,
        "diarized_after_llm": self.diarized_after_llm,
        "words": [w.to_dict() for w in self.words],
        "turns": [t.to_dict() for t in self.get_turns(use_llm_speaker=True)],
    }

  def to_json(self, indent: int = 2) -> str:
    """Serializes transcript to a formatted JSON string."""
    return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
