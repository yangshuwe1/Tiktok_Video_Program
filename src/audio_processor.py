import os
import ffmpeg
import whisper
import torch
import torchaudio
import numpy as np
from pathlib import Path
import subprocess
import librosa
import soundfile as sf
from typing import Dict, List, Optional, Tuple, Any
from contextlib import nullcontext
import re
import json
from dataclasses import dataclass
from scipy import signal
from scipy.spatial.distance import cosine
import hashlib
from logger import get_logger
from shared_models import whisper_model, silero_vad_model, demucs_model, shazam_client
import traceback
logger = get_logger("audio_processor")

@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 16000
    chunk_duration: float = 30.0
    overlap_duration: float = 2.0
    vad_threshold: float = 0.2
    noise_reduce_strength: float = 0.1
    whisper_model_size: str = "large-v3"
    language: str = None  # None for auto-detection
    task: str = "transcribe"

class AudioSeparator:
    def __init__(self, config: AudioConfig):
        self.logger = get_logger("AudioSeparator")
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.debug(f"AudioSeparator initialized with config: {self.config}")

    def separate_audio_sources(self, audio_path: str, output_dir: str) -> Dict[str, str]:
        try:
            model = demucs_model
            if model is None:
                self.logger.warning("Demucs model is not loaded. Returning original audio.")
                return {"vocals": audio_path, "non_vocals": audio_path}
            from demucs import apply
            wav, sr = torchaudio.load(audio_path)
            wav = wav.to(self.device)
            sr = int(sr)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)  
            if wav.size(0) == 1: 
                wav = wav.repeat(2, 1)
            if wav.dim() == 2:
                wav = wav.unsqueeze(0)  
            with torch.no_grad():
                sources = apply.apply_model(model, wav, device=self.device)[0]  # (sources, channels, samples)
            sources_names = getattr(model, 'sources', ["drums", "bass", "other", "vocals"])
            vocals_idx = None
            for i, name in enumerate(sources_names):
                if name.lower() == "vocals":
                    vocals_idx = i
                    break
            if vocals_idx is None:
                self.logger.error("Demucs model does not provide vocals stem.")
                raise ValueError("Demucs model does not provide vocals stem.")
            non_vocals_indices = [i for i in range(len(sources_names)) if i != vocals_idx]
            vocals_audio = sources[vocals_idx]
            non_vocals_audio = sum([sources[i] for i in non_vocals_indices])
            vocals_path = os.path.join(output_dir, "vocals.wav")
            non_vocals_path = os.path.join(output_dir, "non_vocals.wav")
            torchaudio.save(vocals_path, vocals_audio.cpu(), sr)
            torchaudio.save(non_vocals_path, non_vocals_audio.cpu(), sr)
            self.logger.info(f"Two-stems separated and saved to: {vocals_path}, {non_vocals_path}")
            return {"vocals": vocals_path, "non_vocals": non_vocals_path}
        except Exception as e:
            self.logger.error(f"Audio separation error: {e}")
            return {"vocals": audio_path, "non_vocals": audio_path}

class SpeechAnalyzer:
    def __init__(self, config: AudioConfig):
        self.logger = get_logger("SpeechAnalyzer")
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.debug(f"SpeechAnalyzer initialized with config: {self.config}")

    def analyze_speech_content(self, audio_path: str) -> Dict[str, Any]:
        try:
            self.logger.info(f"Starting speech analysis for: {audio_path}")
            
            if whisper_model is None:
                self.logger.warning("Whisper model is not loaded.")
                return {"text": "", "processed_text": "", "confidence": 0.0, "filtered": True}
            
            self.logger.info("Whisper model is loaded, proceeding with transcription...")
            transcription_result = self._transcribe_speech(audio_path)
            confidence = transcription_result.get("confidence", 0.0)
            if confidence is None:
                confidence = 0.0
            self.logger.info(f"Transcription result - confidence: {confidence:.3f}")
            self.logger.info(f"Raw text: {transcription_result.get('text', '')[:100]}...")
            
            text = transcription_result.get("text", "")
            processed_text = self._post_process_text(text)
            
            self.logger.info(f"Speech analysis completed (confidence: {confidence:.2f}, length: {len(processed_text)})")
            self.logger.info(f"Processed text: {processed_text[:100]}...")
            
            return {
                "text": text,
                "processed_text": processed_text,
                "confidence": confidence,
                "filtered": False  # Never filter out text
            }
        except Exception as e:
            self.logger.error(f"Speech analysis error: {e}")
            return {"text": "", "processed_text": "", "confidence": 0.0, "filtered": False}
    
    def _transcribe_speech(self, audio_path: str) -> Dict:
        try:
            self.logger.info(f"Starting transcription for: {audio_path}")
            
            if whisper_model is None:
                self.logger.warning("Whisper model is not loaded.")
                return {"text": "", "segments": [], "language": "en"}
            
            self.logger.info("Starting Whisper transcription directly...")
            
            options = {
                "task": self.config.task,
                "fp16": False,
                "condition_on_previous_text": True,
                "temperature": 0.0,
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6,
            }
            if self.config.language and self.config.language.lower() not in ["auto", "null", ""]:
                options["language"] = self.config.language
            
            self.logger.info(f"Whisper options: {options}")

            # Some PyTorch versions' flash/mem-efficient attention kernels can throw
            # "Expected key.size(1) == value.size(1)" in decoder attention. Use math kernel.
            sdp_ctx = getattr(torch.backends.cuda, "sdp_kernel", None)
            use_safe_attn_ctx = sdp_ctx is not None and torch.cuda.is_available()
            attn_ctx = sdp_ctx(enable_flash=False, enable_mem_efficient=False, enable_math=True) if use_safe_attn_ctx else nullcontext()

            try:
                with attn_ctx:
                    result = whisper_model.transcribe(audio_path, **options)
            except Exception as attn_err:
                self.logger.warning(f"Whisper transcription failed with current device/attention kernels: {attn_err}. Retrying with safer settings...")
                # Retry with condition_on_previous_text disabled to reduce cross-token attention usage
                safe_options = {**options, "condition_on_previous_text": False}
                try:
                    with attn_ctx:
                        result = whisper_model.transcribe(audio_path, **safe_options)
                except Exception as second_err:
                    self.logger.warning(f"Second attempt failed on current model: {second_err}. Falling back to CPU small model...")
                    try:
                        # CPU fallback: load a small model to finish transcription robustly
                        fallback_model = whisper.load_model("small", device="cpu")
                        result = fallback_model.transcribe(audio_path, **safe_options)
                    except Exception as cpu_err:
                        self.logger.error(f"CPU fallback transcription failed: {cpu_err}")
                        raise cpu_err
            
            self.logger.info(f"Whisper transcription completed - language: {result.get('language', 'unknown')}")
            self.logger.info(f"Raw transcription: {result.get('text', '')[:100]}...")
            
            # Calculate confidence but don't filter
            confidence = self._calculate_transcription_confidence(result.get("segments", []))
            
            return {
                **result,
                "confidence": confidence,
                "filtered": False
            }
            
        except Exception as e:
            self.logger.error(f"Speech transcription error: {e}")
            return {"text": "", "segments": [], "language": "en", "confidence": 0.0}
    
    def _calculate_transcription_confidence(self, segments: List[Dict]) -> float:
        """Calculate overall confidence score from segments"""
        try:
            if not segments:
                return 0.0
            
            # Calculate average confidence from segments
            confidences = []
            for segment in segments:
                # Whisper provides avg_logprob, convert to confidence
                avg_logprob = segment.get("avg_logprob", -1.0)
                # Convert log probability to confidence (0-1)
                confidence = max(0.0, min(1.0, (avg_logprob + 1.0) / 2.0))
                confidences.append(confidence)
            
            result = np.mean(confidences) if confidences else 0.0
            # Ensure result is not None
            if result is None:
                return 0.0
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.0
    
    def _post_process_text(self, text: str) -> str:
        try:
            # Simple garbage text filter - only keep English letters, numbers, and common symbols
            allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:()[]{}@#$%^&*+-=_<>/"\'\\|~`')
            filtered_text = ''.join(char for char in text if char in allowed_chars)
            
            # Remove excessive spaces
            filtered_text = ' '.join(filtered_text.split())
            
            # Remove repetitive phrases (common in Whisper output)
            sentences = filtered_text.split('.')
            unique_sentences = []
            seen_sentences = set()
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence) > 10:  # Only consider meaningful sentences
                    # Normalize sentence for comparison
                    normalized = re.sub(r'\s+', ' ', sentence.lower().strip())
                    if normalized not in seen_sentences:
                        seen_sentences.add(normalized)
                        unique_sentences.append(sentence)
            
            filtered_text = '. '.join(unique_sentences)
            
            # Basic text cleaning
            filler_patterns = [
                r'\b(um|uh|er|ah|hmm|mm|mhm|uh-huh|uh-uh)\b',
                r'\b(like|you know|i mean|basically|actually|literally)\b',
                r'\b(so|well|right|okay|ok)\b',
                r'\s+',
            ]
            
            for pattern in filler_patterns:
                filtered_text = re.sub(pattern, ' ', filtered_text, flags=re.IGNORECASE)
            
            error_corrections = {
                r'\b(its|it\'s)\b': "it's",
                r'\b(they\'re|their|there)\b': "they're",
                r'\b(you\'re|your)\b': "you're",
                r'\b(we\'re|were)\b': "we're",
                r'\b(can\'t|cant)\b': "can't",
                r'\b(won\'t|wont)\b': "won't",
                r'\b(don\'t|dont)\b': "don't",
            }
            
            for pattern, replacement in error_corrections.items():
                filtered_text = re.sub(pattern, replacement, filtered_text, flags=re.IGNORECASE)
            
            filtered_text = ' '.join(filtered_text.split())
            if filtered_text and not filtered_text.endswith(('.', '!', '?')):
                filtered_text += '.'
            
            return filtered_text
            
        except Exception as e:
            self.logger.error(f"Text post-processing error: {e}")
            return text

class MusicAnalyzer:
    def __init__(self, config: AudioConfig):
        self.logger = get_logger("MusicAnalyzer")
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.debug(f"MusicAnalyzer initialized with config: {self.config}")

    def analyze_music_copyright(self, audio_path: str) -> Dict[str, Any]:
        try:
            if shazam_client is not None:
                import asyncio
                async def recognize_audio():
                    return await shazam_client.recognize(audio_path)
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import nest_asyncio
                        nest_asyncio.apply()
                        task = asyncio.ensure_future(recognize_audio())
                        shazam_result = loop.run_until_complete(task)
                    else:
                        shazam_result = loop.run_until_complete(recognize_audio())
                except Exception as e:
                    self.logger.warning(f"Asyncio fallback: {e}")
                    shazam_result = {}
            else:
                self.logger.warning("Shazam client is not loaded.")
                shazam_result = {}
            self.logger.info(f"Music analysis finished for {audio_path}")
            return {"audio_fingerprint": "", "shazam_result": shazam_result}
        except Exception as e:
            self.logger.error(f"Music analysis error: {e}")
            return {"audio_fingerprint": "", "shazam_result": {}}

class EventDetector:
    def __init__(self, config: AudioConfig):
        self.logger = get_logger("EventDetector")
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.debug(f"EventDetector initialized with config: {self.config}")

    def detect_audio_events(self, audio_path: str) -> List[Dict]:
        try:
            y, sr = librosa.load(audio_path, sr=self.config.sample_rate)
            sr = int(sr)
            events = []
            sound_events = self._detect_sound_events(y, sr)
            events.extend(sound_events)
            music_events = self._detect_music_events(y, sr)
            events.extend(music_events)
            speech_events = self._detect_speech_events(y, sr)
            events.extend(speech_events)
            ambient_events = self._detect_ambient_events(y, sr)
            events.extend(ambient_events)
            events.sort(key=lambda x: x["start_time"])
            self.logger.info(f"Detected {len(events)} audio events in {audio_path}")
            return events
        except Exception as e:
            self.logger.error(f"Event detection error: {e}")
            return []
    
    def _detect_sound_events(self, y: np.ndarray, sr: int) -> List[Dict]:
        sr = int(sr)
        try:
            events = []
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            for i, time in enumerate(onset_times):
                start_frame = onset_frames[i]
                end_frame = start_frame + int(0.5 * sr / 512)
                if end_frame < len(y) // 512:
                    segment = y[start_frame * 512:end_frame * 512]
                    energy = float(np.mean(librosa.feature.rms(y=segment)))
                    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr)))
                    event_type = self._classify_sound_event(float(energy), float(spectral_centroid))
                    confidence = float(min(0.9, float(energy) * 10))
                    if confidence > 0.2:
                        events.append({
                            "type": "sound_effect",
                            "subtype": event_type,
                            "start_time": float(time),
                            "end_time": float(time + 0.5),
                            "confidence": confidence,
                            "features": {
                                "energy": float(energy),
                                "spectral_centroid": float(spectral_centroid)
                            }
                        })
            return events
        except Exception as e:
            self.logger.error(f"Sound event detection error: {e}")
            return []
    
    def _detect_music_events(self, y: np.ndarray, sr: int) -> List[Dict]:
        sr = int(sr)
        try:
            events = []
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beats, sr=sr)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            rec_mat = librosa.segment.recurrence_matrix(chroma)
            from scipy.signal import find_peaks
            diag = np.mean(rec_mat, axis=0)
            peaks, _ = find_peaks(diag, distance=sr//512)
            segment_times = librosa.frames_to_time(peaks, sr=sr)
            for i, time in enumerate(beat_times[::4]):
                events.append({
                    "type": "music",
                    "subtype": "beat",
                    "start_time": float(time),
                    "end_time": float(time + 0.1),
                    "confidence": 0.8,
                    "features": {"tempo": float(tempo)}
                })
            for seg_time in segment_times:
                events.append({
                    "type": "music",
                    "subtype": "segment",
                    "start_time": float(seg_time),
                    "end_time": float(seg_time + 0.5),
                    "confidence": 0.7,
                    "features": {"method": "recurrence_peaks"}
                })
            return events
        except Exception as e:
            self.logger.error(f"Music event detection error: {e}")
            return []
    
    def _detect_speech_events(self, y: np.ndarray, sr: int) -> List[Dict]:
        sr = int(sr)
        try:
            events = []
            rms = librosa.feature.rms(y=y)
            rms_times = librosa.frames_to_time(range(len(rms[0])), sr=sr)
            speech_threshold = float(np.mean(rms)) * 1.5
            speech_frames = rms[0] > speech_threshold
            speech_segments = []
            start_frame = None
            for i, is_speech in enumerate(speech_frames):
                if is_speech and start_frame is None:
                    start_frame = i
                elif not is_speech and start_frame is not None:
                    speech_segments.append((start_frame, i))
                    start_frame = None
            if start_frame is not None:
                speech_segments.append((start_frame, len(speech_frames)))
            for start_frame, end_frame in speech_segments:
                start_time = rms_times[start_frame]
                end_time = rms_times[min(end_frame, len(rms_times) - 1)]
                if end_time - start_time > 0.5:
                    events.append({
                        "type": "speech",
                        "subtype": "voice",
                        "start_time": float(start_time),
                        "end_time": float(end_time),
                        "confidence": 0.7,
                        "features": {"duration": float(end_time - start_time)}
                    })
            return events
        except Exception as e:
            self.logger.error(f"Speech event detection error: {e}")
            return []
    
    def _detect_ambient_events(self, y: np.ndarray, sr: int) -> List[Dict]:
        sr = int(sr)
        try:
            events = []
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_diff = np.diff(mfcc, axis=1)
            change_threshold = float(np.std(mfcc_diff)) * 2
            change_frames = np.where(np.mean(np.abs(mfcc_diff), axis=0) > change_threshold)[0]
            for frame in change_frames[::int(sr/2)]:
                time = librosa.frames_to_time(frame, sr=sr)
                events.append({
                    "type": "ambient",
                    "subtype": "environment_change",
                    "start_time": float(time),
                    "end_time": float(time + 0.2),
                    "confidence": 0.6,
                    "features": {"change_magnitude": float(np.mean(np.abs(mfcc_diff[:, frame])))}
                })
            return events
        except Exception as e:
            self.logger.error(f"Ambient event detection error: {e}")
            return []
    
    def _classify_sound_event(self, energy: float, spectral_centroid: float) -> str:
        energy = float(energy)
        spectral_centroid = float(spectral_centroid)
        if energy > 0.1:
            if spectral_centroid > 3000:
                return "high_frequency_impact"
            elif spectral_centroid > 1500:
                return "medium_frequency_impact"
            else:
                return "low_frequency_impact"
        else:
            return "subtle_sound"

class AudioAggregator:
    
    def __init__(self, config: AudioConfig):
        self.config = config
    
    def aggregate_results(self, 
                         preprocessed_audio: str,
                         separated_sources: Dict[str, str],
                         speech_analysis: Dict[str, Any],
                         music_analysis: Dict[str, Any],
                         audio_events: List[Dict]) -> Dict[str, Any]:
        try:
            aligned_results = self._align_timeline(
                speech_analysis, music_analysis, audio_events
            )
            
            structured_data = self._create_structured_data(
                preprocessed_audio,
                separated_sources,
                aligned_results
            )
            
            summary = self._generate_summary(structured_data)
            
            return {
                "structured_data": structured_data,
                "summary": summary,
                "metadata": {
                    "processing_time": 0,  
                    "audio_duration": self._get_audio_duration(preprocessed_audio),
                    "analysis_modules": ["preprocessor", "separator", "speech", "music", "events", "aggregator"],
                }
            }
            
        except Exception as e:
            self.logger.error(f"Result aggregation error: {e}")
            return {"structured_data": {}, "summary": {}, "metadata": {}}
    
    def _align_timeline(self, speech_analysis: Dict, music_analysis: Dict, audio_events: List[Dict]) -> Dict:
        try:
            aligned = {
                "timeline": [],
                "speech_segments": speech_analysis.get("speech_segments", []),
                "audio_events": audio_events,
                "music_features": music_analysis.get("music_features", {})
            }
            
            all_events = []
            
            for segment in speech_analysis.get("speech_segments", []):
                all_events.append({
                    "time": segment["start"],
                    "type": "speech_start",
                    "data": segment
                })
                all_events.append({
                    "time": segment["end"],
                    "type": "speech_end",
                    "data": segment
                })
            
            for event in audio_events:
                all_events.append({
                    "time": event["start_time"],
                    "type": "audio_event",
                    "data": event
                })
            
            all_events.sort(key=lambda x: x["time"])
            
            aligned["timeline"] = all_events
            return aligned
            
        except Exception as e:
            self.logger.error(f"Timeline alignment error: {e}")
            return {"timeline": [], "speech_segments": [], "audio_events": [], "music_features": {}}
    
    def _create_structured_data(self, 
                               preprocessed_audio: str,
                               separated_sources: Dict[str, str],
                               aligned_results: Dict) -> Dict:
        try:
            return {
                "audio_info": {
                    "preprocessed_path": preprocessed_audio,
                    "separated_sources": separated_sources,
                    "duration": self._get_audio_duration(preprocessed_audio)
                },
                "speech_analysis": {
                    "transcription": aligned_results.get("speech_analysis", {}).get("processed_text", ""),
                    "language": aligned_results.get("speech_analysis", {}).get("language", "en"),
                    "confidence": aligned_results.get("speech_analysis", {}).get("confidence", 0),
                    "emotion": aligned_results.get("speech_analysis", {}).get("emotion", {}),
                    "segments": aligned_results.get("speech_segments", [])
                },
                "music_analysis": {
                    "copyright_status": aligned_results.get("music_analysis", {}).get("copyright_status", {}),
                    "music_features": aligned_results.get("music_features", {}),
                    "detection_confidence": aligned_results.get("music_analysis", {}).get("detection_confidence", 0)
                },
                "audio_events": aligned_results.get("audio_events", []),
                "timeline": aligned_results.get("timeline", [])
            }
            
        except Exception as e:
            self.logger.error(f"Structured data creation error: {e}")
            return {}
    
    def _generate_summary(self, structured_data: Dict) -> Dict:
        try:
            speech_analysis = structured_data.get("speech_analysis", {})
            music_analysis = structured_data.get("music_analysis", {})
            audio_events = structured_data.get("audio_events", [])
            
            speech_duration = sum(
                seg["end"] - seg["start"] 
                for seg in speech_analysis.get("segments", []) 
                if seg.get("speech", False)
            )
            
            event_counts = {}
            for event in audio_events:
                event_type = event.get("type", "unknown")
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            return {
                "total_duration": structured_data.get("audio_info", {}).get("duration", 0),
                "speech_duration": speech_duration,
                "speech_percentage": (speech_duration / structured_data.get("audio_info", {}).get("duration", 1)) * 100,
                "event_summary": event_counts,
                "copyright_warning": music_analysis.get("copyright_status", {}).get("warning", ""),
                "emotion_summary": speech_analysis.get("emotion", {}).get("emotion", "neutral"),
                "language": speech_analysis.get("language", "en")
            }
            
        except Exception as e:
            self.logger.error(f"Summary generation error: {e}")
            return {}

    def _get_audio_duration(self, audio_path: str) -> float:
        try:
            y, sr = librosa.load(audio_path, sr=None)
            return len(y) / sr
        except Exception as e:
            self.logger.error(f"Audio duration calculation error: {e}")
            return 0.0

class AdvancedAudioProcessor:
    
    def __init__(self, config: Optional[AudioConfig] = None):
        self.logger = get_logger("AdvancedAudioProcessor")
        # Handle both AudioConfig objects and dictionary configs
        if isinstance(config, dict):
            # Convert dictionary to AudioConfig
            self.config = AudioConfig(
                sample_rate=config.get('sample_rate', 16000),
                chunk_duration=config.get('chunk_duration', 30.0),
                overlap_duration=config.get('overlap_duration', 2.0),
                vad_threshold=config.get('vad_threshold', 0.2),
                noise_reduce_strength=config.get('noise_reduce_strength', 0.1),
                whisper_model_size=config.get('whisper_model_size', 'large-v3'),
                language=config.get('language', 'auto'),
                task=config.get('task', 'transcribe')
            )
        else:
            self.config = config or AudioConfig()
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.separator = AudioSeparator(self.config)
        self.speech_analyzer = SpeechAnalyzer(self.config)
        self.speech_rate_analyzer = SpeechRateAnalyzer(self.config)
        self.music_analyzer = MusicAnalyzer(self.config)
        self.event_detector = EventDetector(self.config)
        self.aggregator = AudioAggregator(self.config)
        
        self.logger.info(f"AdvancedAudioProcessor initialized on device: {self.device}")
    
    def process_audio_for_video(self, video_path: str, output_dir: str) -> Dict[str, Any]:
        import time
        start_time = time.time()
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        timings = {}
        try:
            t0 = time.time()
            audio_path = os.path.join(output_dir, f"{video_name}.wav")
            self.extract_audio(video_path, audio_path)
            timings['extract_audio'] = time.time() - t0

            t0 = time.time()
            separated_sources = self.separator.separate_audio_sources(audio_path, output_dir)
            timings['separate_audio_sources'] = time.time() - t0

            vocals_path = separated_sources.get('vocals', None)
            non_vocals_path = separated_sources.get('non_vocals', None)
            if vocals_path and os.path.exists(vocals_path):
                speech_audio = vocals_path
            else:
                speech_audio = audio_path

            t0 = time.time()
            speech_analysis = self.speech_analyzer.analyze_speech_content(speech_audio)
            timings['analyze_speech_content'] = time.time() - t0

            # Use the transcription result from speech analysis for speech rate analysis
            raw_transcription = {
                "text": speech_analysis.get("text", ""),
                "segments": speech_analysis.get("segments", []),
                "language": speech_analysis.get("language", "en"),
                "confidence": speech_analysis.get("confidence", 0.0)
            }
            
            # Perform speech rate analysis
            t0 = time.time()
            speech_rate_analysis = self.speech_rate_analyzer.analyze_speech_rate(speech_audio, raw_transcription)
            timings['analyze_speech_rate'] = time.time() - t0

            # Log speech analysis results
            speech_confidence = speech_analysis.get("confidence", 0.0)
            if speech_confidence is None:
                speech_confidence = 0.0
            speech_text = speech_analysis.get("processed_text", "")
            
            self.logger.info(f"Speech analysis completed - confidence: {speech_confidence:.2f}, text length: {len(speech_text)})")
            if speech_text:
                self.logger.info(f"Speech text: {speech_text[:100]}...")
            else:
                self.logger.info("No speech text extracted")
            
            # Log speech rate analysis results
            self.logger.debug(f"speech_rate_analysis structure: {speech_rate_analysis}")
            speech_rate_score = speech_rate_analysis.get("overall_score", 0.0)
            speech_rate_wpm = speech_rate_analysis.get("speech_rate_wpm", 0.0)
            
            self.logger.debug(f"Extracted speech_rate_score: {speech_rate_score} (type: {type(speech_rate_score)})")
            self.logger.debug(f"Extracted speech_rate_wpm: {speech_rate_wpm} (type: {type(speech_rate_wpm)})")
            
            # Handle None values before formatting
            if speech_rate_score is None:
                self.logger.warning(f"speech_rate_score is None! Setting to 0.0")
                speech_rate_score = 0.0
            if speech_rate_wpm is None:
                self.logger.warning(f"speech_rate_wpm is None! Setting to 0.0")
                speech_rate_wpm = 0.0
                
            self.logger.info(f"Speech rate analysis completed - Overall score: {speech_rate_score:.1f}/10, WPM: {speech_rate_wpm:.1f}")
            self.logger.info(f"Speech rate category: {speech_rate_analysis.get('rate_category', 'unknown')}")
            self.logger.info(f"Speech rate recommendation: {speech_rate_analysis.get('rate_analysis', {}).get('recommendation', 'No recommendation')}")

            music_input = non_vocals_path if non_vocals_path and os.path.exists(non_vocals_path) else audio_path
            t0 = time.time()
            music_analysis = self.music_analyzer.analyze_music_copyright(music_input)
            timings['analyze_music_copyright'] = time.time() - t0

            # t0 = time.time()
            # audio_events = self.event_detector.detect_audio_events(audio_path)
            # timings['detect_audio_events'] = time.time() - t0

            # t0 = time.time()
            # aggregated_results = self.aggregator.aggregate_results(
            #     audio_path, separated_sources, speech_analysis, 
            #     music_analysis, audio_events
            # )
            # timings['aggregate_results'] = time.time() - t0

            music_title = ""
            music_artist = ""
            shazam_result = music_analysis.get("shazam_result", {})
            if isinstance(shazam_result, dict):
                track = shazam_result.get("track", {})
                music_title = track.get("title", "")
                music_artist = track.get("subtitle", "")
            processing_time = time.time() - start_time
            aggregated_results = {
                "metadata": {
                    "processing_time": processing_time,
                    "step_timings": timings
                },
                "speech_text": speech_text,
                "speech_confidence": speech_confidence,
                "speech_filtered": False, # Always False as filtering is removed
                "music_title": music_title,
                "music_artist": music_artist,
                "speech_rate_analysis": speech_rate_analysis
            }
            t0 = time.time()
            self._save_results(output_dir, video_name, aggregated_results)
            timings['save_results'] = time.time() - t0

            return {
                "speech_text": speech_text,
                "speech_confidence": speech_confidence,
                "speech_filtered": False, # Always False as filtering is removed
                "music_title": music_title,
                "music_artist": music_artist,
                "speech_rate_analysis": speech_rate_analysis
            }
        except Exception as e:
            self.logger.error(f"Audio processing error: {e}")
            return {"error": str(e)}

    def extract_audio(self, video_path: str, output_path: str) -> str:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            command = [
                'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
                '-ar', str(self.config.sample_rate), '-ac', '1', '-y', output_path
            ]
            
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode == 0:
                return output_path
            else:
                self.logger.error(f"FFmpeg error: {result.stderr}")
                return output_path  
                
        except Exception as e:
            self.logger.error(f"Audio extraction error: {e}")
            return output_path if output_path is not None else ""

    def _save_results(self, output_dir: str, video_name: str, results: Dict):
        try:
            json_path = os.path.join(output_dir, 'audio_analysis_results.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            speech_text = None
            if isinstance(results, dict):
                if "speech_text" in results:
                    speech_text = results["speech_text"]
                elif "structured_data" in results and "speech_analysis" in results["structured_data"]:
                    speech_text = results["structured_data"]["speech_analysis"].get("transcription", "")
            txt_path = os.path.join(output_dir, 'speech_transcription.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(speech_text or "")
            self.logger.info(f"Results saved to: {output_dir}")
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def has_speech(self, audio_path: str) -> bool:
        speech_analysis = self.speech_analyzer.analyze_speech_content(audio_path)
        return len(speech_analysis.get("speech_segments", [])) > 0
    
    def speech_to_text(self, audio_path: str) -> str:
        speech_analysis = self.speech_analyzer.analyze_speech_content(audio_path)
        return speech_analysis.get("processed_text", "")
    
    def save_speech_text(self, video_output_dir: str, speech_text: str):
        txt_file_path = os.path.join(video_output_dir, 'speech_text.txt')
        try:
            with open(txt_file_path, 'w', encoding='utf-8') as f:
                f.write(speech_text)
            self.logger.info(f"Speech text saved to: {txt_file_path}")
        except Exception as e:
            self.logger.error(f"Error saving speech text: {e}")

    def read_speech_text(self, txt_path: str) -> str:
        if not txt_path or not os.path.exists(txt_path):
            return ""
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            self.logger.error(f"Error reading speech text: {e}")
            return "" 

class SpeechRateAnalyzer:
    def __init__(self, config: AudioConfig):
        self.logger = get_logger("SpeechRateAnalyzer")
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Speech rate analysis parameters
        self.words_per_minute_ranges = {
            "very_slow": (0, 120),      # 0-120 WPM
            "slow": (120, 150),         # 120-150 WPM
            "normal": (150, 180),       # 150-180 WPM (ideal range)
            "fast": (180, 220),        # 180-220 WPM
            "very_fast": (220, 300),   # 220-300 WPM
            "extremely_fast": (300, float('inf'))  # 300+ WPM
        }
        
        # Scoring weights for different factors
        self.scoring_weights = {
            "speech_rate": 0.4,        # 40% weight for speech rate
            "consistency": 0.25,        # 25% weight for consistency
            "clarity": 0.2,            # 20% weight for clarity (confidence)
            "pauses": 0.15             # 15% weight for pause analysis
        }
        
        self.logger.debug(f"SpeechRateAnalyzer initialized with config: {self.config}")

    def analyze_speech_rate(self, audio_path: str, transcription_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze speech rate and provide comprehensive scoring
        
        Args:
            audio_path: Path to audio file
            transcription_result: Result from Whisper transcription
            
        Returns:
            Dictionary containing speech rate analysis and scoring
        """
        try:
            self.logger.info(f"Starting speech rate analysis for: {audio_path}")
            
            # Extract basic speech metrics
            speech_metrics = self._calculate_speech_metrics(transcription_result)
            self.logger.debug(f"Speech metrics: {speech_metrics}")
            
            # Analyze speech rate patterns
            rate_analysis = self._analyze_speech_rate_patterns(speech_metrics)
            self.logger.debug(f"Rate analysis: {rate_analysis}")
            
            # Calculate overall score (1-10)
            overall_score = self._calculate_overall_score(rate_analysis, transcription_result)
            self.logger.debug(f"Overall score: {overall_score} (type: {type(overall_score)})")
            
            # Generate detailed report
            detailed_report = self._generate_detailed_report(
                speech_metrics, rate_analysis, overall_score
            )
            
            result = {
                "speech_rate_wpm": speech_metrics["words_per_minute"],
                "syllables_per_minute": speech_metrics["syllables_per_minute"],
                "rate_category": rate_analysis["category"],
                "rate_score": rate_analysis["score"],
                "overall_score": overall_score,
                "detailed_report": detailed_report,
                "metrics": speech_metrics,
                "rate_analysis": rate_analysis
            }
            
            # Add None check before logging
            if overall_score is None:
                self.logger.warning(f"Overall score is None! Rate analysis: {rate_analysis}")
                overall_score = 0.0
            
            self.logger.info(f"Speech rate analysis completed - Overall score: {overall_score}/10")
            return result
            
        except Exception as e:
            self.logger.error(f"Speech rate analysis error: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return self._create_fallback_result()

    def _calculate_speech_metrics(self, transcription_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate basic speech metrics from transcription"""
        try:
            segments = transcription_result.get("segments", [])
            text = transcription_result.get("text", "")
            
            if not segments or not text.strip():
                return self._create_empty_metrics()
            
            # Calculate total duration
            total_duration = segments[-1]["end"] if segments else 0
            
            # Count words and syllables
            words = text.split()
            word_count = len(words)
            syllable_count = sum(self._count_syllables(word) for word in words)
            
            # Calculate rates
            words_per_minute = (word_count / total_duration * 60) if total_duration > 0 else 0
            syllables_per_minute = (syllable_count / total_duration * 60) if total_duration > 0 else 0
            
            # Calculate speech segments
            speech_segments = []
            for segment in segments:
                segment_text = segment.get("text", "").strip()
                if segment_text:
                    segment_words = segment_text.split()
                    segment_duration = segment["end"] - segment["start"]
                    segment_wpm = (len(segment_words) / segment_duration * 60) if segment_duration > 0 else 0
                    
                    speech_segments.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "duration": segment_duration,
                        "word_count": len(segment_words),
                        "words_per_minute": segment_wpm,
                        "text": segment_text
                    })
            
            return {
                "total_duration": total_duration,
                "word_count": word_count,
                "syllable_count": syllable_count,
                "words_per_minute": words_per_minute,
                "syllables_per_minute": syllables_per_minute,
                "speech_segments": speech_segments,
                "average_segment_duration": np.mean([seg["duration"] for seg in speech_segments]) if speech_segments else 0,
                "segment_count": len(speech_segments)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating speech metrics: {e}")
            return self._create_empty_metrics()

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word using a simple heuristic"""
        try:
            word = word.lower()
            if not word:
                return 0
            
            # Remove common suffixes that don't add syllables
            word = re.sub(r'(es|ed|ing|ly|er|est)$', '', word)
            
            # Count vowel groups
            vowels = 'aeiouy'
            syllable_count = 0
            prev_char_was_vowel = False
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_char_was_vowel:
                    syllable_count += 1
                prev_char_was_vowel = is_vowel
            
            # Ensure at least one syllable
            return max(1, syllable_count)
            
        except Exception:
            return 1  # Fallback to 1 syllable

    def _analyze_speech_rate_patterns(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze speech rate patterns and categorize"""
        try:
            self.logger.debug(f"_analyze_speech_rate_patterns input - metrics: {metrics}")
            wpm = metrics["words_per_minute"]
            self.logger.debug(f"Extracted wpm from metrics: {wpm} (type: {type(wpm)})")
            
            # Handle None values for wpm
            if wpm is None:
                self.logger.warning(f"WPM from metrics is None! Setting to 0.0")
                wpm = 0.0
            
            # Simplified categorization based on WPM
            if wpm < 10:
                category = "no_speech"
            elif wpm < 120:
                category = "slow"
            elif 120 <= wpm < 150:
                category = "below_average"
            elif 150 <= wpm <= 180:
                category = "optimal"
            elif 180 < wpm <= 200:
                category = "above_average"
            elif 200 < wpm <= 250:
                category = "fast"
            else:
                category = "very_fast"
            
            self.logger.debug(f"Category determined: {category}")
            
            # Calculate rate score (0.0 or 1-10)
            if category == "no_speech":
                rate_score = 0.0
            elif category == "optimal":
                rate_score = 10
            elif category == "above_average":
                rate_score = 8
            elif category == "below_average":
                rate_score = 6
            elif category == "fast":
                rate_score = 4
            elif category == "slow":
                rate_score = 2
            else:  # very_fast
                rate_score = 1
            
            self.logger.debug(f"Rate score calculated: {rate_score}")
            
            result = {
                "category": category,
                "score": rate_score,
                "wpm": wpm,
                "optimal_range": "150-180 WPM",
                "recommendation": self._get_rate_recommendation(category, wpm)
            }
            
            self.logger.debug(f"Returning rate analysis: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing speech rate patterns: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {"category": "unknown", "score": 5, "wpm": 0, "optimal_range": "150-180 WPM", "recommendation": "Unable to analyze"}

    def _get_rate_recommendation(self, category: str, wpm: float) -> str:
        """Get recommendation based on speech rate category"""
        # Handle None values for wpm
        if wpm is None:
            wpm = 0.0
            
        recommendations = {
            "no_speech": f"No speech detected ({wpm:.1f} WPM). This video appears to have no spoken content.",
            "slow": f"Speech rate is slow ({wpm:.1f} WPM). Consider speaking faster for better engagement.",
            "below_average": f"Speech rate is below average ({wpm:.1f} WPM). Slightly faster speech could improve engagement.",
            "optimal": f"Speech rate is optimal ({wpm:.1f} WPM). This is an excellent pace for audience engagement.",
            "above_average": f"Speech rate is above average ({wpm:.1f} WPM). Consider slowing down slightly for better clarity.",
            "fast": f"Speech rate is fast ({wpm:.1f} WPM). Slowing down would improve comprehension.",
            "very_fast": f"Speech rate is very fast ({wpm:.1f} WPM). Significant slowing down is recommended."
        }
        return recommendations.get(category, "Unable to provide recommendation")

    def _calculate_consistency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate consistency score based on speech rate variation"""
        try:
            speech_segments = metrics.get("speech_segments", [])
            if len(speech_segments) < 2:
                return 5.0  # Neutral score for insufficient data
            
            # Calculate WPM for each segment
            segment_wpms = [seg["words_per_minute"] for seg in speech_segments]
            
            # Calculate coefficient of variation (CV = std/mean)
            mean_wpm = np.mean(segment_wpms)
            std_wpm = np.std(segment_wpms)
            
            if mean_wpm == 0:
                return 5.0
            
            cv = std_wpm / mean_wpm
            
            # Score based on consistency (lower CV = higher score)
            # CV < 0.1: excellent consistency (9-10)
            # CV < 0.2: good consistency (7-9)
            # CV < 0.3: moderate consistency (5-7)
            # CV >= 0.3: poor consistency (1-5)
            
            if cv < 0.1:
                consistency_score = 9.0 + (0.1 - cv) / 0.1
            elif cv < 0.2:
                consistency_score = 7.0 + (0.2 - cv) / 0.1 * 2
            elif cv < 0.3:
                consistency_score = 5.0 + (0.3 - cv) / 0.1 * 2
            else:
                consistency_score = max(1.0, 5.0 - (cv - 0.3) / 0.1 * 4)
            
            return max(1.0, min(10.0, consistency_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating consistency score: {e}")
            return 5.0

    def _analyze_pauses(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze pauses and breaks in speech"""
        try:
            speech_segments = metrics.get("speech_segments", [])
            total_duration = metrics.get("total_duration", 0)
            
            if len(speech_segments) < 2 or total_duration == 0:
                return {"score": 5.0, "pause_percentage": 0, "average_pause_duration": 0}
            
            # Calculate pauses between segments
            pauses = []
            for i in range(len(speech_segments) - 1):
                current_end = speech_segments[i]["end"]
                next_start = speech_segments[i + 1]["start"]
                pause_duration = next_start - current_end
                if pause_duration > 0:
                    pauses.append(pause_duration)
            
            if not pauses:
                return {"score": 5.0, "pause_percentage": 0, "average_pause_duration": 0}
            
            # Calculate pause metrics
            total_pause_time = sum(pauses)
            pause_percentage = (total_pause_time / total_duration) * 100
            average_pause_duration = np.mean(pauses)
            
            # Score based on pause characteristics
            # Optimal pause percentage: 10-20%
            # Optimal average pause: 0.5-1.5 seconds
            
            pause_score = 5.0  # Start with neutral score
            
            # Adjust score based on pause percentage
            if 10 <= pause_percentage <= 20:
                pause_score += 2.0  # Optimal range
            elif 5 <= pause_percentage <= 25:
                pause_score += 1.0  # Good range
            elif pause_percentage < 5:
                pause_score -= 1.0  # Too few pauses
            elif pause_percentage > 25:
                pause_score -= 2.0  # Too many pauses
            
            # Adjust score based on average pause duration
            if 0.5 <= average_pause_duration <= 1.5:
                pause_score += 1.5  # Optimal duration
            elif 0.3 <= average_pause_duration <= 2.0:
                pause_score += 0.5  # Good duration
            elif average_pause_duration < 0.3:
                pause_score -= 0.5  # Too short pauses
            elif average_pause_duration > 2.0:
                pause_score -= 1.0  # Too long pauses
            
            return {
                "score": max(1.0, min(10.0, pause_score)),
                "pause_percentage": pause_percentage,
                "average_pause_duration": average_pause_duration,
                "pause_count": len(pauses),
                "total_pause_time": total_pause_time
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing pauses: {e}")
            return {"score": 5.0, "pause_percentage": 0, "average_pause_duration": 0}

    def _calculate_clarity_score(self, transcription_result: Dict[str, Any]) -> float:
        """Calculate clarity score based on transcription confidence"""
        try:
            confidence = transcription_result.get("confidence", 0.0)
            
            # Convert confidence to 1-10 scale
            # Confidence is already 0-1, so multiply by 10
            clarity_score = confidence * 10
            
            return max(1.0, min(10.0, clarity_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating clarity score: {e}")
            return 5.0

    def _calculate_overall_score(self, rate_analysis: Dict, transcription_result: Dict):
        """Calculate overall score based on speech rate only"""
        try:
            self.logger.debug(f"_calculate_overall_score input - rate_analysis: {rate_analysis}")
            wpm = rate_analysis.get("wpm", 0.0)
            self.logger.debug(f"Extracted wpm: {wpm} (type: {type(wpm)})")
            
            # Handle None values for wpm
            if wpm is None:
                self.logger.warning(f"WPM is None! Setting to 0.0")
                wpm = 0.0
            
            # Check for no speech first
            if wpm < 10:
                self.logger.debug(f"WPM < 10, returning 0.0")
                return 0.0
            
            # Simplified scoring based on WPM ranges
            if 150 <= wpm <= 180:
                # Optimal range: 150-180 WPM gets highest score
                score = 10
            elif 140 <= wpm < 150 or 180 < wpm <= 190:
                # Good range: 140-150 or 180-190 WPM
                score = 8
            elif 130 <= wpm < 140 or 190 < wpm <= 200:
                # Fair range: 130-140 or 190-200 WPM
                score = 6
            elif 120 <= wpm < 130 or 200 < wpm <= 220:
                # Below average: 120-130 or 200-220 WPM
                score = 4
            elif 10 <= wpm < 120 or 220 < wpm <= 250:
                # Poor: 10-120 or 220-250 WPM
                score = 2
            else:
                # Very poor: >250 WPM
                score = 1
            
            self.logger.debug(f"Calculated score: {score}")
            return score
            
        except Exception as e:
            self.logger.error(f"Error calculating overall score: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return 5

    def _generate_detailed_report(self, metrics: Dict, rate_analysis: Dict, overall_score) -> str:
        """Generate detailed analysis report"""
        try:
            report_parts = []
            
            # Overall score
            if overall_score == 0.0:
                report_parts.append("OVERALL SPEECH RATE SCORE: 0.0/10 (No speech detected)")
            else:
                report_parts.append(f"OVERALL SPEECH RATE SCORE: {overall_score}/10")
            report_parts.append("=" * 50)
            
            # Speech rate analysis
            wpm = metrics.get("words_per_minute", 0)
            category = rate_analysis.get("category", "unknown")
            rate_score = rate_analysis.get("score", 5)
            recommendation = rate_analysis.get("recommendation", "")
            
            # Handle None values
            if wpm is None:
                wpm = 0.0
            if rate_score is None:
                rate_score = 0.0
            
            report_parts.append(f"SPEECH RATE ANALYSIS:")
            report_parts.append(f"  Words per minute: {wpm:.1f} WPM")
            report_parts.append(f"  Syllables per minute: {metrics.get('syllables_per_minute', 0):.1f} SPM")
            report_parts.append(f"  Category: {category.replace('_', ' ').title()}")
            report_parts.append(f"  Rate score: {rate_score}/10")
            report_parts.append(f"  Recommendation: {recommendation}")
            
            # Additional metrics
            total_duration = metrics.get("total_duration", 0)
            word_count = metrics.get("word_count", 0)
            syllable_count = metrics.get("syllable_count", 0)
            
            report_parts.append(f"\nADDITIONAL METRICS:")
            report_parts.append(f"  Total duration: {total_duration:.1f} seconds")
            report_parts.append(f"  Word count: {word_count}")
            report_parts.append(f"  Syllable count: {syllable_count}")
            
            return "\n".join(report_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating detailed report: {e}")
            if overall_score == 0.0:
                return "Speech rate analysis completed. Overall score: 0.0/10 (No speech detected)"
            else:
                return f"Speech rate analysis completed. Overall score: {overall_score}/10"

    def _create_empty_metrics(self) -> Dict[str, Any]:
        """Create empty metrics when no speech is detected"""
        return {
            "total_duration": 0,
            "word_count": 0,
            "syllable_count": 0,
            "words_per_minute": 0,
            "syllables_per_minute": 0,
            "speech_segments": [],
            "average_segment_duration": 0,
            "segment_count": 0
        }

    def _create_fallback_result(self) -> Dict[str, Any]:
        """Create fallback result when analysis fails"""
        return {
            "speech_rate_wpm": 0,
            "syllables_per_minute": 0,
            "rate_category": "unknown",
            "rate_score": 5,
            "overall_score": 5,
            "detailed_report": "Speech rate analysis failed. Unable to provide detailed report.",
            "metrics": self._create_empty_metrics(),
            "rate_analysis": {"category": "unknown", "score": 5, "wpm": 0}
        }

class AudioProcessor(AdvancedAudioProcessor):
    pass 