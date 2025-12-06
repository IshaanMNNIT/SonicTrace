from typing import List, Dict, Optional, Tuple
import os
import logging
import torch
import torchaudio
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from app.diarization.base import Diarizer
from app.config import settings
logger = logging.getLogger(__name__)
class SileroDiarizer(Diarizer):
    """
    Adaptive speaker diarization using Silero VAD + Resemblyzer embeddings.
    Key features:
    - Automatic threshold selection based on embedding distribution
    - Multiple clustering strategies with quality scoring
    - No manual tuning required per audio file
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        win_size_s: float = 2.0,
        hop_s: float = 1.0,
        min_segment_duration: float = 0.3,
        min_speakers: int = 1,
        max_speakers: int = 10,
    ):
        """
        Args:
            sample_rate: Audio sample rate
            win_size_s: Window size for embeddings (seconds)
            hop_s: Hop size for sliding windows (seconds)
            min_segment_duration: Minimum segment duration to keep (seconds)
            min_speakers: Minimum expected speakers (for validation)
            max_speakers: Maximum expected speakers (for validation)
        """
        self.sample_rate = sample_rate
        self.device = settings.device

        # Window parameters
        self.win_size_s = win_size_s
        self.hop_s = hop_s
        
        # Clustering parameters
        self.min_segment_duration = min_segment_duration
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

        # Load Resemblyzer speaker encoder
        try:
            from resemblyzer import VoiceEncoder
            self.encoder = VoiceEncoder(device=str(self.device))
            logger.info("Resemblyzer VoiceEncoder loaded")
        except ImportError:
            raise ImportError(
                "resemblyzer is required. Install with: pip install resemblyzer"
            )

        # Load Silero VAD
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self.model.to(self.device)
        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks,
        ) = self.utils

        logger.info(
            "SileroDiarizer initialized - device=%s, win=%.1fs, "
            "speaker_range=[%d, %d]",
            self.device, self.win_size_s, self.min_speakers, self.max_speakers
        )

    def _load_audio(self, audio_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        
        wav, sr = torchaudio.load(audio_path)
        
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(
                wav, orig_freq=sr, new_freq=self.sample_rate
            )
        
        wav = wav.to(self.device)
        wav_np = wav.squeeze(0).cpu().numpy()
        
        return wav, wav_np

    def _get_speech_segments(self, wav: torch.Tensor) -> List[Dict]:
        """Run VAD to detect speech segments."""
        speech_ts = self.get_speech_timestamps(
            wav.squeeze(0), 
            self.model, 
            sampling_rate=self.sample_rate,
            threshold=0.5,
            min_speech_duration_ms=300,
            min_silence_duration_ms=100,
        )
        
        segments = [
            {"start": int(ts["start"]), "end": int(ts["end"])}
            for ts in speech_ts
        ]
        
        return segments

    def _create_embedding_windows(
        self, 
        speech_segments: List[Dict],
        wav_np: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Create sliding windows and extract speaker embeddings."""
        win_samples = int(self.win_size_s * self.sample_rate)
        hop_samples = int(self.hop_s * self.sample_rate)
        
        embeddings = []
        windows = []
        
        for seg_idx, seg in enumerate(speech_segments):
            start_sample = seg["start"]
            end_sample = seg["end"]
            seg_duration = (end_sample - start_sample) / self.sample_rate
            
            if seg_duration < 0.5:
                continue
            
            # Sliding windows
            pos = start_sample
            while pos + win_samples <= end_sample:
                window_audio = wav_np[pos:pos + win_samples]
                
                try:
                    if window_audio.dtype != np.float32:
                        window_audio = window_audio.astype(np.float32)
                    
                    max_amp = np.abs(window_audio).max()
                    if max_amp > 0:
                        window_audio = window_audio / max_amp
                    
                    embedding = self.encoder.embed_utterance(window_audio)
                    
                    if np.isfinite(embedding).all():
                        embeddings.append(embedding)
                        windows.append({
                            "start": pos,
                            "end": pos + win_samples,
                            "segment_idx": seg_idx
                        })
                        
                except Exception as e:
                    logger.debug("Failed to extract embedding: %s", e)
                
                pos += hop_samples
            
            # Tail window
            remaining = end_sample - pos
            if remaining > win_samples * 0.6:
                window_audio = wav_np[max(start_sample, end_sample - win_samples):end_sample]
                
                try:
                    if window_audio.dtype != np.float32:
                        window_audio = window_audio.astype(np.float32)
                    max_amp = np.abs(window_audio).max()
                    if max_amp > 0:
                        window_audio = window_audio / max_amp
                    
                    embedding = self.encoder.embed_utterance(window_audio)
                    if np.isfinite(embedding).all():
                        embeddings.append(embedding)
                        windows.append({
                            "start": max(start_sample, end_sample - win_samples),
                            "end": end_sample,
                            "segment_idx": seg_idx
                        })
                except Exception as e:
                    logger.debug("Failed to extract tail embedding: %s", e)
        
        if not embeddings:
            return np.array([]), []
        
        embeddings = np.stack(embeddings, axis=0)
        return embeddings, windows

    def _compute_distance_stats(self, embeddings: np.ndarray) -> Dict:
        """Compute pairwise distance statistics."""
        from sklearn.metrics.pairwise import cosine_distances
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embeddings_normalized = embeddings / norms
        
        # Compute pairwise distances
        distances = cosine_distances(embeddings_normalized)
        
        # Get upper triangle (unique pairs)
        triu_indices = np.triu_indices_from(distances, k=1)
        pairwise_dists = distances[triu_indices]
        
        return {
            "min": float(pairwise_dists.min()),
            "max": float(pairwise_dists.max()),
            "mean": float(pairwise_dists.mean()),
            "median": float(np.median(pairwise_dists)),
            "std": float(pairwise_dists.std()),
            "q25": float(np.percentile(pairwise_dists, 25)),
            "q75": float(np.percentile(pairwise_dists, 75)),
        }

    def _adaptive_clustering(
        self, 
        embeddings: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Adaptively cluster embeddings by trying multiple thresholds
        and selecting the best based on clustering quality metrics.
        """
        if len(embeddings) <= 1:
            return np.array([0] * len(embeddings)), {"method": "single"}
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embeddings_normalized = embeddings / norms
        
        # Get distance statistics
        dist_stats = self._compute_distance_stats(embeddings)
        logger.info("Distance stats: min=%.3f, max=%.3f, mean=%.3f, median=%.3f",
                dist_stats["min"], dist_stats["max"], 
                dist_stats["mean"], dist_stats["median"])
        
        # Generate candidate thresholds based on distance distribution
        # Strategy: Try multiple thresholds around key percentiles
        candidates = []
        
        # Conservative (fewer speakers)
        candidates.append(("high", dist_stats["q75"]))
        candidates.append(("med-high", (dist_stats["median"] + dist_stats["q75"]) / 2))
        
        # Moderate
        candidates.append(("median", dist_stats["median"]))
        
        # Aggressive (more speakers)
        candidates.append(("med-low", (dist_stats["q25"] + dist_stats["median"]) / 2))
        candidates.append(("low", dist_stats["q25"]))
        
        # Very aggressive (for 3+ speakers with clear separation)
        candidates.append(("very-low", dist_stats["q25"] * 0.8))
        
        best_result = None
        best_score = -float('inf')
        results = []
        
        for name, threshold in candidates:
            try:
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    metric="cosine",
                    linkage="average",
                    distance_threshold=threshold,
                )
            except TypeError:
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    affinity="cosine",
                    linkage="average",
                    distance_threshold=threshold,
                )
            
            labels = clustering.fit_predict(embeddings_normalized)
            n_clusters = len(np.unique(labels))
            
            # Skip if outside expected speaker range
            if n_clusters < self.min_speakers or n_clusters > self.max_speakers:
                continue
            
            # Compute quality score
            score = 0.0
            
            # 1. Silhouette score (if possible)
            if n_clusters > 1 and len(embeddings) > n_clusters:
                try:
                    silhouette = silhouette_score(
                        embeddings_normalized, labels, metric='cosine'
                    )
                    score += silhouette * 0.6  # 60% weight
                except:
                    silhouette = 0.0
            else:
                silhouette = 0.0
            
            # 2. Cluster size balance (penalize very unbalanced clusters)
            cluster_sizes = [int((labels == l).sum()) for l in np.unique(labels)]
            min_size = min(cluster_sizes)
            max_size = max(cluster_sizes)
            balance = min_size / max_size if max_size > 0 else 0
            score += balance * 0.2  # 20% weight
            
            # 3. Prefer 2-4 speakers (common case)
            if 2 <= n_clusters <= 4:
                score += 0.2  # 20% weight bonus
            
            results.append({
                "name": name,
                "threshold": threshold,
                "n_clusters": n_clusters,
                "silhouette": silhouette,
                "balance": balance,
                "score": score,
                "labels": labels.copy(),
            })
            
            logger.info(
                "  Threshold %.3f (%s): %d speakers, silhouette=%.3f, "
                "balance=%.3f, score=%.3f",
                threshold, name, n_clusters, silhouette, balance, score
            )
            
            if score > best_score:
                best_score = score
                best_result = results[-1]
        
        if best_result is None:
            # Fallback: use median threshold
            logger.warning("No valid clustering found, using median threshold")
            threshold = dist_stats["median"]
            try:
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    metric="cosine",
                    linkage="average",
                    distance_threshold=threshold,
                )
            except TypeError:
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    affinity="cosine",
                    linkage="average",
                    distance_threshold=threshold,
                )
            labels = clustering.fit_predict(embeddings_normalized)
            best_result = {
                "name": "fallback",
                "threshold": threshold,
                "n_clusters": len(np.unique(labels)),
                "score": 0.0,
                "labels": labels,
            }
        
        logger.info("✓ Selected: %s (threshold=%.3f, %d speakers, score=%.3f)",
                best_result["name"], best_result["threshold"],
                best_result["n_clusters"], best_result["score"])
        
        return best_result["labels"], {
            "threshold": best_result["threshold"],
            "n_clusters": best_result["n_clusters"],
            "silhouette": best_result.get("silhouette", 0.0),
            "method": best_result["name"],
            "distance_stats": dist_stats,
        }

    def _assign_segments_to_speakers(
        self,
        speech_segments: List[Dict],
        windows: List[Dict],
        labels: np.ndarray,
        sample_rate: int
    ) -> List[Dict]:
        """Assign speaker labels to segments using majority vote."""
        segment_window_labels = {}
        for win_idx, window in enumerate(windows):
            seg_idx = window["segment_idx"]
            segment_window_labels.setdefault(seg_idx, []).append(int(labels[win_idx]))
        
        results = []
        for seg_idx, seg in enumerate(speech_segments):
            if seg_idx not in segment_window_labels:
                continue
            
            # Majority vote
            window_labels = segment_window_labels[seg_idx]
            vals, counts = np.unique(window_labels, return_counts=True)
            speaker_id = int(vals[np.argmax(counts)]) + 1
            
            results.append({
                "start": seg["start"] / sample_rate,
                "end": seg["end"] / sample_rate,
                "speaker": f"Speaker {speaker_id}"
            })
        
        return results

    def _postprocess_segments(self, segments: List[Dict]) -> List[Dict]:
        """Merge adjacent same-speaker segments and remove very short ones."""
        if not segments:
            return []
        
        segments = sorted(segments, key=lambda x: x["start"])
        
        merged = []
        for seg in segments:
            duration = seg["end"] - seg["start"]
            
            if duration < self.min_segment_duration:
                if merged:
                    merged[-1]["end"] = seg["end"]
                continue
            
            if merged and merged[-1]["speaker"] == seg["speaker"]:
                gap = seg["start"] - merged[-1]["end"]
                if gap < 0.5:
                    merged[-1]["end"] = seg["end"]
                    continue
            
            merged.append(seg)
        
        for seg in merged:
            seg["start"] = round(float(seg["start"]), 3)
            seg["end"] = round(float(seg["end"]), 3)
        
        return merged

    def diarize(self, audio_path: str) -> List[Dict]:
        """
        Perform adaptive speaker diarization on audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of dicts with keys: start, end, speaker
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        logger.info("=" * 70)
        logger.info("Starting adaptive diarization: %s", os.path.basename(audio_path))
        
        # 1. Load audio
        wav, wav_np = self._load_audio(audio_path)
        duration = len(wav_np) / self.sample_rate
        logger.info("✓ Audio loaded: %.2fs", duration)
        
        # 2. VAD
        speech_segments = self._get_speech_segments(wav)
        
        if not speech_segments:
            logger.info("✗ No speech detected")
            return []
        
        total_speech = sum(s["end"] - s["start"] for s in speech_segments) / self.sample_rate
        logger.info(
            "✓ VAD: %d segments, %.2fs speech (%.1f%%)",
            len(speech_segments), total_speech, 100 * total_speech / duration
        )
        
        # 3. Extract embeddings
        embeddings, windows = self._create_embedding_windows(speech_segments, wav_np)
        
        if len(embeddings) == 0:
            logger.info("✗ No embeddings extracted")
            return []
        
        logger.info("✓ Extracted %d embeddings", len(embeddings))
        
        # 4. Adaptive clustering
        labels, cluster_info = self._adaptive_clustering(embeddings)
        
        # 5. Assign to segments
        segments = self._assign_segments_to_speakers(
            speech_segments, windows, labels, self.sample_rate
        )
        
        # 6. Post-process
        final_segments = self._postprocess_segments(segments)
        
        n_speakers = len(set(s["speaker"] for s in final_segments))
        logger.info("✓ Complete: %d speakers, %d segments", n_speakers, len(final_segments))
        logger.info("=" * 70)
        
        return final_segments