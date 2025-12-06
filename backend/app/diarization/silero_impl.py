from typing import List, Dict, Tuple
import os

import torch
import torchaudio
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

from app.diarization.base import Diarizer
from app.config import settings


class SileroDiarizer(Diarizer):
    """
    Adaptive speaker diarization using Silero VAD + Resemblyzer embeddings.
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
        self.sample_rate = sample_rate
        self.device = settings.device
        self.win_size_s = win_size_s
        self.hop_s = hop_s
        self.min_segment_duration = min_segment_duration
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

        # Load Resemblyzer
        try:
            from resemblyzer import VoiceEncoder
            self.encoder = VoiceEncoder(device=str(self.device))
        except ImportError:
            raise ImportError("resemblyzer required: pip install resemblyzer")

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
        speech_ts = self.get_speech_timestamps(
            wav.squeeze(0), 
            self.model, 
            sampling_rate=self.sample_rate,
            threshold=0.5,
            min_speech_duration_ms=300,
            min_silence_duration_ms=100,
        )
        
        return [
            {"start": int(ts["start"]), "end": int(ts["end"])}
            for ts in speech_ts
        ]

    def _create_embedding_windows(
        self, 
        speech_segments: List[Dict],
        wav_np: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict]]:
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
                except:
                    pass
                
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
                except:
                    pass
        
        if not embeddings:
            return np.array([]), []
        
        return np.stack(embeddings, axis=0), windows

    def _compute_distance_stats(self, embeddings: np.ndarray) -> Dict:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embeddings_normalized = embeddings / norms
        
        distances = cosine_distances(embeddings_normalized)
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

    def _adaptive_clustering(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict]:
        if len(embeddings) <= 1:
            return np.array([0] * len(embeddings)), {"method": "single"}
        
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embeddings_normalized = embeddings / norms
        
        dist_stats = self._compute_distance_stats(embeddings)
        
        # Generate candidate thresholds
        candidates = [
            ("high", dist_stats["q75"]),
            ("med-high", (dist_stats["median"] + dist_stats["q75"]) / 2),
            ("median", dist_stats["median"]),
            ("med-low", (dist_stats["q25"] + dist_stats["median"]) / 2),
            ("low", dist_stats["q25"]),
            ("very-low", dist_stats["q25"] * 0.8),
        ]
        
        best_result = None
        best_score = -float('inf')
        
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
            
            if n_clusters < self.min_speakers or n_clusters > self.max_speakers:
                continue
            
            score = 0.0
            
            # Silhouette score
            if n_clusters > 1 and len(embeddings) > n_clusters:
                try:
                    silhouette = silhouette_score(
                        embeddings_normalized, labels, metric='cosine'
                    )
                    score += silhouette * 0.6
                except:
                    pass
            
            # Cluster balance
            cluster_sizes = [int((labels == l).sum()) for l in np.unique(labels)]
            min_size = min(cluster_sizes)
            max_size = max(cluster_sizes)
            balance = min_size / max_size if max_size > 0 else 0
            score += balance * 0.2
            
            # Prefer 2-4 speakers
            if 2 <= n_clusters <= 4:
                score += 0.2
            
            if score > best_score:
                best_score = score
                best_result = {
                    "name": name,
                    "threshold": threshold,
                    "n_clusters": n_clusters,
                    "score": score,
                    "labels": labels.copy(),
                }
        
        if best_result is None:
            # Fallback
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
                "labels": labels,
            }
        
        return best_result["labels"], {
            "threshold": best_result["threshold"],
            "n_clusters": best_result["n_clusters"],
            "method": best_result["name"],
        }

    def _merge_similar_speakers(
        self, 
        embeddings: np.ndarray, 
        labels: np.ndarray,
        similarity_threshold: float = 0.85
    ) -> np.ndarray:
        unique_labels = np.unique(labels)
        n_speakers = len(unique_labels)
        
        if n_speakers <= 1:
            return labels
        
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embeddings_normalized = embeddings / norms
        
        # Compute centroids
        centroids = {}
        for label in unique_labels:
            mask = labels == label
            centroids[int(label)] = embeddings_normalized[mask].mean(axis=0)
        
        # Pairwise similarities
        speaker_ids = sorted(centroids.keys())
        n = len(speaker_ids)
        similarities = np.zeros((n, n))
        
        for i, sp1 in enumerate(speaker_ids):
            for j, sp2 in enumerate(speaker_ids):
                if i < j:
                    sim = cosine_similarity(
                        centroids[sp1].reshape(1, -1),
                        centroids[sp2].reshape(1, -1)
                    )[0, 0]
                    similarities[i, j] = sim
                    similarities[j, i] = sim
        
        # Merge similar speakers
        merged = False
        for i, sp1 in enumerate(speaker_ids):
            for j, sp2 in enumerate(speaker_ids):
                if i < j and similarities[i, j] >= similarity_threshold:
                    labels[labels == sp2] = sp1
                    merged = True
        
        if merged:
            unique_labels, inverse = np.unique(labels, return_inverse=True)
            labels = inverse
        
        return labels

    def _assign_segments_to_speakers(
        self,
        speech_segments: List[Dict],
        windows: List[Dict],
        labels: np.ndarray,
        sample_rate: int
    ) -> List[Dict]:
        segment_window_labels = {}
        for win_idx, window in enumerate(windows):
            seg_idx = window["segment_idx"]
            segment_window_labels.setdefault(seg_idx, []).append(int(labels[win_idx]))
        
        results = []
        for seg_idx, seg in enumerate(speech_segments):
            if seg_idx not in segment_window_labels:
                continue
            
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
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")
        
        # Load audio
        wav, wav_np = self._load_audio(audio_path)
        
        # VAD
        speech_segments = self._get_speech_segments(wav)
        if not speech_segments:
            return []
        
        # Extract embeddings
        embeddings, windows = self._create_embedding_windows(speech_segments, wav_np)
        if len(embeddings) == 0:
            return []
        
        # Adaptive clustering
        labels, cluster_info = self._adaptive_clustering(embeddings)
        
        # Merge similar speakers
        labels = self._merge_similar_speakers(embeddings, labels)
        
        # Assign to segments
        segments = self._assign_segments_to_speakers(
            speech_segments, windows, labels, self.sample_rate
        )
        
        # Post-process
        final_segments = self._postprocess_segments(segments)
        
        return final_segments