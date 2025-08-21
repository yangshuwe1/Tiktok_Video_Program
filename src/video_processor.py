import os
import ffmpeg
import hashlib
import re
import subprocess
import json
from logger import get_logger
logger = get_logger("video_processor")

class VideoProcessor:
    """Video processing: metadata extraction and keyframe extraction."""
    def __init__(self, config=None):
        self.logger = get_logger("VideoProcessor")
        self.config = config or {}
        self.frame_rate = self.config.get('frame_rate', 1)
        self.max_frames = self.config.get('max_frames', 50)
        self.should_extract_keyframes = self.config.get('extract_keyframes', True)
        self.should_extract_highlights = self.config.get('extract_highlights', True)
        self.logger.debug(f"VideoProcessor initialized with config: {self.config}")

    def get_video_metadata(self, video_path):
        self.logger.info(f"Extracting metadata for video: {video_path}")
        try:
            probe = ffmpeg.probe(video_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            duration = float(probe['format']['duration'])
            width = int(video_info['width'])
            height = int(video_info['height'])
            if 'r_frame_rate' in video_info:
                frame_rate_str = video_info['r_frame_rate']
                if '/' in frame_rate_str:
                    num, den = map(int, frame_rate_str.split('/'))
                    frame_rate = num / den if den != 0 else 0
                else:
                    frame_rate = float(frame_rate_str)
            else:
                frame_rate = 0
            file_size = int(probe['format']['size'])
            self.logger.debug(f"Metadata extracted: duration={duration}, width={width}, height={height}, frame_rate={frame_rate}, file_size={file_size}")
            return {
                'duration': duration,
                'width': width,
                'height': height,
                'frame_rate': frame_rate,
                'file_size': file_size
            }
        except Exception as e:
            self.logger.error(f"Error extracting metadata from {video_path}: {e}")
            return {
                'duration': 0,
                'width': 0,
                'height': 0,
                'frame_rate': 0,
                'file_size': 0
            }

    def extract_speech_highlight_frames(self, video_path, output_dir, speech_segments, confidence_threshold=0.7):
        highlight_times = []
        for seg in speech_segments:
            if seg.get('confidence', 0) >= confidence_threshold:
                t = (seg['start'] + seg['end']) / 2
                highlight_times.append(t)
        highlight_files = []
        for idx, t in enumerate(highlight_times):
            out_path = os.path.join(output_dir, f"speech_highlight_{idx:03d}_{t:.2f}.jpg")
            try:
                (
                    ffmpeg
                    .input(video_path, ss=t)
                    .output(out_path, vframes=1)
                    .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
                )
                highlight_files.append({'file': out_path, 'timestamp': t})
            except Exception as e:
                self.logger.error(f"Error extracting speech highlight frame at {t}s: {e}")
        return highlight_files

    def extract_keyframes(self, video_path, output_dir, speech_segments=None):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        keyframes_dir = output_dir
        os.makedirs(keyframes_dir, exist_ok=True)
        
        # Check if keyframe extraction is enabled
        if not self.should_extract_keyframes:
            self.logger.info(f"Keyframe extraction disabled for {video_name}")
            return keyframes_dir, 0

        # Batch extract I-frames using ffmpeg select filter
        i_frame_pattern = os.path.join(keyframes_dir, f"{video_name}_keyframe_%04d.jpg")
        try:
            (
                ffmpeg
                .input(video_path)
                .output(i_frame_pattern, vf="select=eq(pict_type\\,I)", vsync="vfr")
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )
        except Exception as e:
            self.logger.error(f"Error batch extracting I-frames: {e}")

        # Sampled frames (using configured frame rate)
        duration = 0
        try:
            probe = ffmpeg.probe(video_path)
            duration = float(probe['format']['duration'])
        except Exception as e:
            self.logger.error(f"Error getting video duration: {e}")
        
        # Use configured frame rate
        fps = self.frame_rate
        # Batch extract sampled frames using ffmpeg -vf fps=fps
        sampled_out_pattern = os.path.join(keyframes_dir, f"{video_name}_sampled_%04d.jpg")
        try:
            (
                ffmpeg
                .input(video_path)
                .output(sampled_out_pattern, vf=f"fps={fps}", vframes=self.max_frames)
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )
        except Exception as e:
            self.logger.error(f"Error batch extracting sampled frames: {e}")

        # Highlight frames
        # if speech_segments is not None and self.should_extract_highlights:
        #     highlight_frames = self.extract_speech_highlight_frames(video_path, output_dir, speech_segments, confidence_threshold=0.7)
        #     for idx, item in enumerate(highlight_frames):
        #         t = item['timestamp']
        #         out_path = os.path.join(output_dir, f"speech_highlight_{idx:03d}_{t:.2f}.jpg")
        #         os.rename(item['file'], out_path)
        #         item['file'] = out_path

        keyframe_count = len([f for f in os.listdir(keyframes_dir) if f.endswith('.jpg')])
        return keyframes_dir, keyframe_count 