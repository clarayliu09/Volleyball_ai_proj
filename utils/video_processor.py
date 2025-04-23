import os
from typing import Optional, Tuple
import yt_dlp
import cv2
import numpy as np
from datetime import datetime
import subprocess
from pathlib import Path
import tempfile


class VideoProcessor:

    def __init__(self, output_dir: str = "downloads"):
        """Initialize the video processor with an output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.downloads_dir = Path(output_dir)
        self.downloads_dir.mkdir(exist_ok=True)

    def download_video(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Download a video from YouTube and return the path to the downloaded file.
        
        Args:
            url (str): YouTube video URL
            
        Returns:
            Tuple[Optional[str], Optional[str]]: (file path, error message)
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"serve_video_{timestamp}.mp4"
            output_path = os.path.join(self.output_dir, filename)

            # First, get available formats
            format_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'nocheckcertificate': True,
                'ignoreerrors': True,
                'no_color': True,
                'geo_bypass': True,
                'extractor_retries': 3,
                'retries': 3,
                'fragment_retries': 3,
                'file_access_retries': 3
            }

            with yt_dlp.YoutubeDL(format_opts) as ydl:
                try:
                    # Extract video info to get available formats
                    info = ydl.extract_info(url, download=False)
                    if info is None:
                        return None, "Could not extract video information. Please check if the URL is valid and accessible."
                    
                    # Get available formats
                    formats = info.get('formats', [])
                    if not formats:
                        return None, "No downloadable formats found for this video."
                    
                    # Find the best format that's downloadable
                    best_format = None
                    for f in formats:
                        if f.get('vcodec', 'none') != 'none' and f.get('acodec', 'none') != 'none':
                            best_format = f.get('format_id')
                            break
                    
                    if not best_format:
                        # If no combined format found, try to find best video and audio separately
                        video_format = None
                        audio_format = None
                        for f in formats:
                            if f.get('vcodec', 'none') != 'none' and not video_format:
                                video_format = f.get('format_id')
                            if f.get('acodec', 'none') != 'none' and not audio_format:
                                audio_format = f.get('format_id')
                        
                        if video_format and audio_format:
                            best_format = f"{video_format}+{audio_format}"
                        elif video_format:
                            best_format = video_format
                        else:
                            return None, "No suitable video format found."
                    
                    # Now download with the selected format
                    download_opts = {
                        'format': best_format,
                        'outtmpl': output_path,
                        'quiet': True,
                        'no_warnings': True,
                        'extract_flat': False,
                        'nocheckcertificate': True,
                        'ignoreerrors': True,
                        'no_color': True,
                        'geo_bypass': True,
                        'extractor_retries': 3,
                        'retries': 3,
                        'fragment_retries': 3,
                        'file_access_retries': 3
                    }
                    
                    with yt_dlp.YoutubeDL(download_opts) as download_ydl:
                        download_ydl.download([url])
                    
                except Exception as e:
                    return None, f"Error processing video: {str(e)}"

            if os.path.exists(output_path):
                return output_path, None
            return None, "Download completed but file not found."

        except Exception as e:
            return None, f"Unexpected error: {str(e)}"

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Process a single frame to detect volleyball and track movement.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            tuple: (processed frame, detection results)
        """
        # Convert to RGB for processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a copy for drawing
        output_frame = frame.copy()

        # TODO: Implement ball detection using YOLO or other object detection
        # For now, return placeholder data
        results = {
            "ball_detected": False,
            "ball_position": None,
            "frame_height": frame.shape[0],
            "frame_width": frame.shape[1]
        }

        return output_frame, results

    def extract_video_frames(self, video_path: str, frame_interval: int = 1):
        """
        Extract frames from a video file at specified intervals.
        
        Args:
            video_path (str): Path to the video file
            frame_interval (int): Number of frames to skip between extractions
            
        Yields:
            tuple: (frame number, frame data)
        """
        cap = cv2.VideoCapture(video_path)
        # Set video capture properties for better quality
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Increase buffer size
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                # Convert to RGB for better quality
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame_count, frame

            frame_count += 1

        cap.release()

    def save_annotated_frame(self, frame: np.ndarray,
                             frame_number: int, output_dir: str = None) -> str:
        """
        Save an annotated frame to disk.
        
        Args:
            frame (np.ndarray): The frame to save
            frame_number (int): The frame number
            output_dir (str, optional): Directory to save the frame (defaults to temp directory)
            
        Returns:
            str: Path to the saved frame
        """
        if output_dir is None:
            output_dir = tempfile.gettempdir()
            
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        frame_path = output_dir / f"frame_{frame_number:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        return frame_path
        
    def save_annotated_video(self, original_video_path: str, annotated_frames: list, output_filename: str = None) -> str:
        """
        Save annotated frames as a video.
        
        Args:
            original_video_path (str): Path to the original video for getting properties
            annotated_frames (list): List of annotated frames to save
            output_filename (str, optional): Custom filename for the output video
            
        Returns:
            str: Path to the saved video
        """
        # Get video properties from original video
        cap = cv2.VideoCapture(original_video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        
        # Create output directory if needed
        output_dir = self.downloads_dir / "annotated"
        output_dir.mkdir(exist_ok=True)
        
        # Generate output filename if not provided
        if output_filename is None:
            original_name = Path(original_video_path).stem
            output_filename = f"{original_name}_annotated.mp4"
            
        output_path = output_dir / output_filename
        
        # Create video writer with high quality settings
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Write frames with quality preservation
        for frame in annotated_frames:
            # Convert back to BGR for saving
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            
        out.release()
        
        # Use ffmpeg to ensure high quality output
        try:
            temp_path = output_path.with_suffix('.temp.mp4')
            ffmpeg_path = os.path.expanduser('~/bin/ffmpeg')
            subprocess.run([
                ffmpeg_path, '-i', str(output_path),
                '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',
                '-c:a', 'aac', '-b:a', '192k',
                str(temp_path)
            ], check=True)
            temp_path.replace(output_path)
        except Exception as e:
            print(f"Warning: Could not optimize video with ffmpeg: {str(e)}")
            # If ffmpeg fails, we'll still return the original video
            print("Using original video without optimization")
        
        return output_path
