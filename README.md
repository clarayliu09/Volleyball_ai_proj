# Volleyball Serving Analysis

A computer vision and AI-powered application for analyzing volleyball serving techniques. This application uses pose detection and AI to provide detailed feedback on serving form and technique.

## Features

- Upload videos or provide YouTube URLs for analysis
- Real-time pose detection and tracking
- Comprehensive motion analysis
- AI-powered technique feedback (requires OpenAI API key)
- Detailed metrics and visualizations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/volleyball-serving-analysis.git
cd volleyball-serving-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up OpenAI API key (optional, for AI feedback):
   - Create a `.env` file in the project root
   - Add your OpenAI API key: `OPENAI_API_KEY=your_key_here`

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Choose your input method:
   - Upload a video file (MP4, MOV, or AVI)
   - Provide a YouTube URL

3. Click "Analyze" to process the video

4. View the results:
   - Pose analysis with annotated video
   - Motion metrics
   - Comprehensive analysis
   - AI technique feedback (if API key is configured)

## Project Structure

```
volleyball-serving-analysis/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (API keys)
├── utils/
│   ├── video_processor.py # Video processing utilities
│   ├── pose_analyzer.py   # Pose detection and analysis
│   └── gpt_analyzer.py    # AI analysis using GPT
└── downloads/            # Temporary storage for videos
```

## Dependencies

- Python 3.9+
- OpenCV
- MediaPipe
- Streamlit
- OpenAI API (optional)
- yt-dlp

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe for pose detection
- OpenAI for GPT integration
- Streamlit for the web interface 