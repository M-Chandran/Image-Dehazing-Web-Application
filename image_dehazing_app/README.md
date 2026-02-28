# Image Dehazing Web Application

A Flask-based web application for removing haze and fog from images using advanced computer vision and deep learning algorithms.

## Features

- **User Authentication**: Secure user registration and login system
- **Image Upload**: Easy drag-and-drop or file selection upload
- **Dehazing Processing**: Advanced algorithms to remove haze, fog, and atmospheric pollution from images
- **Step-by-Step Visualization**: View each processing step (preprocessing, model inference, post-processing, etc.)
- **Download Results**: Download the dehazed image in high quality
- **Responsive Design**: Modern, mobile-friendly web interface

## Technology Stack

- **Backend**: Flask (Python web framework)
- **Image Processing**: OpenCV, NumPy
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Docker, Docker Compose

## Prerequisites

- Python 3.8 or higher
- Docker (optional, for containerized deployment)

## Installation

### Local Development

1. Clone the repository:
```
bash
git clone <repository-url>
cd image_dehazing_app
```

2. Create a virtual environment (recommended):
```
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```
bash
pip install -r requirements.txt
```

4. Run the application:
```
bash
python app.py
```

5. Open http://localhost:5000 in your browser

### Docker Deployment

1. Build and run with Docker Compose:
```
bash
docker-compose up --build
```

2. Open http://localhost:5000 in your browser

## Usage

1. **Register**: Create a new account by clicking "Sign Up"
2. **Login**: Use your credentials to log in
3. **Upload**: Select or drag-and-drop a hazy image (PNG, JPG, JPEG, GIF)
4. **Process**: Click the dehaze button to process your image
5. **Download**: View the results and download the dehazed image

## Project Structure

```
image_dehazing_app/
├── app.py                 # Main Flask application
├── dehazing/
│   └── dehaze.py         # Dehazing algorithm implementation
├── static/
│   ├── css/
│   │   └── style.css     # Styling
│   ├── js/
│   │   └── script.js     # Frontend JavaScript
│   ├── uploads/          # User uploaded images
│   └── outputs/          # Processed images
├── templates/
│   ├── base.html         # Base template
│   ├── dashboard.html    # Main dashboard
│   ├── login.html        # Login page
│   ├── signup.html       # Registration page
│   └── processing.html   # Processing animation
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
└── requirements.txt      # Python dependencies
```

## Configuration

### Environment Variables

- `FLASK_APP`: Flask application entry point (default: app.py)
- `FLASK_ENV`: Environment mode (development/production)
- `SECRET_KEY`: Application secret key for sessions

### Customization

To change the secret key, update the `app.secret_key` in `app.py` or set it as an environment variable.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
