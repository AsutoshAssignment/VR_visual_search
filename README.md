# Setup Instructions

## 1. Clone the Repository

```bash
git clone https://github.com/AsutoshAssignment/VR_visual_search
cd VR_visual_search
```

## 2. Download the `report` Folder

Download the `report` folder from the following Google Drive link:

https://drive.google.com/drive/u/0/folders/1Wa8gpdcKxBOwjH1TEqf0S29j92dWlZGQ

## 3. Place the Folder

Move the downloaded `report` folder into the root directory of the cloned repository.

Expected structure:

```text
VR_visual_search/
│
├── report/
├── streamlit_app/
├── requirements.txt
└── ...
```

## 4. Create and Activate a Virtual Environment

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## 6. Run the Application

```bash
streamlit run streamlit_app/app.py
```

## 7. Upload an Image

- Wait for the application to load completely
- The upload button will appear shortly
- Upload an image to begin the visual product search

## Note

The first startup may take some time because the CLIP and BLIP models need to be downloaded and loaded into memory.
