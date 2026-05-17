# Setup Instructions

## 1. Clone the Repository

```bash
git clone https://github.com/AsutoshAssignment/VR_visual_search
cd VR_visual_search
```

## 2. Download the `report` Folder

Download the clip_fine_tune file from the following Google Drive link:

https://drive.google.com/file/d/1s1a--6uAmNNTOLEKJF2L2Buq2WB5qdz4/view?usp=sharing

## 3. Place the Folder

Move the downloaded file into the result/experiment_C directory of the cloned repository.

Expected structure:

```text
VR_visual_search/
│
├── results/experiment_C/clip_visual_finetuned.pth
├── streamlit_app/
├── requirements.txt
└── ...
```


## BONUS. If you want all the image suggestion to be if the same dimensions :
Download the "raw" folder from the following Google Drive link:

https://drive.google.com/drive/folders/1Dk-p0ov6jb-RUETjkoTg9b97Dxs2KA33?usp=sharing

## Place the Folder

Move the downloaded folder into the data folder of the cloned repository.

Expected structure:

```text
VR_visual_search/
│
├── data/raw/
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
