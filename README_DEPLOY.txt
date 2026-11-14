Gauze V6B Streamlit deployment

Keep these four files in the ROOT of your GitHub repository:
  best.pt (your ORIGINAL frozen model, already uploaded)
  pipeline.py (your ORIGINAL frozen pipeline, already uploaded)
  streamlit_app.py (from this package)
  requirements.txt (from this package)

DO NOT replace your original pipeline.py or best.pt.

Streamlit Community Cloud:
Repository: chayanina-ship-it/gauze-blood-loss-v6b
Branch: main
Main file path: streamlit_app.py

The app uses CPU by default and may take time to initialize. For research only.
Images uploaded to a public app should not contain identifying patient information.

Check reproducibility on the same test image against Colab outputs before using for a manuscript. Different dependency versions and runtimes may affect results.
