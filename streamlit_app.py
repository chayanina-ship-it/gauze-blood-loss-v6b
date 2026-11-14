"""Streamlit frontend for the unchanged frozen V6B pipeline.py."""
import io
import importlib.util
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / 'best.pt'
PIPELINE = ROOT / 'pipeline.py'

st.set_page_config(page_title='Gauze V6B Validator', layout='wide')
st.title('Gauze Blood Loss Proxy — V6B-Final-1.0')
st.info('Research-use proxy only, not measured blood volume. Fully unfold gauzes, place them flat and separately without overlap, and photograph from above in even lighting.')

@st.cache_resource(show_spinner='Loading frozen V6B model…')
def load_pipeline():
    if not WEIGHTS.is_file() or not PIPELINE.is_file():
        raise FileNotFoundError('Both best.pt and pipeline.py must be in the app root.')
    yolo = YOLO(str(WEIGHTS))
    spec = importlib.util.spec_from_file_location('frozen_v6b_pipeline', PIPELINE)
    module = importlib.util.module_from_spec(spec)
    module.cv2 = cv2
    module.np = np
    module.model = yolo
    spec.loader.exec_module(module)
    if not callable(getattr(module, 'process_gauze_image', None)):
        raise RuntimeError('pipeline.py does not define process_gauze_image')
    return module.process_gauze_image

uploaded = st.file_uploader('Upload an unfolded, non-overlapping gauze image', type=['jpg', 'jpeg', 'png'])
if uploaded is not None:
    raw = uploaded.getvalue()
    image_arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
    if img is None:
        st.error('Cannot decode image. Please upload a valid JPG or PNG.')
        st.stop()
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption='Uploaded image', use_container_width=True)
    eligibility = st.checkbox('I confirm gauzes are fully unfolded, separate, non-overlapping and clearly visible.')
    if st.button('Analyze with frozen V6B', type='primary'):
        if not eligibility:
            st.warning('Out-of-protocol image: unfold and separate the gauzes before standard validation. You may still run a challenge test by checking the confirmation only if the condition is true.')
            st.stop()
        suffix = Path(uploaded.name).suffix.lower()
        temp_path = None
        try:
            fn = load_pipeline()
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(raw)
                temp_path = Path(f.name)
            with st.spinner('Analyzing image…'):
                result = fn(str(temp_path))
            count = int(result['detections'])
            st.metric('YOLO detections (not necessarily true gauze count)', count)
            if count == 0:
                st.warning('NO_DETECTION — Blood loss proxy is NOT ESTIMABLE (not zero).')
            else:
                records = []
                for r in result['records']:
                    g = int(r['Gauze_pixels'])
                    b = int(r['Blood_pixels'])
                    proxy = 10.0 * b / g if g > 0 else np.nan
                    records.append({**r, 'Proxy_mL_equiv': proxy})
                df = pd.DataFrame(records)
                total = float(df['Proxy_mL_equiv'].sum(min_count=1)) if len(df) else np.nan
                c1, c2, c3 = st.columns(3)
                c1.metric('Stained area', f"{result['stained_pct']:.2f}%" if np.isfinite(result['stained_pct']) else 'N/A')
                c2.metric('Blood loss proxy', f'{total:.3f} mL-eq' if np.isfinite(total) else 'Not estimable')
                c3.metric('Overlapping mask pixels (before correction)', int(result['overlap_pixels_before']))
                st.dataframe(df, use_container_width=True)
                st.download_button('Download per-detection CSV', df.to_csv(index=False).encode('utf-8-sig'), file_name='v6b_detection_results.csv', mime='text/csv')
                owner = result['owner']
                blood = result['blood_mask'] > 0
                overlay = img.copy()
                # Visualize model-assigned gauze pixels in green and blood pixels in red.
                assigned = owner >= 0
                overlay[assigned] = (0.55 * overlay[assigned] + 0.45 * np.array([0, 200, 0])).astype(np.uint8)
                stained = assigned & blood
                overlay[stained] = (0.45 * img[stained] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption='Assigned gauze pixels (green), HSV-C stained pixels (red)', use_container_width=True)
                ok, encoded = cv2.imencode('.png', overlay)
                if ok:
                    st.download_button('Download overlay PNG', encoded.tobytes(), file_name='v6b_overlay.png', mime='image/png')
            st.caption('This output is an assumed 10 mL-equivalent per fully stained detection region, not a measurement of actual blood volume. Detection errors can change the sum.')
        except Exception as exc:
            st.error(f'Analysis failed: {type(exc).__name__}: {exc}')
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
