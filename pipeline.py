def process_gauze_image(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    H, W = img.shape[:2]

    pred = model.predict(
        img,
        conf=0.5,
        imgsz=640,
        verbose=False
    )[0]

    boxes = pred.boxes.xyxy.cpu().numpy()
    confs = pred.boxes.conf.cpu().numpy()
    N = len(boxes)

    # HSV-C strict
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blood_mask = cv2.bitwise_or(
        cv2.inRange(hsv, (0,130,30), (10,255,255)),
        cv2.inRange(hsv, (175,130,30), (179,255,255))
    )

    owner = np.full((H, W), -1, dtype=np.int32)

    # Handle no detections
    if N == 0:
        return {
            "detections": 0,
            "gauze_pixels": 0,
            "blood_pixels": 0,
            "stained_pct": np.nan,
            "overlap_pixels_before": 0,
            "owner": owner,
            "blood_mask": blood_mask,
            "records": []
        }

    gauze_masks = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)

        x1 = max(0, min(W, x1))
        y1 = max(0, min(H, y1))
        x2 = max(0, min(W, x2))
        y2 = max(0, min(H, y2))

        full_mask = np.zeros((H, W), dtype=np.uint8)

        if x2 <= x1 or y2 <= y1:
            gauze_masks.append(full_mask)
            continue

        pad = 12
        px1, py1 = max(0, x1-pad), max(0, y1-pad)
        px2, py2 = min(W, x2+pad), min(H, y2+pad)

        crop = img[py1:py2, px1:px2].copy()
        ch, cw = crop.shape[:2]

        rx1, ry1 = x1-px1, y1-py1
        rx2, ry2 = x2-px1, y2-py1

        gc_mask = np.full(
            (ch, cw), cv2.GC_PR_BGD, dtype=np.uint8
        )

        gc_mask[ry1:ry2, rx1:rx2] = cv2.GC_PR_FGD

        ix = max(1, (rx2-rx1)//5)
        iy = max(1, (ry2-ry1)//5)

        if rx2-rx1 > 2*ix and ry2-ry1 > 2*iy:
            gc_mask[
                ry1+iy:ry2-iy,
                rx1+ix:rx2-ix
            ] = cv2.GC_FGD

        bgd = np.zeros((1,65), np.float64)
        fgd = np.zeros((1,65), np.float64)

        try:
            # GrabCut requires foreground and background seeds
            if not np.any(gc_mask == cv2.GC_FGD):
                raise ValueError("No definite foreground seed")

            if not np.any(
                (gc_mask == cv2.GC_BGD) |
                (gc_mask == cv2.GC_PR_BGD)
            ):
                raise ValueError("No background seed")

            cv2.grabCut(
                crop, gc_mask, None, bgd, fgd,
                5, cv2.GC_INIT_WITH_MASK
            )

            local = np.where(
                (gc_mask == cv2.GC_FGD) |
                (gc_mask == cv2.GC_PR_FGD),
                255, 0
            ).astype(np.uint8)

        except (cv2.error, ValueError):
            # Fallback: bounding box, flagged indirectly
            # by the resulting segmentation quality
            local = np.zeros((ch, cw), dtype=np.uint8)
            local[ry1:ry2, rx1:rx2] = 255

        # Restrict foreground to YOLO bounding box
        box_only = np.zeros((ch, cw), dtype=np.uint8)
        box_only[ry1:ry2, rx1:rx2] = 255
        local = cv2.bitwise_and(local, box_only)

        full_mask[py1:py2, px1:px2] = local
        gauze_masks.append(full_mask)

    masks = np.stack(
        [(m > 0) for m in gauze_masks],
        axis=0
    )

    membership = masks.sum(axis=0)
    overlap_before = int(np.count_nonzero(membership >= 2))

    # Single-owner pixels: guard empty selection
    single = membership == 1
    if np.any(single):
        owner[single] = np.argmax(
            masks[:, single], axis=0
        ).astype(np.int32)

    # Overlapping pixels: nearest normalized box center
    cx = (boxes[:,0] + boxes[:,2]) / 2
    cy = (boxes[:,1] + boxes[:,3]) / 2
    bw = np.maximum(boxes[:,2] - boxes[:,0], 1)
    bh = np.maximum(boxes[:,3] - boxes[:,1], 1)

    ys, xs = np.where(membership >= 2)

    for y, x in zip(ys, xs):
        candidates = np.flatnonzero(masks[:,y,x])
        if len(candidates) == 0:
            continue

        distances = (
            ((x-cx[candidates])/bw[candidates])**2 +
            ((y-cy[candidates])/bh[candidates])**2
        )

        owner[y,x] = int(
            candidates[np.argmin(distances)]
        )

    records = []

    for i in range(N):
        gm = owner == i
        bm = gm & (blood_mask > 0)

        gp = int(np.count_nonzero(gm))
        bp = int(np.count_nonzero(bm))

        records.append({
            "Detection": i+1,
            "Confidence": float(confs[i]),
            "Gauze_pixels": gp,
            "Blood_pixels": bp,
            "Stained_area_pct": (
                100*bp/gp if gp > 0 else np.nan
            )
        })

    gauze_total = int(np.count_nonzero(owner >= 0))
    blood_total = int(np.count_nonzero(
        (owner >= 0) & (blood_mask > 0)
    ))

    assert sum(r["Gauze_pixels"] for r in records) == gauze_total
    assert sum(r["Blood_pixels"] for r in records) == blood_total

    return {
        "detections": N,
        "gauze_pixels": gauze_total,
        "blood_pixels": blood_total,
        "stained_pct": (
            100*blood_total/gauze_total
            if gauze_total > 0 else np.nan
        ),
        "overlap_pixels_before": overlap_before,
        "owner": owner,
        "blood_mask": blood_mask,
        "records": records
    }
