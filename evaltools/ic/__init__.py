# import os
# from pycocotools.coco import COCO
# from pycocoevalcap.eval import COCOEvalCap

# def coco_caption_eval(annotation_file, results_file):
#     # 디코더에서 생성된 결과 파일을 평가하기 위한 함수
#     print("[Debug] evaltools/ic/__init__.py -> coco_caption_eval()함수 호출 : generate 결과 평가")
#     assert os.path.exists(annotation_file)

#     # create coco object and coco_result object
#     coco = COCO(annotation_file)
#     coco_result = coco.loadRes(results_file)

#     # create coco_eval object by taking coco and coco_result
#     coco_eval = COCOEvalCap(coco, coco_result)

#     # evaluate results
#     coco_eval.evaluate() #평가 객체 반환

#     # print output evaluation scores
#     for metric, score in coco_eval.eval.items():
#         print(f'{metric}: {score:.3f}', flush=True)

#     return coco_eval

# --------------------
# Failure Case 수집용 함수 (revised)
# --------------------
import os, json, numpy as np
from pathlib import Path
from pycocotools.coco         import COCO
from pycocoevalcap.eval       import COCOEvalCap


def coco_caption_eval(
    annotation_file,
    results_file,
    dataset_name="rsicd",
    mask_name="mask_90",
    base_out="results/failure_case/captioning",
    q=10,                               # 🔄 하위 q% (0 제외) 기준
    metrics=None                        # 🔄 필터링할 지표 지정 가능
):
    """
    RSICD 캡션 평가 + failure case JSON 저장
    - 0 점은 기본 실패로 간주, 0을 제외한 하위 q%에도 포함
    - metric 여러 개를 넣으면 OR 조건(합집합)으로 수집
    """
    assert os.path.exists(annotation_file), f"{annotation_file} not found"
    assert os.path.exists(results_file),    f"{results_file} not found"

    # 1) 평가 ---------------------------------------------------------------
    coco, coco_res = COCO(annotation_file), None
    coco_res = coco.loadRes(results_file)
    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.evaluate()

    for m, sc in coco_eval.eval.items():
        print(f"{m}: {sc:.3f}")

    per_img = coco_eval.evalImgs
    if per_img is None:
        print("[Warning] no evalImgs → cannot extract failure cases")
        return coco_eval

    if metrics is None:
        metrics = list(coco_eval.eval.keys())

    # 2) per-metric 점수 배열 + threshold ------------------------------------
    def score_of(info, metric):
        v = info.get(metric, 0.0)
        if isinstance(v, dict):                    # SPICE 등
            v = v.get("All", v.get("all", {}))
            if isinstance(v, dict):
                v = v.get("f", 0.0)
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    scores = {m: np.array([score_of(e, m) for e in per_img]) for m in metrics}

    def nz_percentile(arr, q):
        nz = arr[arr > 0]
        return 0.0 if nz.size == 0 else float(np.percentile(nz, q))

    thresholds = {m: nz_percentile(v, q) for m, v in scores.items()}
    print("\n[Thresholds (0 제외, {}-ptl)]".format(q))
    for m, t in thresholds.items():
        print(f"  {m:<8}: {t:.6g}")

    # 3) 캡션 lookup --------------------------------------------------------
    with open(results_file, "r", encoding="utf-8") as f:
        preds = json.load(f)
    gen_list = preds["annotations"] if isinstance(preds, dict) and "annotations" in preds else preds
    gen_caps = {d["image_id"]: d["caption"] for d in gen_list}

    with open(annotation_file, "r", encoding="utf-8") as f:
        gt_caps = {}
        for a in json.load(f)["annotations"]:
            gt_caps.setdefault(a["image_id"], []).append(a["caption"])

    # 4) 실패 엔트리 수집 (중복 제거) ----------------------------------------
    fail_dict = {}                                     # 🔄 image_id → info
    for info in per_img:
        iid = info["image_id"]
        for m in metrics:
            sc = score_of(info, m)
            if sc == 0 or sc <= thresholds[m]:         # 0 점 or 하위 q%
                fd = fail_dict.setdefault(iid, {
                    "image_id":    iid,
                    "failed_metrics": [],
                    "scores":      {},
                    "gen_caption": gen_caps.get(iid, ""),
                    "gt_captions": gt_caps.get(iid, [])
                })
                fd["failed_metrics"].append(m)
                fd["scores"][m] = sc

    fail_list = list(fail_dict.values())
    print(f"\n[Fail-Dump] {len(fail_list)} entries")

    # 5) 저장 ---------------------------------------------------------------
    out_dir  = Path(base_out) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_f    = out_dir / f"{dataset_name}_failcases_{mask_name}.json"
    with open(out_f, "w", encoding="utf-8") as fp:
        json.dump(fail_list, fp, ensure_ascii=False, indent=2)
    print(f"[Saved] {out_f.resolve()}")
    return coco_eval
