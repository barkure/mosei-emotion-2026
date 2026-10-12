# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：DeepSeek，型号：DeepSeek V4.1 Flash，开发公司：杭州深度求索人工智能基础技术研究有限公司，版本颁布日期：2026-09-10

"""Apply archived content reviews without changing predictions or candidate times."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/q3_explanation/q3'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    text_path = OUT / 'text_time_user_review_20260925.json'
    video_path = OUT / 'video05_user_review_20260925.json'
    texts = json.loads(text_path.read_text(encoding="utf-8"))
    video = json.loads(video_path.read_text(encoding="utf-8"))
    csv_path = OUT / 'attachment4_predictions_explanations.csv'
    full_path = OUT / 'attachment4_explanations_full.json'
    rows = read_csv(csv_path)
    details = json.loads(full_path.read_text(encoding="utf-8"))
    lookup = {r['sample_id']: r for r in rows}
    full = {r['sample_id']: r for r in details}
    assert len(lookup) == len(full) == 20
    for item in texts['samples']:
        sid = item['sample_id']
        row = lookup[sid]
        expected = item['candidate_interval_seconds']
        assert all(abs(float(row[k])-v) < 1e-6 for k, v in
                   zip(('time_start_seconds', 'time_end_seconds'), expected)), sid
        assert full[sid]['evidence_time_seconds'] == expected, sid
        assert row['key_text'] == item['text'], sid
        row['time_status'] = texts['status']
        full[sid]['time_status'] = texts['status']
        full[sid]['human_review'] = dict(status=texts['status'], scope=texts['scope'],
                                        source=str(text_path.relative_to(ROOT)))
    row = lookup[video['sample_id']]
    assert abs(float(row['frame_candidate_seconds']) - video['frame_seconds']) < .001
    row['frame_candidate_status'] = video['status']
    full[video['sample_id']]['frame_candidate']['status'] = video['status']
    full[video['sample_id']]['human_review'] = dict(status=video['status'], scope=video['scope'],
        feature_to_frame_mapping=video['feature_to_frame_mapping'], source=str(video_path.relative_to(ROOT)))
    review_path = OUT / 'evidence_review.csv'
    reviews = read_csv(review_path)
    for r in reviews:
        sid = r['sample_id']
        if sid in {x['sample_id'] for x in texts['samples']}:
            r['review_status'] = texts['status']
        elif sid == video['sample_id']:
            r['review_status'] = video['status']
    write_csv(csv_path, rows)
    write_csv(review_path, reviews)
    full_path.write_text(json.dumps(details, ensure_ascii=False, indent=2)+'\n', encoding="utf-8")
    path = OUT / 'attachment4_manifest.json'
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        details_sha256=hashlib.sha256(full_path.read_bytes()).hexdigest(),
        time_status='7 text intervals reviewed for approximate content coverage; 05 nearby frame reviewed',
        n_candidate_text_times=sum(bool(r['time_start_seconds']) for r in rows),
        n_reviewed_text_intervals=len(texts['samples']), n_reviewed_nearby_video_frames=1,
        human_review_sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (text_path, video_path)})
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding="utf-8")
    print('Merged 7 text reviews and 1 nearby-frame review; predictions and times preserved.')


if __name__ == '__main__':
    main()
