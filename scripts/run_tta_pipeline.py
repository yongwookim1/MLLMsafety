#!/usr/bin/env python3
"""
TTA 데이터셋 파이프라인 실행 스크립트

이 스크립트는 다음 작업을 수행합니다:
1. 텍스트 샘플을 이미지로 변환하여 증강
2. 멀티모달 샘플을 그대로 유지
3. 증강된 데이터셋으로 평가 실행
"""

import os
import sys
import argparse
import json
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_requirements():
    """필수 파일들이 있는지 확인"""
    required_files = [
        "data_cache/TTA01_AssurAI/data-00000-of-00001.arrow",
        "models_cache/qwen-image",
        "models_cache/qwen2.5-vl-7b-instruct"
    ]

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print("❌ 다음 파일들이 누락되었습니다:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\n먼저 다음 명령어들을 실행하세요:")
        print("1. python scripts/manual_download_tta.py  # TTA 데이터셋 다운로드")
        print("2. python scripts/download_models.py       # 모델 다운로드")
        return False

    print("✅ 모든 필수 파일이 존재합니다.")
    return True

def run_data_preparation():
    """1단계: 텍스트 데이터를 이미지로 변환"""
    print("\n" + "="*50)
    print("📝 1단계: 텍스트 데이터를 이미지로 변환")
    print("="*50)

    from scripts.prepare_tta_data import process_tta_dataset

    try:
        process_tta_dataset()
        print("✅ 데이터 준비 완료")
        return True
    except Exception as e:
        print(f"❌ 데이터 준비 실패: {e}")
        return False

def run_evaluation(limit=None):
    """2단계: 평가 실행"""
    print("\n" + "="*50)
    print("🔍 2단계: 멀티모달 안전성 평가")
    print("="*50)

    from scripts.run_tta_evaluation import main as run_eval

    # 커맨드 라인 인자 설정
    if limit:
        sys.argv = ['run_tta_evaluation.py', '--limit', str(limit)]
    else:
        sys.argv = ['run_tta_evaluation.py']

    try:
        run_eval()
        print("✅ 평가 완료")
        return True
    except Exception as e:
        print(f"❌ 평가 실패: {e}")
        return False

def generate_summary():
    """결과 요약 생성"""
    print("\n" + "="*50)
    print("📊 결과 요약")
    print("="*50)

    results_file = "outputs/tta_results/evaluation_results.json"
    mapping_file = "outputs/tta_image_mapping.json"

    if os.path.exists(results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

        print(f"📈 평가된 샘플 수: {len(results)}")

        # 위험 점수별 통계
        scores = []
        for result in results:
            score = result.get('judge_result', {}).get('parsed_score')
            if score is not None:
                scores.append(score)

        if scores:
            avg_score = sum(scores) / len(scores)
            print(".2f"
            # 점수 분포
            score_counts = {}
            for score in scores:
                score_counts[score] = score_counts.get(score, 0) + 1

            print("📊 점수 분포:")
            for score in sorted(score_counts.keys()):
                count = score_counts[score]
                percentage = (count / len(scores)) * 100
                print(".1f"
        else:
            print("⚠️  파싱된 점수가 없습니다.")

    if os.path.exists(mapping_file):
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping = json.load(f)

        print(f"🖼️  생성된 이미지 수: {len(mapping)}")

    print(f"\n📁 결과 파일 위치:")
    print(f"  - 평가 결과: {results_file}")
    print(f"  - 이미지 매핑: {mapping_file}")
    print(f"  - 생성된 이미지: outputs/tta_images/")

def main():
    parser = argparse.ArgumentParser(description="TTA 데이터셋 파이프라인 실행")
    parser.add_argument("--skip-preparation", action="store_true",
                       help="데이터 준비 단계 건너뛰기")
    parser.add_argument("--skip-evaluation", action="store_true",
                       help="평가 단계 건너뛰기")
    parser.add_argument("--limit", type=int,
                       help="평가할 최대 샘플 수 (테스트용)")
    parser.add_argument("--check-only", action="store_true",
                       help="요구사항만 확인하고 실행하지 않음")

    args = parser.parse_args()

    print("🚀 TTA 데이터셋 멀티모달 증강 및 평가 파이프라인")
    print("="*60)

    # 요구사항 확인
    if not check_requirements():
        return 1

    if args.check_only:
        print("✅ 모든 확인 완료. 파이프라인을 실행할 준비가 되었습니다.")
        return 0

    success = True

    # 1단계: 데이터 준비
    if not args.skip_preparation:
        if not run_data_preparation():
            success = False
    else:
        print("⏭️  데이터 준비 단계 건너뜀")

    # 2단계: 평가
    if not args.skip_evaluation and success:
        if not run_evaluation(args.limit):
            success = False
    else:
        print("⏭️  평가 단계 건너뜀")

    # 결과 요약
    if success:
        generate_summary()
        print("\n🎉 파이프라인 실행 완료!")
    else:
        print("\n❌ 파이프라인 실행 실패")

    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
