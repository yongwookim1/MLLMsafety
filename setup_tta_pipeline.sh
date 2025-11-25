#!/bin/bash
# TTA 데이터셋 파이프라인 설정 및 실행 스크립트
# 방화벽 환경에서 수동 다운로드 후 서버에서 실행

set -e

echo "🚀 TTA 데이터셋 멀티모달 증강 파이프라인 설정"
echo "==============================================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 현재 디렉토리가 프로젝트 루트인지 확인
if [ ! -f "requirements.txt" ] || [ ! -f "scripts/run_tta_pipeline.py" ]; then
    echo -e "${RED}❌ 오류: 이 스크립트는 MLLMsafety 프로젝트 루트 디렉토리에서 실행해야 합니다.${NC}"
    exit 1
fi

echo -e "${YELLOW}📋 필수 파일 확인 중...${NC}"

# TTA 데이터셋 확인 (코어 파일 + 이미지 파일)
if [ ! -d "data_cache/TTA01_AssurAI" ] || [ ! -f "data_cache/TTA01_AssurAI/data-00000-of-00001.arrow" ]; then
    echo -e "${RED}❌ TTA 데이터셋이 없습니다. 다음 단계를 따르세요:${NC}"
    echo "   1. 인터넷이 가능한 컴퓨터에서 다음을 실행:"
    echo "      python scripts/manual_download_tta.py  # ~230개 이미지 파일 포함"
    echo "   2. 다운로드된 data_cache/TTA01_AssurAI 폴더를 이 서버로 전송"
    exit 1
else
    # 이미지 파일 수 확인
    image_count=$(find data_cache/TTA01_AssurAI -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.webp" \) | wc -l)
    if [ "$image_count" -gt 0 ]; then
        echo -e "${GREEN}✅ TTA 데이터셋 확인됨 (${image_count}개 이미지 파일 포함)${NC}"
    else
        echo -e "${YELLOW}⚠️  TTA 데이터셋 확인됨 (이미지 파일 없음 - 재다운로드 권장)${NC}"
    fi
fi

# 모델 확인
if [ ! -d "models_cache/qwen-image" ] || [ ! -d "models_cache/qwen2.5-vl-7b-instruct" ]; then
    echo -e "${RED}❌ 모델이 없습니다. 다음을 실행하세요:${NC}"
    echo "   python scripts/download_models.py"
    exit 1
else
    echo -e "${GREEN}✅ 모델 파일 확인됨${NC}"
fi

# 의존성 설치 확인
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 가상환경 생성 및 의존성 설치...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    echo -e "${GREEN}✅ 가상환경 확인됨${NC}"
    source venv/bin/activate
fi

echo -e "${YELLOW}🔧 TTA 파이프라인 실행 준비...${NC}"

# 기존 결과 정리 (선택사항)
read -p "기존 TTA 결과를 정리하시겠습니까? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}🧹 기존 결과 정리 중...${NC}"
    rm -rf outputs/tta_images outputs/tta_results outputs/tta_image_mapping.json
    echo -e "${GREEN}✅ 정리 완료${NC}"
fi

echo -e "${GREEN}🎯 TTA 파이프라인 실행${NC}"
echo "=============================="

# 파이프라인 실행
if [ "$1" = "--test" ]; then
    echo "테스트 모드로 실행 (샘플 제한)"
    python scripts/run_tta_pipeline.py --limit 5
else
    python scripts/run_tta_pipeline.py
fi

echo -e "${GREEN}🎉 TTA 파이프라인 실행 완료!${NC}"
echo ""
echo "📁 결과 파일 위치:"
echo "   - 생성된 이미지: outputs/tta_images/"
echo "   - 평가 결과: outputs/tta_results/evaluation_results.json"
echo "   - 이미지 매핑: outputs/tta_image_mapping.json"
