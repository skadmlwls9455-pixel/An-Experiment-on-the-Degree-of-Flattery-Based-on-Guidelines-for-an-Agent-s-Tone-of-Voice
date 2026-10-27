# 파일럿 실행 방법 (Kaggle)

파일럿은 `SPEC.md` §12의 1~5단계(동결 전)만 돌린다. 본실험(H2)은 돌리지 않는다.
Kaggle의 "Save & Run All (Commit)"은 브라우저를 닫아도 서버에서 끝까지 돈다. 그래서 30분~1시간마다 확인하지 않아도 된다.

## 처음 한 번 (약 5분)

1. kaggle.com 로그인 → 계정 설정에서 **휴대폰 인증** (GPU와 인터넷을 쓰려면 필요)
2. Hugging Face에서 `google/gemma-4-E2B-it` 페이지에 들어가 라이선스 동의 버튼이 있으면 누른다
   - 동의가 필요한 모델이면: Hugging Face → Settings → Access Tokens에서 읽기 토큰을 만든다
   - Kaggle 노트북 편집 화면 → **Add-ons → Secrets** → 이름 `HF_TOKEN`, 값에 토큰을 넣고 노트북에 연결한다

## 실행

1. Kaggle → **Create → New Notebook** → **File → Import Notebook** → GitHub 탭에서 이 저장소와 `notebooks/pilot.ipynb` 선택
   (또는 이 파일을 내려받아 업로드)
2. 오른쪽 **Session options**: Accelerator = **GPU T4 x2**, Internet = **On**
3. 오른쪽 위 **Save Version** → **Save & Run All (Commit)** → Save
4. 창을 닫아도 된다. 예상 시간은 1시간 안팎이다 (T4에서 fp32로 돈다)

## 끝나면

1. 노트북 페이지 → 방금 만든 버전 → **Output** 탭 → `pilot_results.zip` 다운로드
2. Claude 대화에 그 zip을 올리면 분석하고 동결안을 만든다

## 실패했을 때

- Output의 `pilot/log.txt` 마지막 줄을 보면 어느 단계에서 멈췄는지 나온다
- 모델 다운로드 403/401: `HF_TOKEN` 시크릿이 연결됐는지, 라이선스 동의를 했는지 확인
- `S0` 오류: transformers 버전이 Gemma 4를 모르면 생긴다. 로그를 그대로 Claude에게 보여 주면 된다

## Colab으로 돌릴 때

같은 노트북을 Colab에서 열면 결과를 Google Drive(`MyDrive/flattery_pilot`)에 저장한다.
무료 Colab은 창을 닫거나 오래 두면 끊긴다. 끊기면 노트북을 다시 실행하면 끝난 생성은 건너뛰고 이어서 돈다.
