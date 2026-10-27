# 에이전트 말투와 아첨: 하십시오체 방향 개입 실험

Gemma 4 E2B IT에서 하십시오체를 늘리는 활성 방향을 L27에 더했을 때, 근거 없는 반박 뒤의 정답 번복이 반박 없는 재질문 뒤보다 더 늘어나는지를 같은 문항끼리 비교한다.

- 실험 명세: [`SPEC.md`](SPEC.md) (v0.2.1, 동결 전)
- 파일럿 실행 방법: [`RUN_KAGGLE.md`](RUN_KAGGLE.md)

## 구조

```
SPEC.md                 실험 명세
RULE.md, freeze.json    동결 시 생성 (아직 없음)
notebooks/pilot.ipynb   Kaggle/Colab 파일럿 노트북 (동결 전 단계만)
src/
  config.py             고정 설정과 임시 기준값
  template.py           채팅 템플릿 렌더링 (thinking 꺼짐)
  model.py              모델 로딩, 층 출력 주입·캡처, 생성
  textmetrics.py        답 줄 파싱, 화계·길이·한글 비율
  readouts.py           M2 로짓 판독, M3 유발 변화
  kmmlu.py              KMMLU 로딩과 문항 분할
  stages.py             파일럿 S0~S7
  run_pilot.py          파일럿 실행 진입점
  freeze.py             동결 파일 생성·검사
  e2_main.py            본실험 H2 (동결 검사 통과 시에만 실행)
tests/test_pipeline.py  오프라인 테스트 (작은 무작위 Gemma 4 + 각본 응답)
third_party/neuron-map/ 원 저장소에서 쓴 파일 사본, 라이선스, SHA256
```

## 테스트

```
pip install torch transformers tokenizers numpy
python tests/test_pipeline.py
```

모델을 내려받지 않고 CPU에서 돈다. 주입·캡처·KV 공유 국소성·`layer_scalar` 보정·재시작·눈가림·동결 검사를 확인한다.

## 출처

`third_party/neuron-map/`의 데이터와 벡터는 Ryou Jiho, *Gemma 4 E2B neuron map* (`salixkoreensis94-spec/gemma4-e2b-neuron-map`, CC BY 4.0), 분류기 코드는 같은 저장소의 Apache-2.0 코드다. Gemma는 Google LLC의 상표다.
