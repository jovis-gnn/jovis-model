# Text Model

개인화 추천 진행 시 상품명을 이용하기 위한 각종 모델 및 관련 코드를 관리하기 위한 디텍토리입니다.

## Usage

Config에 공통적으로 `use_pooler_output`이라는 인자가 있는데 언어모델로부터 추출한 텍스트 representation이 256이냐 768이냐 차이임 (`use_pooler_output == True`이면 256).

### Inference

- Top-$k$ inference를 위해서는 `inference.py` 실행.
- `args.mode == "topk"`인 것을 확인.
- Config file: `configs/prod_name_topk_config.json`.

### Get Embedding

- 상품명을 입력값으로 받았을 때 가장 비슷한 임베딩 추출하기 위해서는 `get_embedding.py` 실행.
- `args.mode == "embedding"`인 것을 확인.
- Config file: `configs/prod_name_embedding_config.json`

### Create Embeddings

- 텍스트를 이용해 임베딩을 미리 추출하기 위한 스크립트.
- Config file: `configs/prod_name_create_embeddings_config.json`
