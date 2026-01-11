# AWS ML Specialty - 3~4주차 학습 계획

## 목표
- AWS 데이터 엔지니어링 서비스 마스터
- ETL 파이프라인 구축 능력
- 실시간/배치 데이터 처리 이해

---

## 시험 출제 포인트 (Domain 1: Data Engineering - 20%)

| 주제 | 출제 비중 | 핵심 서비스 |
|------|---------|------------|
| 데이터 저장소 | 30% | S3, EFS, FSx |
| 데이터 수집 | 25% | Kinesis, IoT, DMS |
| 데이터 변환 | 30% | Glue, EMR, Lambda |
| 데이터 쿼리 | 15% | Athena, Redshift Spectrum |

---

## 3주차: 데이터 저장 + 수집

### Day 15: S3 for ML - 심화

**학습 목표**: ML 워크로드에 최적화된 S3 사용법

**S3 스토리지 클래스 (시험 필수)**:

| 클래스 | 용도 | 비용 | ML 사용 시기 |
|--------|------|------|-------------|
| **S3 Standard** | 빈번한 접근 | 높음 | 학습 데이터, 활성 데이터 |
| **S3 Intelligent-Tiering** | 접근 패턴 변동 | 중간 | 자동 최적화 필요 시 |
| **S3 Standard-IA** | 비빈번 접근 | 낮음 | 과거 학습 데이터 |
| **S3 Glacier** | 아카이브 | 매우 낮음 | 장기 보관 모델/데이터 |
| **S3 Glacier Deep Archive** | 장기 아카이브 | 최저 | 규정 준수용 보관 |

**S3 성능 최적화**:
```python
# 멀티파트 업로드 (대용량 파일)
import boto3
from boto3.s3.transfer import TransferConfig

s3 = boto3.client('s3')

# 100MB 이상 파일은 멀티파트
config = TransferConfig(
    multipart_threshold=100 * 1024 * 1024,  # 100MB
    max_concurrency=10,
    multipart_chunksize=100 * 1024 * 1024
)

s3.upload_file(
    'large_dataset.csv',
    'my-bucket',
    'data/large_dataset.csv',
    Config=config
)
```

**S3 접두사 설계 (파티셔닝)**:
```
# 좋은 예 - 날짜 기반 파티셔닝
s3://bucket/data/year=2024/month=01/day=15/data.parquet

# 좋은 예 - 해시 접두사 (높은 처리량)
s3://bucket/data/a1b2c3d4/file1.csv
s3://bucket/data/e5f6g7h8/file2.csv

# 나쁜 예 - 순차적 접두사
s3://bucket/data/2024-01-15-001.csv
s3://bucket/data/2024-01-15-002.csv
```

**시험 포인트**:
- S3 Select: S3에서 직접 SQL 쿼리 (데이터 전송량 감소)
- S3 Transfer Acceleration: 글로벌 업로드 가속
- S3 Batch Operations: 대량 객체 작업

---

### Day 16: Amazon Kinesis 개요

**학습 목표**: 실시간 스트리밍 데이터 처리 이해

**Kinesis 서비스 패밀리**:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Amazon Kinesis                               │
├─────────────────┬─────────────────┬─────────────────┬───────────┤
│  Data Streams   │   Firehose      │   Analytics     │   Video   │
│  (실시간 수집)   │  (전송/변환)     │   (실시간 SQL)   │  (비디오)  │
├─────────────────┼─────────────────┼─────────────────┼───────────┤
│ • 샤드 기반     │ • 완전 관리형    │ • SQL 쿼리      │ • 비디오   │
│ • 순서 보장     │ • 자동 스케일링  │ • 실시간 분석   │   스트림   │
│ • 1-365일 보존 │ • S3/Redshift   │ • 이상 탐지     │ • ML 연동  │
│ • 재처리 가능   │ • 변환 가능     │ • 집계/윈도우   │           │
└─────────────────┴─────────────────┴─────────────────┴───────────┘
```

**Kinesis Data Streams vs Firehose (시험 필수)**:

| 특성 | Data Streams | Firehose |
|------|-------------|----------|
| **관리** | 샤드 수동 관리 | 완전 관리형 |
| **지연시간** | 70-200ms | 60초+ (버퍼링) |
| **데이터 보존** | 1-365일 | 없음 (즉시 전송) |
| **재처리** | 가능 | 불가능 |
| **대상** | 커스텀 앱 | S3, Redshift, ES |
| **변환** | 직접 구현 | Lambda 연동 |
| **비용** | 샤드당 과금 | 데이터량 과금 |

**Kinesis Data Streams 코드**:
```python
import boto3
import json

kinesis = boto3.client('kinesis')

# 스트림 생성
kinesis.create_stream(
    StreamName='ml-data-stream',
    ShardCount=2
)

# 데이터 전송 (Producer)
def send_to_kinesis(data, partition_key):
    response = kinesis.put_record(
        StreamName='ml-data-stream',
        Data=json.dumps(data).encode(),
        PartitionKey=partition_key
    )
    return response

# 예시: 거래 데이터 전송
transaction = {
    'transaction_id': 'TX001',
    'amount': 150.00,
    'timestamp': '2024-01-15T10:30:00Z'
}
send_to_kinesis(transaction, 'user123')
```

---

### Day 17: Kinesis Firehose + 변환

**학습 목표**: Firehose로 S3/Redshift 자동 적재

**Firehose 아키텍처**:
```
데이터 소스 → Firehose → (Lambda 변환) → 대상
                ↓
         버퍼링 (크기/시간)
                ↓
         S3 / Redshift / ES / Splunk
```

**Firehose 생성 (콘솔)**:
```
1. Kinesis → Firehose → Create delivery stream
2. Source: Direct PUT or Kinesis Data Streams
3. Transform: Enable Lambda (선택)
4. Destination: S3
5. S3 버퍼: 5MB or 300초 (먼저 충족되는 조건)
6. Compression: GZIP (권장)
7. Encryption: SSE-S3 or SSE-KMS
```

**Lambda 변환 함수**:
```python
import base64
import json

def lambda_handler(event, context):
    output = []

    for record in event['records']:
        # Base64 디코딩
        payload = base64.b64decode(record['data']).decode('utf-8')
        data = json.loads(payload)

        # 데이터 변환 (예: 필드 추가)
        data['processed_at'] = '2024-01-15T10:30:00Z'
        data['source'] = 'firehose'

        # 결과 인코딩
        output_record = {
            'recordId': record['recordId'],
            'result': 'Ok',
            'data': base64.b64encode(
                (json.dumps(data) + '\n').encode()
            ).decode()
        }
        output.append(output_record)

    return {'records': output}
```

**시험 포인트**:
- Firehose는 **최소 60초** 버퍼링 (실시간 아님!)
- Data Streams 없이 **Direct PUT** 가능
- **Parquet/ORC 변환** 지원 (S3 대상)

---

### Day 18: Kinesis Data Analytics

**학습 목표**: 실시간 SQL 분석

**사용 사례**:
- 실시간 대시보드
- 이상 탐지 (RANDOM_CUT_FOREST)
- 실시간 집계/윈도우 분석

**SQL 예시 - 실시간 집계**:
```sql
-- 입력 스트림에서 5분 윈도우 집계
CREATE OR REPLACE STREAM "DESTINATION_SQL_STREAM" (
    ticker_symbol VARCHAR(4),
    avg_price DOUBLE,
    trade_count INTEGER
);

CREATE OR REPLACE PUMP "STREAM_PUMP" AS
INSERT INTO "DESTINATION_SQL_STREAM"
SELECT STREAM
    ticker_symbol,
    AVG(price) AS avg_price,
    COUNT(*) AS trade_count
FROM "SOURCE_SQL_STREAM_001"
GROUP BY
    ticker_symbol,
    STEP("SOURCE_SQL_STREAM_001".ROWTIME BY INTERVAL '5' MINUTE);
```

**이상 탐지 (RANDOM_CUT_FOREST)**:
```sql
-- 실시간 이상 탐지
CREATE OR REPLACE STREAM "ANOMALY_STREAM" (
    anomaly_score DOUBLE,
    transaction_amount DOUBLE
);

CREATE OR REPLACE PUMP "ANOMALY_PUMP" AS
INSERT INTO "ANOMALY_STREAM"
SELECT STREAM
    ANOMALY_SCORE,
    transaction_amount
FROM TABLE(
    RANDOM_CUT_FOREST(
        CURSOR(SELECT STREAM transaction_amount FROM "SOURCE_STREAM"),
        100,   -- numberOfTrees
        256,   -- subSampleSize
        100000 -- timeDecay
    )
)
WHERE ANOMALY_SCORE > 2.0;  -- 임계값
```

---

### Day 19: AWS Glue 개요

**학습 목표**: 서버리스 ETL 이해

**Glue 구성요소**:

```
┌─────────────────────────────────────────────────────────────┐
│                      AWS Glue                                │
├──────────────────┬──────────────────┬───────────────────────┤
│   Data Catalog   │    ETL Jobs      │    Crawlers           │
│   (메타데이터)    │   (변환 작업)     │   (스키마 탐지)        │
├──────────────────┼──────────────────┼───────────────────────┤
│ • 테이블 정의    │ • PySpark/Scala  │ • S3 스캔             │
│ • 스키마 저장    │ • Python Shell   │ • JDBC 스캔           │
│ • Athena 연동   │ • Ray (신규)     │ • 자동 스키마 추론     │
│ • Redshift 연동 │ • 북마크 지원    │ • 파티션 탐지         │
└──────────────────┴──────────────────┴───────────────────────┘
```

**Glue Crawler 설정**:
```
1. Glue Console → Crawlers → Add crawler
2. Data source: S3 path (s3://bucket/data/)
3. IAM role: Glue service role
4. Database: ml_database
5. Schedule: On demand / Hourly / Daily
6. 실행 → 자동으로 테이블 생성
```

**Glue ETL Job (PySpark)**:
```python
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

# Glue Context 초기화
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
job.init(args['JOB_NAME'], args)

# Data Catalog에서 데이터 읽기
datasource = glueContext.create_dynamic_frame.from_catalog(
    database="ml_database",
    table_name="raw_transactions"
)

# 변환 1: 필드 매핑
mapped = ApplyMapping.apply(
    frame=datasource,
    mappings=[
        ("transaction_id", "string", "id", "string"),
        ("amount", "double", "transaction_amount", "double"),
        ("timestamp", "string", "event_time", "timestamp")
    ]
)

# 변환 2: 필터링
filtered = Filter.apply(
    frame=mapped,
    f=lambda x: x["transaction_amount"] > 0
)

# 변환 3: null 제거
dropped_nulls = DropNullFields.apply(frame=filtered)

# S3에 Parquet으로 저장
glueContext.write_dynamic_frame.from_options(
    frame=dropped_nulls,
    connection_type="s3",
    connection_options={
        "path": "s3://bucket/processed/",
        "partitionKeys": ["year", "month"]
    },
    format="parquet"
)

job.commit()
```

---

### Day 20-21: 주말 실습 - 실시간 파이프라인

**프로젝트**: 실시간 거래 데이터 수집 → 변환 → 저장

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Producer │ →  │ Kinesis  │ →  │ Firehose │ →  │    S3    │
│ (Python) │    │ Streams  │    │ +Lambda  │    │ (Parquet)│
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                                      ↓
                               ┌──────────┐
                               │  Glue    │
                               │ Crawler  │
                               └──────────┘
                                      ↓
                               ┌──────────┐
                               │  Athena  │
                               │  Query   │
                               └──────────┘
```

**Producer 코드**:
```python
import boto3
import json
import random
import time
from datetime import datetime

kinesis = boto3.client('kinesis', region_name='us-east-1')

def generate_transaction():
    return {
        'transaction_id': f'TX{random.randint(10000, 99999)}',
        'user_id': f'USER{random.randint(1, 1000)}',
        'amount': round(random.uniform(10, 1000), 2),
        'category': random.choice(['food', 'transport', 'shopping', 'entertainment']),
        'timestamp': datetime.now().isoformat(),
        'is_fraud': random.random() < 0.02  # 2% 이상 거래
    }

# 데이터 스트리밍
while True:
    transaction = generate_transaction()

    kinesis.put_record(
        StreamName='transaction-stream',
        Data=json.dumps(transaction),
        PartitionKey=transaction['user_id']
    )

    print(f"Sent: {transaction['transaction_id']}")
    time.sleep(0.5)  # 초당 2건
```

---

## 4주차: 데이터 변환 + 쿼리

### Day 22: Glue ETL 심화

**학습 목표**: 고급 변환 기법

**DynamicFrame vs DataFrame**:

| 특성 | DynamicFrame | DataFrame |
|------|-------------|-----------|
| **스키마** | 유연 (자동 추론) | 고정 |
| **중첩 데이터** | 자동 처리 | 수동 처리 |
| **성능** | 약간 느림 | 빠름 |
| **변환 함수** | Glue 전용 | Spark SQL |

**상호 변환**:
```python
# DynamicFrame → DataFrame
df = dynamic_frame.toDF()

# DataFrame → DynamicFrame
from awsglue.dynamicframe import DynamicFrame
dynamic_frame = DynamicFrame.fromDF(df, glueContext, "converted")
```

**Glue 북마크 (중복 처리 방지)**:
```python
# Job 파라미터에서 북마크 활성화
# --job-bookmark-option job-bookmark-enable

# 마지막 처리 위치 기억 → 새 데이터만 처리
datasource = glueContext.create_dynamic_frame.from_catalog(
    database="ml_database",
    table_name="raw_data",
    transformation_ctx="datasource"  # 북마크 키
)
```

**Glue Job 유형**:

| 유형 | 용도 | DPU | 비용 |
|------|------|-----|------|
| **Spark** | 대용량 ETL | 2-100 | 높음 |
| **Spark Streaming** | 실시간 ETL | 2-100 | 높음 |
| **Python Shell** | 소규모 작업 | 0.0625-1 | 낮음 |
| **Ray** | ML 워크로드 | 2-100 | 높음 |

---

### Day 23: Amazon Athena

**학습 목표**: S3 데이터 SQL 쿼리

**Athena 특징**:
- **서버리스**: 인프라 관리 불필요
- **비용**: 스캔한 데이터량 기준 ($5/TB)
- **형식**: CSV, JSON, Parquet, ORC, Avro
- **연동**: Glue Data Catalog

**비용 최적화 (시험 필수)**:

| 기법 | 효과 | 방법 |
|------|------|------|
| **Parquet/ORC** | 30-90% 절감 | 컬럼 기반 압축 |
| **파티셔닝** | 대폭 절감 | WHERE 조건 필터 |
| **압축** | 추가 절감 | GZIP, Snappy |
| **LIMIT 사용** | 스캔량 감소 | 결과 제한 |

**Athena 쿼리 예시**:
```sql
-- 테이블 생성 (외부 테이블)
CREATE EXTERNAL TABLE IF NOT EXISTS transactions (
    transaction_id STRING,
    user_id STRING,
    amount DOUBLE,
    category STRING,
    is_fraud BOOLEAN
)
PARTITIONED BY (year STRING, month STRING, day STRING)
STORED AS PARQUET
LOCATION 's3://bucket/processed/transactions/'
TBLPROPERTIES ('parquet.compression'='SNAPPY');

-- 파티션 로드
MSCK REPAIR TABLE transactions;

-- 분석 쿼리
SELECT
    category,
    COUNT(*) as total_transactions,
    SUM(amount) as total_amount,
    AVG(amount) as avg_amount,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as fraud_count
FROM transactions
WHERE year = '2024' AND month = '01'
GROUP BY category
ORDER BY total_amount DESC;
```

**ML 관련 쿼리**:
```sql
-- 특성 엔지니어링용 집계
CREATE TABLE ml_features AS
SELECT
    user_id,
    COUNT(*) as transaction_count,
    AVG(amount) as avg_amount,
    STDDEV(amount) as std_amount,
    MAX(amount) as max_amount,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*) as fraud_rate
FROM transactions
GROUP BY user_id;
```

---

### Day 24: AWS Data Pipeline vs Step Functions

**학습 목표**: 워크플로우 오케스트레이션

**비교**:

| 특성 | Data Pipeline | Step Functions |
|------|--------------|----------------|
| **목적** | 데이터 이동/변환 | 범용 워크플로우 |
| **EC2 의존** | 필요 | 불필요 |
| **복잡도** | 단순 | 복잡한 분기 가능 |
| **상태** | 레거시 | 권장 |
| **통합** | EMR, Redshift | 모든 AWS 서비스 |

**Step Functions + SageMaker**:
```json
{
  "Comment": "ML Pipeline",
  "StartAt": "Preprocessing",
  "States": {
    "Preprocessing": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:createProcessingJob.sync",
      "Parameters": {
        "ProcessingJobName.$": "$.preprocessing_job_name",
        "ProcessingResources": {
          "ClusterConfig": {
            "InstanceCount": 1,
            "InstanceType": "ml.m5.xlarge",
            "VolumeSizeInGB": 30
          }
        }
      },
      "Next": "Training"
    },
    "Training": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:createTrainingJob.sync",
      "Parameters": {
        "TrainingJobName.$": "$.training_job_name",
        "AlgorithmSpecification": {
          "TrainingImage": "..."
        }
      },
      "Next": "Evaluation"
    },
    "Evaluation": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:invoke",
      "Parameters": {
        "FunctionName": "evaluate-model"
      },
      "Next": "DeployDecision"
    },
    "DeployDecision": {
      "Type": "Choice",
      "Choices": [
        {
          "Variable": "$.accuracy",
          "NumericGreaterThan": 0.9,
          "Next": "DeployModel"
        }
      ],
      "Default": "NotifyFailure"
    },
    "DeployModel": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:createEndpoint",
      "End": true
    },
    "NotifyFailure": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sns:publish",
      "End": true
    }
  }
}
```

---

### Day 25: EMR for ML

**학습 목표**: 대규모 데이터 처리

**EMR 사용 시기**:
- 데이터 > 수십 TB
- 복잡한 Spark ML 파이프라인
- 커스텀 라이브러리 필요
- Glue 비용 > EMR 비용

**EMR + SageMaker 연동**:
```python
# EMR에서 SageMaker 학습 트리거
from sagemaker import Session
from sagemaker.estimator import Estimator

# Spark에서 전처리 완료 후
processed_data_path = "s3://bucket/processed/"

# SageMaker 학습 시작
estimator = Estimator(
    image_uri=container,
    role=role,
    instance_count=2,
    instance_type='ml.m5.xlarge'
)

estimator.fit({'train': processed_data_path})
```

**EMR Serverless (신규)**:
- 클러스터 관리 불필요
- 자동 스케일링
- 초 단위 과금
- Spark/Hive 지원

---

### Day 26: 데이터 보안 + 암호화

**학습 목표**: ML 데이터 보안 Best Practice

**암호화 옵션**:

| 서비스 | 저장 시 암호화 | 전송 중 암호화 |
|--------|-------------|--------------|
| **S3** | SSE-S3, SSE-KMS, SSE-C | HTTPS |
| **Kinesis** | SSE-KMS | HTTPS |
| **Glue** | SSE-S3, SSE-KMS | HTTPS |
| **SageMaker** | KMS | HTTPS, VPC |

**SageMaker 네트워크 보안**:
```python
from sagemaker.estimator import Estimator

estimator = Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',

    # VPC 설정
    subnets=['subnet-xxx'],
    security_group_ids=['sg-xxx'],

    # 암호화
    output_kms_key='arn:aws:kms:...',
    volume_kms_key='arn:aws:kms:...',

    # 네트워크 격리
    enable_network_isolation=True
)
```

**시험 포인트**:
- **VPC Endpoint**: S3, SageMaker API 접근
- **PrivateLink**: 인터넷 없이 서비스 접근
- **Network Isolation**: 컨테이너 아웃바운드 차단

---

### Day 27-28: 주말 종합 실습

**프로젝트**: 완전한 데이터 파이프라인

```
┌──────────────────────────────────────────────────────────────────┐
│                    End-to-End Data Pipeline                       │
└──────────────────────────────────────────────────────────────────┘

  Raw Data (S3)
       │
       ▼
  ┌─────────┐     ┌─────────┐     ┌─────────┐
  │  Glue   │ ──▶ │  Glue   │ ──▶ │   S3    │
  │ Crawler │     │   ETL   │     │(Parquet)│
  └─────────┘     └─────────┘     └─────────┘
                                       │
       ┌───────────────────────────────┤
       │                               │
       ▼                               ▼
  ┌─────────┐                    ┌──────────┐
  │ Athena  │                    │SageMaker │
  │  Query  │                    │ Training │
  └─────────┘                    └──────────┘
```

**체크리스트**:
- [ ] S3 버킷 생성 (raw, processed, models)
- [ ] Glue Crawler로 스키마 탐지
- [ ] Glue ETL Job 작성 (PySpark)
- [ ] Parquet 변환 + 파티셔닝
- [ ] Athena 테이블 생성 + 쿼리
- [ ] SageMaker Training Job 연동
- [ ] Step Functions 워크플로우 생성

---

## 3-4주차 퀴즈 (자가 점검)

### Q1. Kinesis Data Streams vs Firehose 선택 기준은?
<details>
<summary>정답 보기</summary>

- **Data Streams**: 실시간 처리 필요, 재처리 필요, 커스텀 앱 개발
- **Firehose**: S3/Redshift 자동 적재, 관리 최소화, 60초 지연 허용
</details>

### Q2. Athena 비용을 줄이는 방법 3가지는?
<details>
<summary>정답 보기</summary>

1. **Parquet/ORC** 컬럼 형식 사용
2. **파티셔닝**으로 스캔 범위 제한
3. **압축** (GZIP, Snappy) 적용
</details>

### Q3. Glue 북마크(Bookmark)의 용도는?
<details>
<summary>정답 보기</summary>

ETL Job이 마지막으로 처리한 위치를 기억하여, 다음 실행 시 새로운 데이터만 처리. 중복 처리 방지.
</details>

### Q4. SageMaker에서 Network Isolation을 활성화하면?
<details>
<summary>정답 보기</summary>

학습/추론 컨테이너의 모든 아웃바운드 네트워크 접근 차단. S3 데이터는 VPC 엔드포인트로만 접근.
</details>

### Q5. Glue ETL에서 DynamicFrame의 장점은?
<details>
<summary>정답 보기</summary>

- 스키마 자동 추론 (유연함)
- 중첩 데이터 자동 처리
- 데이터 불일치 처리 용이
</details>

---

## 핵심 서비스 요약표

| 서비스 | 용도 | 시험 빈도 | 핵심 포인트 |
|--------|------|---------|------------|
| **S3** | 저장소 | ★★★★★ | 스토리지 클래스, 파티셔닝 |
| **Kinesis Streams** | 실시간 수집 | ★★★★★ | 샤드, 순서 보장, 재처리 |
| **Kinesis Firehose** | 자동 적재 | ★★★★ | 60초 버퍼, 변환, S3/Redshift |
| **Kinesis Analytics** | 실시간 SQL | ★★★ | RANDOM_CUT_FOREST |
| **Glue** | ETL | ★★★★★ | Crawler, 북마크, DynamicFrame |
| **Athena** | S3 쿼리 | ★★★★ | 파티셔닝, Parquet, 비용 |
| **EMR** | 대규모 처리 | ★★★ | Spark, 클러스터 |
| **Step Functions** | 워크플로우 | ★★★★ | ML 파이프라인 오케스트레이션 |

---

## 다음 주 예고 (5-6주차)

- 탐색적 데이터 분석 (EDA)
- 특성 엔지니어링
- 데이터 시각화
- SageMaker Data Wrangler
