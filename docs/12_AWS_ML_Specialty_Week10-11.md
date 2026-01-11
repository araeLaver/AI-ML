# AWS ML Specialty - 10~11주차 학습 계획

## 목표
- 모델 배포 전략 완벽 마스터
- SageMaker MLOps 파이프라인 구축
- 모니터링, 보안, 비용 최적화

---

## 시험 출제 포인트 (Domain 4: ML Implementation - 20%)

| 주제 | 출제 비중 | 핵심 내용 |
|------|---------|----------|
| 모델 배포 | 35% | Endpoint, Batch, Serverless |
| MLOps | 30% | Pipelines, Model Registry |
| 모니터링 | 20% | Model Monitor, Clarify |
| 보안 | 15% | IAM, VPC, 암호화 |

---

## 10주차: 모델 배포 전략

### Day 62: 배포 옵션 개요

**학습 목표**: SageMaker 배포 옵션 완벽 이해

**배포 옵션 비교 (시험 필수)**:

| 옵션 | 지연시간 | 비용 | 사용 사례 |
|------|---------|------|---------|
| **Real-time Endpoint** | ms | 상시 과금 | 실시간 추론 |
| **Serverless Inference** | 초 | 요청당 | 간헐적 트래픽 |
| **Batch Transform** | 분~시간 | 처리량 | 대량 배치 |
| **Async Inference** | 초~분 | 요청당 | 긴 처리 시간 |

**의사결정 트리**:
```
실시간 응답 필요?
│
├── Yes → 지연시간 < 1초?
│         ├── Yes → Real-time Endpoint
│         └── No  → Async Inference
│
└── No  → 트래픽 예측 가능?
          ├── Yes, 상시 → Real-time Endpoint
          ├── Yes, 간헐 → Serverless Inference
          └── No, 배치 → Batch Transform
```

---

### Day 63: Real-time Endpoint 심화

**학습 목표**: 실시간 추론 엔드포인트 최적화

**엔드포인트 생성**:
```python
from sagemaker.model import Model
from sagemaker.predictor import Predictor

# 모델 생성
model = Model(
    image_uri=container,
    model_data=f's3://{bucket}/models/model.tar.gz',
    role=role,
    name='fraud-detection-model'
)

# 엔드포인트 배포
predictor = model.deploy(
    initial_instance_count=2,
    instance_type='ml.m5.large',
    endpoint_name='fraud-detection-endpoint'
)

# 추론
result = predictor.predict(data)
```

**Production Variants (A/B 테스트)**:
```python
from sagemaker.model import Model
from sagemaker import ProductionVariant

# 모델 A (기존)
model_a = Model(image_uri=container, model_data=model_a_data, role=role)

# 모델 B (신규)
model_b = Model(image_uri=container, model_data=model_b_data, role=role)

# Production Variants 정의
variant_a = ProductionVariant(
    model_name='model-a',
    instance_type='ml.m5.large',
    initial_instance_count=2,
    initial_weight=90  # 90% 트래픽
)

variant_b = ProductionVariant(
    model_name='model-b',
    instance_type='ml.m5.large',
    initial_instance_count=1,
    initial_weight=10  # 10% 트래픽
)

# 엔드포인트 생성
endpoint = Endpoint.create(
    endpoint_name='ab-test-endpoint',
    production_variants=[variant_a, variant_b]
)
```

**Auto Scaling**:
```python
import boto3

asg = boto3.client('application-autoscaling')

# 스케일링 타겟 등록
asg.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId='endpoint/fraud-detection-endpoint/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=1,
    MaxCapacity=10
)

# 스케일링 정책 (Target Tracking)
asg.put_scaling_policy(
    PolicyName='Invocations-ScalingPolicy',
    ServiceNamespace='sagemaker',
    ResourceId='endpoint/fraud-detection-endpoint/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 1000,  # 인스턴스당 1000 요청/분
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
        },
        'ScaleInCooldown': 300,
        'ScaleOutCooldown': 60
    }
)
```

**시험 포인트**:
- Auto Scaling 메트릭: `SageMakerVariantInvocationsPerInstance`
- Scale Out: 빠르게 (60초)
- Scale In: 천천히 (300초)

---

### Day 64: Serverless Inference

**학습 목표**: 서버리스 추론 활용

**Serverless Inference 특징**:
- 자동 스케일링 (0까지)
- 요청당 과금
- Cold Start 존재 (수초)
- 최대 6MB 페이로드

**Serverless 배포**:
```python
from sagemaker.serverless import ServerlessInferenceConfig

# Serverless 설정
serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=2048,      # 1024, 2048, 3072, 4096, 5120, 6144
    max_concurrency=20,           # 최대 동시 요청
    provisioned_concurrency=2     # Warm 인스턴스 (선택)
)

# 배포
predictor = model.deploy(
    serverless_inference_config=serverless_config,
    endpoint_name='serverless-endpoint'
)
```

**Serverless vs Real-time 선택**:

| 상황 | 선택 |
|------|------|
| 트래픽 예측 불가 | Serverless |
| 간헐적 요청 | Serverless |
| 비용 최적화 우선 | Serverless |
| 밀리초 지연시간 필요 | Real-time |
| 상시 트래픽 | Real-time |
| Cold Start 불가 | Real-time + Provisioned |

---

### Day 65: Batch Transform + Async Inference

**학습 목표**: 배치 및 비동기 추론

**Batch Transform**:
```python
from sagemaker.transformer import Transformer

transformer = model.transformer(
    instance_count=2,
    instance_type='ml.m5.xlarge',
    strategy='MultiRecord',        # 또는 'SingleRecord'
    assemble_with='Line',
    output_path=f's3://{bucket}/output/predictions/'
)

# 배치 추론 실행
transformer.transform(
    data=f's3://{bucket}/data/batch-input/',
    content_type='text/csv',
    split_type='Line',
    join_source='Input'  # 입력과 출력 조인
)

transformer.wait()
```

**Async Inference**:
```python
from sagemaker.async_inference import AsyncInferenceConfig

# Async 설정
async_config = AsyncInferenceConfig(
    output_path=f's3://{bucket}/async-output/',
    max_concurrent_invocations_per_instance=4,
    notification_config={
        'SuccessTopic': 'arn:aws:sns:...:success-topic',
        'ErrorTopic': 'arn:aws:sns:...:error-topic'
    }
)

# 배포
predictor = model.deploy(
    instance_type='ml.m5.large',
    initial_instance_count=1,
    async_inference_config=async_config
)

# 비동기 호출
response = predictor.predict_async(
    data=large_payload,
    input_path='s3://bucket/input/data.csv'
)

# 결과 위치
output_location = response.output_path
```

**Async Inference 사용 사례**:
- 대용량 페이로드 (최대 1GB)
- 긴 추론 시간 (최대 15분)
- 비동기 처리 허용
- SNS 알림 필요

---

### Day 66: Multi-Model Endpoint (MME)

**학습 목표**: 단일 엔드포인트에 다중 모델

**MME 특징**:
- 수천 개 모델 호스팅
- 동적 모델 로딩
- 비용 효율적
- GPU 지원

```python
from sagemaker.multidatamodel import MultiDataModel

# Multi-Model 생성
mme = MultiDataModel(
    name='multi-model-endpoint',
    model_data_prefix=f's3://{bucket}/models/',
    image_uri=container,
    role=role
)

# 배포
predictor = mme.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge',
    endpoint_name='mme-endpoint'
)

# 모델 추가
mme.add_model(
    model_data_source=f's3://{bucket}/models/model-a.tar.gz',
    model_data_path='model-a.tar.gz'
)

# 특정 모델로 추론
predictor.predict(
    data=payload,
    target_model='model-a.tar.gz'  # 호출할 모델 지정
)
```

**MME vs 개별 Endpoint**:

| 상황 | 선택 |
|------|------|
| 모델 수 많음 (100+) | MME |
| 비용 최적화 | MME |
| 모델별 독립 스케일링 | 개별 Endpoint |
| 모델 격리 필요 | 개별 Endpoint |
| 유사 모델 (같은 프레임워크) | MME |

---

### Day 67: 추론 최적화

**학습 목표**: 추론 성능 및 비용 최적화

**Neo Compilation (모델 최적화)**:
```python
from sagemaker.neo import NeoCompiler

# 모델 컴파일
compiler = NeoCompiler(
    role=role,
    input_config={
        'S3Uri': f's3://{bucket}/models/model.tar.gz',
        'Framework': 'XGBOOST',
        'DataInputConfig': '{"input": [1, 10]}'
    },
    output_config={
        'S3OutputLocation': f's3://{bucket}/compiled-models/',
        'TargetInstance': 'ml_m5'  # 대상 인스턴스
    }
)

compiler.compile()
```

**Inference Recommender**:
```python
from sagemaker.inference_recommender import InferenceRecommender

recommender = InferenceRecommender(
    role=role,
    model_package_version_arn='arn:aws:sagemaker:...:model-package/...',
    job_name='inference-recommender-job'
)

# 기본 추천 (인스턴스 타입)
recommender.run_default_job()

# 고급 추천 (상세 벤치마크)
recommender.run_advanced_job(
    traffic_pattern='PHASES',
    phases=[
        {'InitialNumberOfUsers': 1, 'SpawnRate': 1, 'DurationInSeconds': 120}
    ]
)
```

**최적화 기법 요약**:

| 기법 | 효과 | 적용 |
|------|------|------|
| **Neo** | 추론 속도 2-3배 | 엣지, 특정 인스턴스 |
| **Elastic Inference** | GPU 비용 75% 절감 | GPU 부분 활용 |
| **Graviton** | 비용 40% 절감 | ARM 기반 |
| **Spot Instance** | 비용 90% 절감 | 학습 (추론 X) |
| **Model Pruning** | 모델 크기 축소 | 딥러닝 |
| **Quantization** | 메모리/속도 개선 | 딥러닝 |

---

### Day 68-69: 주말 실습 - 배포 파이프라인

**프로젝트**: 완전한 배포 파이프라인

**체크리스트**:
- [ ] 모델 학습 및 저장
- [ ] Real-time Endpoint 배포
- [ ] Auto Scaling 설정
- [ ] A/B 테스트 구성
- [ ] Batch Transform 실행
- [ ] Serverless Endpoint 배포
- [ ] 성능 비교
- [ ] 리소스 정리

---

## 11주차: MLOps + 모니터링 + 보안

### Day 70: SageMaker Pipelines

**학습 목표**: ML 파이프라인 자동화

**Pipeline 구성요소**:
```
┌─────────────────────────────────────────────────────────────┐
│                    SageMaker Pipelines                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐     │
│  │Processing│ → │Training │ → │  Eval   │ → │ Deploy  │     │
│  │  Step   │   │  Step   │   │  Step   │   │  Step   │     │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘     │
│                                    │                        │
│                              ┌─────┴─────┐                  │
│                              │ Condition │                  │
│                              │   Step    │                  │
│                              └───────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

**Pipeline 코드**:
```python
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CreateModelStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.parameters import ParameterString, ParameterFloat

# 파라미터 정의
processing_instance_type = ParameterString(
    name='ProcessingInstanceType',
    default_value='ml.m5.xlarge'
)
model_approval_status = ParameterString(
    name='ModelApprovalStatus',
    default_value='PendingManualApproval'
)
accuracy_threshold = ParameterFloat(
    name='AccuracyThreshold',
    default_value=0.8
)

# Step 1: 전처리
processing_step = ProcessingStep(
    name='PreprocessData',
    processor=sklearn_processor,
    inputs=[...],
    outputs=[...],
    code='preprocess.py'
)

# Step 2: 학습
training_step = TrainingStep(
    name='TrainModel',
    estimator=xgb_estimator,
    inputs={
        'train': TrainingInput(s3_data=processing_step.properties.ProcessingOutputConfig.Outputs['train'].S3Output.S3Uri)
    }
)

# Step 3: 평가
evaluation_step = ProcessingStep(
    name='EvaluateModel',
    processor=sklearn_processor,
    inputs=[
        ProcessingInput(source=training_step.properties.ModelArtifacts.S3ModelArtifacts, destination='/opt/ml/processing/model'),
        ProcessingInput(source=processing_step.properties.ProcessingOutputConfig.Outputs['test'].S3Output.S3Uri, destination='/opt/ml/processing/test')
    ],
    outputs=[ProcessingOutput(source='/opt/ml/processing/evaluation', destination=f's3://{bucket}/evaluation/')],
    code='evaluate.py'
)

# Step 4: 조건부 배포
condition = ConditionGreaterThanOrEqualTo(
    left=JsonGet(step_name='EvaluateModel', property_file='evaluation.json', json_path='metrics.accuracy'),
    right=accuracy_threshold
)

# 모델 등록
register_step = RegisterModel(
    name='RegisterModel',
    estimator=xgb_estimator,
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=['text/csv'],
    response_types=['text/csv'],
    inference_instances=['ml.m5.large'],
    transform_instances=['ml.m5.large'],
    model_package_group_name='fraud-detection-models',
    approval_status=model_approval_status
)

# 조건 Step
condition_step = ConditionStep(
    name='CheckAccuracy',
    conditions=[condition],
    if_steps=[register_step],
    else_steps=[]
)

# Pipeline 생성
pipeline = Pipeline(
    name='fraud-detection-pipeline',
    parameters=[processing_instance_type, model_approval_status, accuracy_threshold],
    steps=[processing_step, training_step, evaluation_step, condition_step]
)

# Pipeline 실행
pipeline.upsert(role_arn=role)
execution = pipeline.start()
execution.wait()
```

---

### Day 71: Model Registry

**학습 목표**: 모델 버전 관리

**Model Registry 개념**:
```
Model Package Group
├── Model Package v1 (Approved)
│   ├── Model Artifacts (S3)
│   ├── Inference Image
│   └── Metadata
├── Model Package v2 (Rejected)
└── Model Package v3 (PendingManualApproval)
```

**Model Registry 코드**:
```python
from sagemaker.model import ModelPackage

# 모델 패키지 그룹 생성
sm_client = boto3.client('sagemaker')

sm_client.create_model_package_group(
    ModelPackageGroupName='fraud-detection-models',
    ModelPackageGroupDescription='Fraud detection model versions'
)

# 모델 등록
model_package = ModelPackage(
    role=role,
    model_data=model_data,
    image_uri=container,
    content_types=['text/csv'],
    response_types=['text/csv'],
    inference_instances=['ml.m5.large'],
    transform_instances=['ml.m5.large']
)

model_package.register(
    content_types=['text/csv'],
    response_types=['text/csv'],
    inference_instances=['ml.m5.large'],
    model_package_group_name='fraud-detection-models',
    approval_status='PendingManualApproval',
    description='Fraud detection model v3'
)

# 모델 승인
sm_client.update_model_package(
    ModelPackageArn='arn:aws:sagemaker:...:model-package/fraud-detection-models/3',
    ModelApprovalStatus='Approved'
)

# 승인된 모델 배포
approved_model = ModelPackage(
    role=role,
    model_package_arn='arn:aws:sagemaker:...:model-package/fraud-detection-models/3'
)

predictor = approved_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)
```

---

### Day 72: SageMaker Model Monitor

**학습 목표**: 모델 모니터링 설정

**Model Monitor 유형**:

| 유형 | 감지 대상 | 사용 시기 |
|------|---------|---------|
| **Data Quality** | 입력 데이터 드리프트 | 항상 |
| **Model Quality** | 성능 저하 | Ground Truth 있을 때 |
| **Bias Drift** | 편향 변화 | 공정성 필요 시 |
| **Feature Attribution** | 특성 중요도 변화 | 설명 가능성 필요 |

**Data Quality Monitor**:
```python
from sagemaker.model_monitor import DataQualityMonitor, DefaultModelMonitor

# 기준선 생성
data_quality_monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    volume_size_in_gb=20
)

# 기준선 생성 작업
data_quality_monitor.suggest_baseline(
    baseline_dataset=f's3://{bucket}/baseline/train.csv',
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri=f's3://{bucket}/baseline/output/'
)

# 모니터링 스케줄 생성
from sagemaker.model_monitor import CronExpressionGenerator

data_quality_monitor.create_monitoring_schedule(
    monitor_schedule_name='data-quality-schedule',
    endpoint_input=endpoint_name,
    output_s3_uri=f's3://{bucket}/monitoring/output/',
    statistics=data_quality_monitor.baseline_statistics(),
    constraints=data_quality_monitor.suggested_constraints(),
    schedule_cron_expression=CronExpressionGenerator.hourly()
)
```

**Model Quality Monitor (성능 모니터링)**:
```python
from sagemaker.model_monitor import ModelQualityMonitor

model_quality_monitor = ModelQualityMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge'
)

# Ground Truth 데이터와 비교
model_quality_monitor.create_monitoring_schedule(
    monitor_schedule_name='model-quality-schedule',
    endpoint_input=EndpointInput(
        endpoint_name=endpoint_name,
        destination='/opt/ml/processing/input',
        inference_attribute='prediction',
        probability_attribute='probability'
    ),
    ground_truth_input=f's3://{bucket}/ground-truth/',
    problem_type='BinaryClassification',
    output_s3_uri=f's3://{bucket}/model-quality/',
    schedule_cron_expression=CronExpressionGenerator.daily()
)
```

---

### Day 73: SageMaker Clarify (편향 탐지)

**학습 목표**: 모델 공정성 및 설명 가능성

**Clarify 기능**:

| 기능 | 설명 |
|------|------|
| **Pre-training Bias** | 데이터 편향 분석 |
| **Post-training Bias** | 모델 예측 편향 |
| **Feature Attribution** | SHAP 기반 설명 |

**편향 분석 코드**:
```python
from sagemaker.clarify import (
    SageMakerClarifyProcessor,
    DataConfig,
    BiasConfig,
    ModelConfig,
    SHAPConfig
)

clarify_processor = SageMakerClarifyProcessor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge'
)

# 데이터 설정
data_config = DataConfig(
    s3_data_input_path=f's3://{bucket}/data/test.csv',
    s3_output_path=f's3://{bucket}/clarify/output/',
    label='target',
    headers=['feature1', 'feature2', ..., 'target'],
    dataset_type='text/csv'
)

# 편향 설정
bias_config = BiasConfig(
    label_values_or_threshold=[1],
    facet_name='gender',
    facet_values_or_threshold=[0]  # 여성 = 0
)

# 모델 설정
model_config = ModelConfig(
    model_name='fraud-detection-model',
    instance_type='ml.m5.large',
    instance_count=1,
    content_type='text/csv',
    accept_type='text/csv'
)

# 편향 분석 실행
clarify_processor.run_bias(
    data_config=data_config,
    bias_config=bias_config,
    model_config=model_config,
    pre_training_methods='all',
    post_training_methods='all'
)
```

**SHAP 분석 (특성 중요도)**:
```python
shap_config = SHAPConfig(
    baseline=[baseline_data],
    num_samples=500,
    agg_method='mean_abs'
)

clarify_processor.run_explainability(
    data_config=data_config,
    model_config=model_config,
    explainability_config=shap_config
)
```

**주요 편향 메트릭**:

| 메트릭 | 설명 | 이상적 값 |
|--------|------|---------|
| **Class Imbalance (CI)** | 클래스 불균형 | 0 |
| **Difference in Proportions (DPL)** | 라벨 비율 차이 | 0 |
| **Disparate Impact (DI)** | 불균형 영향 | 1 |
| **Conditional Demographic Disparity (CDD)** | 조건부 차이 | 0 |

---

### Day 74: 보안 Best Practices

**학습 목표**: ML 워크로드 보안

**보안 계층**:

```
┌─────────────────────────────────────────────────────────────┐
│                        Security Layers                       │
├─────────────────────────────────────────────────────────────┤
│  IAM: 역할 기반 접근 제어                                    │
├─────────────────────────────────────────────────────────────┤
│  VPC: 네트워크 격리                                          │
├─────────────────────────────────────────────────────────────┤
│  Encryption: 저장/전송 암호화                                │
├─────────────────────────────────────────────────────────────┤
│  Logging: CloudTrail, CloudWatch                            │
└─────────────────────────────────────────────────────────────┘
```

**IAM 최소 권한 정책**:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:CreateProcessingJob",
        "sagemaker:CreateEndpoint"
      ],
      "Resource": "arn:aws:sagemaker:*:*:*",
      "Condition": {
        "StringEquals": {
          "sagemaker:VpcSecurityGroupIds": ["sg-xxx"],
          "sagemaker:VpcSubnets": ["subnet-xxx"]
        }
      }
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::my-ml-bucket/*"
    }
  ]
}
```

**VPC 설정**:
```python
from sagemaker.estimator import Estimator

estimator = Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',

    # VPC 설정
    subnets=['subnet-xxx', 'subnet-yyy'],
    security_group_ids=['sg-xxx'],

    # 네트워크 격리 (인터넷 차단)
    enable_network_isolation=True,

    # 암호화
    output_kms_key='arn:aws:kms:...:key/xxx',
    volume_kms_key='arn:aws:kms:...:key/xxx',

    # 컨테이너 간 암호화
    encrypt_inter_container_traffic=True
)
```

**VPC Endpoint (Private Link)**:

| 엔드포인트 | 서비스 |
|-----------|--------|
| `com.amazonaws.region.sagemaker.api` | SageMaker API |
| `com.amazonaws.region.sagemaker.runtime` | 추론 API |
| `com.amazonaws.region.s3` | S3 접근 |
| `com.amazonaws.region.ecr.api` | ECR 이미지 |

---

### Day 75: 비용 최적화

**학습 목표**: ML 비용 최적화 전략

**비용 최적화 기법**:

| 기법 | 절감률 | 적용 대상 |
|------|-------|---------|
| **Spot Instance** | 70-90% | Training |
| **Savings Plans** | 최대 64% | 상시 사용 |
| **Reserved Instance** | 최대 75% | Endpoint |
| **Graviton** | 40% | CPU 워크로드 |
| **Elastic Inference** | 75% | GPU 부분 사용 |
| **Multi-Model Endpoint** | 80%+ | 다중 모델 |
| **Serverless** | 가변 | 간헐적 트래픽 |

**Spot Training**:
```python
from sagemaker.estimator import Estimator

estimator = Estimator(
    image_uri=container,
    role=role,
    instance_count=2,
    instance_type='ml.p3.2xlarge',

    # Spot 설정
    use_spot_instances=True,
    max_wait=7200,  # 최대 대기 시간 (초)
    max_run=3600,   # 최대 실행 시간 (초)

    # 체크포인트 (Spot 중단 대비)
    checkpoint_s3_uri=f's3://{bucket}/checkpoints/'
)
```

**비용 모니터링**:
```python
import boto3

ce = boto3.client('ce')

# SageMaker 비용 조회
response = ce.get_cost_and_usage(
    TimePeriod={
        'Start': '2024-01-01',
        'End': '2024-01-31'
    },
    Granularity='MONTHLY',
    Metrics=['UnblendedCost'],
    Filter={
        'Dimensions': {
            'Key': 'SERVICE',
            'Values': ['Amazon SageMaker']
        }
    },
    GroupBy=[
        {'Type': 'DIMENSION', 'Key': 'USAGE_TYPE'}
    ]
)

for group in response['ResultsByTime'][0]['Groups']:
    print(f"{group['Keys'][0]}: ${group['Metrics']['UnblendedCost']['Amount']}")
```

---

### Day 76-77: 주말 종합 실습

**프로젝트**: 완전한 MLOps 파이프라인

```
┌─────────────────────────────────────────────────────────────┐
│                    End-to-End MLOps                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Data → Preprocess → Train → Evaluate → Register → Deploy   │
│                                   │                          │
│                            Model Monitor                     │
│                                   │                          │
│                         Clarify (Bias/Explain)               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**체크리스트**:
- [ ] SageMaker Pipeline 구축
- [ ] 조건부 모델 등록
- [ ] Model Registry 설정
- [ ] 자동 배포 (승인 후)
- [ ] Data Quality Monitor 설정
- [ ] Model Quality Monitor 설정
- [ ] Clarify 편향 분석
- [ ] 비용 대시보드 생성
- [ ] VPC 보안 설정

---

## 10-11주차 퀴즈 (자가 점검)

### Q1. Real-time Endpoint vs Serverless Inference 선택 기준은?
<details>
<summary>정답 보기</summary>

- **Real-time**: 밀리초 지연, 상시 트래픽, Cold Start 불가
- **Serverless**: 간헐적 트래픽, 비용 최적화, Cold Start 허용
</details>

### Q2. SageMaker Model Monitor의 4가지 유형은?
<details>
<summary>정답 보기</summary>

1. **Data Quality**: 입력 데이터 드리프트
2. **Model Quality**: 성능 저하 (Ground Truth 필요)
3. **Bias Drift**: 편향 변화
4. **Feature Attribution**: 특성 중요도 변화
</details>

### Q3. Spot Instance로 학습 시 필수 설정은?
<details>
<summary>정답 보기</summary>

- `use_spot_instances=True`
- `max_wait`: 최대 대기 시간
- `checkpoint_s3_uri`: 체크포인트 저장 (중단 대비)
</details>

### Q4. SageMaker VPC에서 인터넷 없이 S3 접근 방법은?
<details>
<summary>정답 보기</summary>

**VPC Endpoint (Gateway)** 생성:
- `com.amazonaws.region.s3`
</details>

### Q5. A/B 테스트에서 트래픽 분배 방법은?
<details>
<summary>정답 보기</summary>

**Production Variants**:
- `initial_weight` 파라미터로 트래픽 비율 설정
- 예: Model A = 90%, Model B = 10%
</details>

---

## 핵심 서비스 요약표

| 서비스 | 용도 | 시험 빈도 |
|--------|------|---------|
| **Real-time Endpoint** | 실시간 추론 | ★★★★★ |
| **Batch Transform** | 배치 추론 | ★★★★ |
| **Serverless Inference** | 간헐적 트래픽 | ★★★★ |
| **Multi-Model Endpoint** | 다중 모델 | ★★★ |
| **Pipelines** | ML 워크플로우 | ★★★★★ |
| **Model Registry** | 모델 버전 관리 | ★★★★ |
| **Model Monitor** | 드리프트 감지 | ★★★★★ |
| **Clarify** | 편향/설명 | ★★★★ |
| **Auto Scaling** | 자동 확장 | ★★★★ |
| **Spot Training** | 비용 절감 | ★★★★ |

---

## 다음 주 예고 (12주차)

- 모의고사 3회 실전 풀이
- 오답 정리 및 취약점 보완
- 시험 팁 및 전략
- 최종 복습
